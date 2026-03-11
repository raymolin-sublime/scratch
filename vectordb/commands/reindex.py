"""Reindex command for rebuilding HNSW indexes after degradation."""

import json
import threading
import time

import psycopg

from .common import (
    add_common_args,
    build_conninfo,
    poll_docker_stats,
    poll_shared_buffers,
    snapshot_buffers,
    summarize_stats,
    write_stats,
)


def _find_hnsw_index(cur, table):
    """Find the HNSW index on the given table. Returns (indexname, indexdef) or None."""
    cur.execute(
        "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = %s AND indexdef LIKE '%%hnsw%%'",
        (table,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return row[0], row[1]


def _parse_index_params(indexdef):
    """Extract m and ef_construction from an existing index definition."""
    params = {}
    # e.g. WITH (m='16', ef_construction='64')
    if "WITH" in indexdef.upper():
        with_part = indexdef[indexdef.index("(", indexdef.upper().index("WITH")) :]
        with_part = with_part.strip("()")
        for part in with_part.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip().strip("'\"")
                v = v.strip().strip("'\"")
                params[k] = v
    return params


def _reindex_strategy(cur, conn, index_name, concurrently, autocommit):
    """Rebuild the index in-place using REINDEX."""
    concurrently_sql = "CONCURRENTLY " if concurrently else ""
    sql = f"REINDEX INDEX {concurrently_sql}{index_name}"
    print(f"Running: {sql}")
    cur.execute(sql)
    if not autocommit:
        conn.commit()


def _create_strategy(cur, conn, table, index_name, concurrently, autocommit, m, ef_construction):
    """Create a new index, drop the old one, and rename."""
    concurrently_sql = "CONCURRENTLY " if concurrently else ""
    temp_name = f"{index_name}_new"

    with_opts = []
    if m is not None:
        with_opts.append(f"m = {m}")
    if ef_construction is not None:
        with_opts.append(f"ef_construction = {ef_construction}")
    with_clause = f" WITH ({', '.join(with_opts)})" if with_opts else ""

    create_sql = (
        f"CREATE INDEX {concurrently_sql}{temp_name} "
        f"ON {table} USING hnsw (embedding vector_cosine_ops){with_clause}"
    )
    print(f"Running: {create_sql}")
    cur.execute(create_sql)
    if not autocommit:
        conn.commit()

    drop_sql = f"DROP INDEX {index_name}"
    print(f"Running: {drop_sql}")
    cur.execute(drop_sql)
    if not autocommit:
        conn.commit()

    rename_sql = f"ALTER INDEX {temp_name} RENAME TO {index_name}"
    print(f"Running: {rename_sql}")
    cur.execute(rename_sql)
    if not autocommit:
        conn.commit()


def execute(args):
    """Execute the reindex command."""
    conninfo = build_conninfo(args)
    autocommit = args.concurrently

    # Start docker stats polling thread
    db_samples = []
    stop_event = threading.Event()
    poll_thread = threading.Thread(
        target=poll_docker_stats,
        args=(args.container, db_samples, stop_event),
        daemon=True,
    )
    poll_thread.start()

    # Start shared buffer polling thread
    buffer_samples = []
    buffer_thread = threading.Thread(
        target=poll_shared_buffers,
        args=(conninfo, args.table, buffer_samples, stop_event),
        daemon=True,
    )

    # Wait for at least one docker stats sample
    deadline = time.monotonic() + 15
    while not db_samples and time.monotonic() < deadline:
        time.sleep(0.1)

    print(f"Connecting to PostgreSQL at {args.host}:{args.port}/{args.database}")

    buffer_snapshots = {}

    with psycopg.connect(conninfo, autocommit=autocommit) as conn:
        with conn.cursor() as cur:
            # Find existing HNSW index
            result = _find_hnsw_index(cur, args.table)
            if result is None:
                print(f"Error: no HNSW index found on table '{args.table}'")
                stop_event.set()
                poll_thread.join()
                return
            index_name, indexdef = result
            print(f"Found index: {index_name}")

            existing_params = _parse_index_params(indexdef)

            # Set maintenance_work_mem if specified
            if args.maintenance_work_mem:
                print(f"Setting maintenance_work_mem = {args.maintenance_work_mem}")
                cur.execute(
                    psycopg.sql.SQL("SET maintenance_work_mem = {}").format(
                        psycopg.sql.Literal(args.maintenance_work_mem)
                    )
                )
                if not autocommit:
                    conn.commit()

            buffer_thread.start()

            # Snapshot and sizes before reindex
            buffer_snapshots["before_reindex"] = snapshot_buffers(cur, args.table)
            cur.execute(
                "SELECT pg_table_size(%s::regclass)::bigint, pg_indexes_size(%s::regclass)::bigint",
                (args.table, args.table),
            )
            table_size_bytes, index_size_before = cur.fetchone()

            # Execute the chosen strategy
            print(f"Strategy: {args.strategy} (concurrently={args.concurrently})")
            t0 = time.perf_counter()

            if args.strategy == "reindex":
                _reindex_strategy(cur, conn, index_name, args.concurrently, autocommit)
            else:
                _create_strategy(
                    cur, conn, args.table, index_name,
                    args.concurrently, autocommit,
                    args.m, args.ef_construction,
                )

            reindex_time = time.perf_counter() - t0
            print(f"Reindex time: {reindex_time:.2f}s")

            # Snapshot and sizes after reindex
            buffer_snapshots["after_reindex"] = snapshot_buffers(cur, args.table)
            cur.execute(
                "SELECT pg_indexes_size(%s::regclass)::bigint",
                (args.table,),
            )
            index_size_after = cur.fetchone()[0]

    # Stop polling threads
    stop_event.set()
    poll_thread.join()
    buffer_thread.join()

    # Determine effective m/ef_construction
    if args.strategy == "create" and (args.m is not None or args.ef_construction is not None):
        effective_m = args.m if args.m is not None else existing_params.get("m")
        effective_ef = args.ef_construction if args.ef_construction is not None else existing_params.get("ef_construction")
    else:
        effective_m = existing_params.get("m")
        effective_ef = existing_params.get("ef_construction")

    # Build stats
    stats = {
        "strategy": args.strategy,
        "concurrently": args.concurrently,
        "reindex_time_s": round(reindex_time, 3),
        "table": args.table,
        "table_size_bytes": table_size_bytes,
        "index_name": index_name,
        "index_size_before_bytes": index_size_before,
        "index_size_after_bytes": index_size_after,
        "buffer_snapshots": buffer_snapshots,
    }

    if args.maintenance_work_mem:
        stats["maintenance_work_mem"] = args.maintenance_work_mem
    if effective_m is not None:
        stats["m"] = int(effective_m)
    if effective_ef is not None:
        stats["ef_construction"] = int(effective_ef)

    stats.update(summarize_stats(db_samples, buffer_samples, args.container, conninfo))

    print(f"\nDone! Reindexed {index_name} on {args.table}")
    print(json.dumps(stats, indent=2))

    if args.stats_file:
        write_stats(stats, args.stats_file)


def register_reindex_command(subparsers):
    """Register the reindex subcommand."""
    parser = subparsers.add_parser(
        "reindex", help="Rebuild HNSW index after degradation"
    )
    add_common_args(parser)

    parser.add_argument(
        "--strategy",
        choices=["reindex", "create"],
        required=True,
        help="Reindex strategy: 'reindex' for in-place REINDEX, 'create' for new index + swap",
    )
    parser.add_argument(
        "--concurrently",
        action="store_true",
        help="Use CONCURRENTLY variant (non-blocking but slower)",
    )
    parser.add_argument(
        "--maintenance-work-mem",
        type=str,
        default=None,
        help="Set maintenance_work_mem (e.g. 1GB, 512MB)",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="HNSW m parameter for create strategy (default: use existing)",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=None,
        help="HNSW ef_construction parameter for create strategy (default: use existing)",
    )
    parser.set_defaults(func=execute)
