"""Load command for importing embeddings into PostgreSQL."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

import h5py
import numpy as np
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


def _format_chunk(texts_chunk, embeddings_chunk):
    """Format a chunk of texts and embeddings into tab-delimited COPY format."""
    buf = StringIO()
    for text, embedding in zip(texts_chunk, embeddings_chunk):
        text_escaped = (
            text.replace("\\", "\\\\")
            .replace("\t", "\\t")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )
        vector_str = "[" + ",".join(map(str, embedding)) + "]"
        buf.write(f"{text_escaped}\t{vector_str}\n")
    return buf.getvalue()


def execute(args):
    """Execute the load command."""
    # Validate --num-vectors
    if args.num_vectors is not None and args.num_vectors <= 0:
        print("Error: --num-vectors must be a positive integer")
        return

    # Read HDF5 file
    print(f"Reading HDF5 file: {args.input}")
    with h5py.File(args.input, "r") as f:
        total_vectors = int(f.attrs["num_vectors"])
        embedding_dim = int(f.attrs["embedding_dim"])

        num_vectors = (
            min(args.num_vectors, total_vectors)
            if args.num_vectors is not None
            else total_vectors
        )

        embeddings = f["embeddings"][:num_vectors]
        texts = f["texts"][:num_vectors].astype(str)

    if args.num_vectors is not None and args.num_vectors > total_vectors:
        print(
            f"Requested {args.num_vectors} vectors but file only contains {total_vectors}, loading all"
        )

    print(
        f"Loaded {num_vectors} vectors with dimension {embedding_dim} (file contains {total_vectors})"
    )

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
    conninfo = build_conninfo(args)
    buffer_samples = []
    buffer_thread = threading.Thread(
        target=poll_shared_buffers,
        args=(conninfo, args.table, buffer_samples, stop_event),
        daemon=True,
    )

    # Wait for at least one sample before proceeding
    deadline = time.monotonic() + 15
    while not db_samples and time.monotonic() < deadline:
        time.sleep(0.1)

    # Connect to PostgreSQL
    print(f"Connecting to PostgreSQL at {args.host}:{args.port}/{args.database}")

    buffer_snapshots = {}

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            # Enable extensions
            print("Enabling pgvector extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_buffercache")
            conn.commit()

            # Set maintenance_work_mem if specified
            if args.maintenance_work_mem:
                print(f"Setting maintenance_work_mem = {args.maintenance_work_mem}")
                cur.execute(
                    psycopg.sql.SQL("SET maintenance_work_mem = {}").format(
                        psycopg.sql.Literal(args.maintenance_work_mem)
                    )
                )
                conn.commit()

            buffer_thread.start()

            # Drop table if requested
            if args.drop_table:
                print(f"Dropping table {args.table} if exists...")
                cur.execute(f"DROP TABLE IF EXISTS {args.table}")
                conn.commit()

            # Create table if requested
            if args.create_table:
                print(f"Creating table {args.table}...")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {args.table} (
                        id SERIAL PRIMARY KEY,
                        text TEXT,
                        embedding VECTOR({embedding_dim})
                    )
                """)
                conn.commit()

            # Snapshot buffers before load
            buffer_snapshots["before_load"] = snapshot_buffers(cur, args.table)

            # Prepare data for COPY (parallel)
            print(f"Preparing data for COPY...")
            num_chunks = 4
            text_chunks = np.array_split(texts, num_chunks)
            embedding_chunks = np.array_split(embeddings, num_chunks)

            with ThreadPoolExecutor(max_workers=num_chunks) as pool:
                chunk_results = list(
                    pool.map(_format_chunk, text_chunks, embedding_chunks)
                )

            buffer = StringIO()
            for chunk in chunk_results:
                buffer.write(chunk)
            buffer.seek(0)

            # Use COPY to bulk insert
            print(f"Inserting {num_vectors} embeddings using COPY...")
            with cur.copy(f"COPY {args.table} (text, embedding) FROM STDIN") as copy:
                copy.write(buffer.read())

            conn.commit()

            # Snapshot buffers after load
            buffer_snapshots["after_load"] = snapshot_buffers(cur, args.table)

            # Create index if requested
            if args.create_index:
                print("Creating HNSW index...")
                t0 = time.perf_counter()
                cur.execute(f"""
                    CREATE INDEX ON {args.table}
                    USING hnsw (embedding vector_cosine_ops)
                """)
                conn.commit()
                index_elapsed = time.perf_counter() - t0
                print(f"Index creation time: {index_elapsed:.2f}s")

                # Snapshot buffers after index creation
                buffer_snapshots["after_index"] = snapshot_buffers(cur, args.table)

            # Collect table and index sizes
            cur.execute(
                """
                SELECT pg_table_size(%s::regclass)::bigint,
                       pg_indexes_size(%s::regclass)::bigint
            """,
                (args.table, args.table),
            )
            table_size_bytes, index_size_bytes = cur.fetchone()

    # Stop polling threads
    stop_event.set()
    poll_thread.join()
    buffer_thread.join()

    # Build stats JSON
    stats = {
        "num_vectors": int(num_vectors),
        "embedding_dim": int(embedding_dim),
        "table": args.table,
        "table_size_bytes": table_size_bytes,
        "index_size_bytes": index_size_bytes,
        "buffer_snapshots": buffer_snapshots,
    }

    if args.maintenance_work_mem:
        stats["maintenance_work_mem"] = args.maintenance_work_mem

    if args.create_index:
        stats["index_creation_time_s"] = round(index_elapsed, 3)

    stats.update(summarize_stats(db_samples, buffer_samples, args.container, conninfo))

    print(f"\nDone! Inserted {num_vectors} embeddings into {args.table}")
    print(json.dumps(stats, indent=2))

    if args.stats_file:
        write_stats(stats, args.stats_file)


def register_load_command(subparsers):
    """Register the load subcommand."""
    parser = subparsers.add_parser(
        "load", help="Load embeddings from HDF5 into PostgreSQL"
    )
    add_common_args(parser)

    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input HDF5 file path"
    )
    parser.add_argument(
        "-n",
        "--num-vectors",
        type=int,
        default=None,
        help="Number of vectors to load from the HDF5 file (default: all)",
    )
    parser.add_argument(
        "--create-table", action="store_true", help="Create table if it does not exist"
    )
    parser.add_argument(
        "--drop-table",
        action="store_true",
        help="Drop table if it exists before creating",
    )
    parser.add_argument(
        "--create-index", action="store_true", help="Create HNSW index after loading"
    )
    parser.add_argument(
        "--maintenance-work-mem",
        type=str,
        default=None,
        help="Set maintenance_work_mem for index creation (e.g. 1GB, 512MB)",
    )
    parser.set_defaults(func=execute)
