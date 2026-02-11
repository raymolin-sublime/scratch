"""Load command for importing embeddings into PostgreSQL."""
import json
import subprocess
import threading
import time
from io import StringIO

import h5py
import psycopg


def _poll_docker_stats(container, samples, stop_event):
    """Background thread that polls docker stats every 2 seconds."""
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                ['docker', 'stats', container, '--no-stream',
                 '--format', '{{.CPUPerc}}\t{{.MemPerc}}'],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Output looks like "1.23%\t4.56%"
                parts = result.stdout.strip().split('\t')
                cpu = float(parts[0].strip('%'))
                mem = float(parts[1].strip('%'))
                samples.append({'cpu_pct': cpu, 'mem_pct': mem})
        except Exception:
            pass
        stop_event.wait(2)


def _poll_shared_buffers(conninfo, table_name, samples, stop_event):
    """Background thread that polls shared buffer usage every 2 seconds."""
    try:
        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                while not stop_event.is_set():
                    try:
                        snap = _snapshot_buffers(cur, table_name)
                        snap['ts'] = time.monotonic()
                        samples.append(snap)
                    except Exception:
                        pass
                    stop_event.wait(2)
    except Exception:
        pass


def _snapshot_buffers(cur, table_name):
    """Query pg_buffercache for buffers belonging to the given table."""
    cur.execute("""
        SELECT count(*) AS buffer_count,
               count(*) * current_setting('block_size')::bigint AS total_bytes,
               pg_size_pretty(count(*) * current_setting('block_size')::bigint) AS total_size
        FROM pg_buffercache b
        JOIN pg_class c ON c.relfilenode = b.relfilenode
        WHERE c.relname = %s
    """, (table_name,))
    row = cur.fetchone()
    return {'buffer_count': row[0], 'total_bytes': row[1], 'total_size': row[2]}


def execute(args):
    """Execute the load command."""
    # Validate --num-vectors
    if args.num_vectors is not None and args.num_vectors <= 0:
        print("Error: --num-vectors must be a positive integer")
        return

    # Read HDF5 file
    print(f"Reading HDF5 file: {args.input}")
    with h5py.File(args.input, 'r') as f:
        total_vectors = int(f.attrs['num_vectors'])
        embedding_dim = int(f.attrs['embedding_dim'])

        num_vectors = min(args.num_vectors, total_vectors) if args.num_vectors is not None else total_vectors

        embeddings = f['embeddings'][:num_vectors]
        texts = f['texts'][:num_vectors].astype(str)

    if args.num_vectors is not None and args.num_vectors > total_vectors:
        print(f"Requested {args.num_vectors} vectors but file only contains {total_vectors}, loading all")

    print(f"Loaded {num_vectors} vectors with dimension {embedding_dim} (file contains {total_vectors})")

    # Start docker stats polling thread
    db_samples = []
    stop_event = threading.Event()
    poll_thread = threading.Thread(
        target=_poll_docker_stats,
        args=(args.container, db_samples, stop_event),
        daemon=True,
    )
    poll_thread.start()

    # Start shared buffer polling thread
    conninfo = f"host={args.host} port={args.port} dbname={args.database} user={args.user} password={args.password}"
    buffer_samples = []
    buffer_thread = threading.Thread(
        target=_poll_shared_buffers,
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
                cur.execute(psycopg.sql.SQL("SET maintenance_work_mem = {}").format(
                    psycopg.sql.Literal(args.maintenance_work_mem)
                ))
                conn.commit()

            # Query configured shared_buffers size in bytes
            cur.execute("""
                SELECT setting::bigint * current_setting('block_size')::bigint
                FROM pg_settings WHERE name = 'shared_buffers'
            """)
            shared_buffers_total_bytes = cur.fetchone()[0]

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
            buffer_snapshots['before_load'] = _snapshot_buffers(cur, args.table)

            # Prepare data for COPY
            print(f"Preparing data for COPY...")
            buffer = StringIO()
            for text, embedding in zip(texts, embeddings):
                # Escape text for tab-delimited format
                text_escaped = text.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')
                # Format vector as [1,2,3,...]
                vector_str = '[' + ','.join(map(str, embedding)) + ']'
                buffer.write(f"{text_escaped}\t{vector_str}\n")

            buffer.seek(0)

            # Use COPY to bulk insert
            print(f"Inserting {num_vectors} embeddings using COPY...")
            with cur.copy(f"COPY {args.table} (text, embedding) FROM STDIN") as copy:
                copy.write(buffer.read())

            conn.commit()

            # Snapshot buffers after load
            buffer_snapshots['after_load'] = _snapshot_buffers(cur, args.table)

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
                buffer_snapshots['after_index'] = _snapshot_buffers(cur, args.table)

            # Collect table and index sizes
            cur.execute("""
                SELECT pg_table_size(%s::regclass)::bigint,
                       pg_indexes_size(%s::regclass)::bigint
            """, (args.table, args.table))
            table_size_bytes, index_size_bytes = cur.fetchone()

    # Stop polling threads
    stop_event.set()
    poll_thread.join()
    buffer_thread.join()

    # Build stats JSON
    stats = {
        'num_vectors': int(num_vectors),
        'embedding_dim': int(embedding_dim),
        'table': args.table,
        'table_size_bytes': table_size_bytes,
        'index_size_bytes': index_size_bytes,
        'buffer_snapshots': buffer_snapshots,
    }

    if args.maintenance_work_mem:
        stats['maintenance_work_mem'] = args.maintenance_work_mem

    if args.create_index:
        stats['index_creation_time_s'] = round(index_elapsed, 3)

    if db_samples:
        cpu_vals = [s['cpu_pct'] for s in db_samples]
        mem_vals = [s['mem_pct'] for s in db_samples]
        stats['db_stats'] = {
            'container': args.container,
            'samples': len(db_samples),
            'cpu_pct': {
                'avg': round(sum(cpu_vals) / len(cpu_vals), 2),
                'max': round(max(cpu_vals), 2),
            },
            'mem_pct': {
                'avg': round(sum(mem_vals) / len(mem_vals), 2),
                'max': round(max(mem_vals), 2),
            },
        }

    if buffer_samples:
        buf_bytes = [s['total_bytes'] for s in buffer_samples]
        avg_bytes = round(sum(buf_bytes) / len(buf_bytes))
        max_bytes = max(buf_bytes)
        stats['shared_buffers'] = {
            'configured_bytes': shared_buffers_total_bytes,
            'samples': len(buffer_samples),
            'bytes': {
                'avg': avg_bytes,
                'max': max_bytes,
            },
            'utilization_pct': {
                'avg': round(avg_bytes / shared_buffers_total_bytes * 100, 2),
                'max': round(max_bytes / shared_buffers_total_bytes * 100, 2),
            },
        }

    print(f"\nDone! Inserted {num_vectors} embeddings into {args.table}")
    print(json.dumps(stats, indent=2))

    # Append to stats file
    if args.stats_file:
        with open(args.stats_file, 'a') as f:
            f.write(json.dumps(stats) + '\n')
        print(f"Stats appended to {args.stats_file}")


def register_load_command(subparsers):
    """Register the load subcommand."""
    parser = subparsers.add_parser(
        'load',
        help='Load embeddings from HDF5 into PostgreSQL'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input HDF5 file path'
    )
    parser.add_argument(
        '-n', '--num-vectors',
        type=int,
        default=None,
        help='Number of vectors to load from the HDF5 file (default: all)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='PostgreSQL host (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5432,
        help='PostgreSQL port (default: 5432)'
    )
    parser.add_argument(
        '--database',
        type=str,
        default='postgres',
        help='Database name (default: postgres)'
    )
    parser.add_argument(
        '--user',
        type=str,
        default='postgres',
        help='Database user (default: postgres)'
    )
    parser.add_argument(
        '--password',
        type=str,
        default='postgres',
        help='Database password (default: postgres)'
    )
    parser.add_argument(
        '--table',
        type=str,
        default='embeddings',
        help='Table name (default: embeddings)'
    )
    parser.add_argument(
        '--create-table',
        action='store_true',
        help='Create table if it does not exist'
    )
    parser.add_argument(
        '--drop-table',
        action='store_true',
        help='Drop table if it exists before creating'
    )
    parser.add_argument(
        '--create-index',
        action='store_true',
        help='Create HNSW index after loading'
    )
    parser.add_argument(
        '--container',
        type=str,
        default='pgvector-db',
        help='Docker container name for RAM polling (default: pgvector-db)'
    )
    parser.add_argument(
        '--maintenance-work-mem',
        type=str,
        default=None,
        help='Set maintenance_work_mem for index creation (e.g. 1GB, 512MB)'
    )
    parser.add_argument(
        '--stats-file',
        type=str,
        default=None,
        help='File path to append stats JSON (one object per line)'
    )
    parser.set_defaults(func=execute)
