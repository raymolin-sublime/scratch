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


def _snapshot_buffers(cur, table_name):
    """Query pg_buffercache for buffers belonging to the given table."""
    cur.execute("""
        SELECT count(*) AS buffer_count,
               pg_size_pretty(count(*) * current_setting('block_size')::bigint) AS total_size
        FROM pg_buffercache b
        JOIN pg_class c ON c.relfilenode = b.relfilenode
        WHERE c.relname = %s
    """, (table_name,))
    row = cur.fetchone()
    return {'buffer_count': row[0], 'total_size': row[1]}


def execute(args):
    """Execute the load command."""
    # Read HDF5 file
    print(f"Reading HDF5 file: {args.input}")
    with h5py.File(args.input, 'r') as f:
        embeddings = f['embeddings'][:]
        texts = f['texts'][:].astype(str)
        num_vectors = f.attrs['num_vectors']
        embedding_dim = f.attrs['embedding_dim']

    print(f"Loaded {num_vectors} vectors with dimension {embedding_dim}")

    # Start docker stats polling thread
    db_samples = []
    stop_event = threading.Event()
    poll_thread = threading.Thread(
        target=_poll_docker_stats,
        args=(args.container, db_samples, stop_event),
        daemon=True,
    )
    poll_thread.start()

    # Wait for at least one sample before proceeding
    deadline = time.monotonic() + 15
    while not db_samples and time.monotonic() < deadline:
        time.sleep(0.1)

    # Connect to PostgreSQL
    conninfo = f"host={args.host} port={args.port} dbname={args.database} user={args.user} password={args.password}"
    print(f"Connecting to PostgreSQL at {args.host}:{args.port}/{args.database}")

    buffer_snapshots = {}

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            # Enable extensions
            print("Enabling pgvector extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_buffercache")
            conn.commit()

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

    # Stop docker stats polling
    stop_event.set()
    poll_thread.join()

    # Build stats JSON
    stats = {
        'num_vectors': int(num_vectors),
        'embedding_dim': int(embedding_dim),
        'table': args.table,
        'buffer_snapshots': buffer_snapshots,
    }

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
        '--stats-file',
        type=str,
        default=None,
        help='File path to append stats JSON (one object per line)'
    )
    parser.set_defaults(func=execute)
