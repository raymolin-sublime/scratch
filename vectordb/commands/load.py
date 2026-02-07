"""Load command for importing embeddings into PostgreSQL."""
from io import StringIO

import h5py
import psycopg


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

    # Connect to PostgreSQL
    conninfo = f"host={args.host} port={args.port} dbname={args.database} user={args.user} password={args.password}"
    print(f"Connecting to PostgreSQL at {args.host}:{args.port}/{args.database}")

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            # Enable pgvector extension
            print("Enabling pgvector extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
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

    print(f"Done! Inserted {num_vectors} embeddings into {args.table}")


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
    parser.set_defaults(func=execute)
