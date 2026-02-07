"""Query command for finding nearest neighbors."""
import random
import string

import psycopg
from sentence_transformers import SentenceTransformer


def generate_random_text(min_length=100, max_length=500):
    """Generate random ASCII text of specified length range."""
    length = random.randint(min_length, max_length)
    chars = string.ascii_letters + string.digits + string.punctuation + ' ' * 10
    return ''.join(random.choice(chars) for _ in range(length))


def execute(args):
    """Execute the query command."""
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Generate or use provided query text
    if args.query_text:
        query_text = args.query_text
        print(f"Using provided query text: {query_text[:100]}...")
    else:
        query_text = generate_random_text()
        print(f"Generated random query text: {query_text[:100]}...")

    # Load model and create embedding
    print(f"Loading BGE model...")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')

    print(f"Creating embedding...")
    query_embedding = model.encode(query_text, convert_to_numpy=True)

    # Format embedding as vector string for PostgreSQL
    vector_str = '[' + ','.join(map(str, query_embedding)) + ']'

    # Connect to PostgreSQL
    conninfo = f"host={args.host} port={args.port} dbname={args.database} user={args.user} password={args.password}"
    print(f"Connecting to PostgreSQL at {args.host}:{args.port}/{args.database}")

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            # Query nearest neighbors using cosine distance
            print(f"Querying {args.neighbors} nearest neighbors...")
            cur.execute(f"""
                SELECT
                    id,
                    text,
                    embedding <=> %s::vector AS distance
                FROM {args.table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (vector_str, vector_str, args.neighbors))

            results = cur.fetchall()

    # Display results
    print(f"\n{'='*80}")
    print(f"Found {len(results)} nearest neighbors:")
    print(f"{'='*80}\n")

    for idx, (id, text, distance) in enumerate(results, 1):
        print(f"Rank {idx}:")
        print(f"  ID: {id}")
        print(f"  Cosine Distance: {distance:.6f}")
        print(f"  Text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print()


def register_query_command(subparsers):
    """Register the query subcommand."""
    parser = subparsers.add_parser(
        'query',
        help='Query nearest neighbors using text embedding'
    )
    parser.add_argument(
        '-n', '--neighbors',
        type=int,
        default=10,
        help='Number of nearest neighbors to retrieve (default: 10)'
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
        '--query-text',
        type=str,
        default=None,
        help='Query text (default: generate random text)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for text generation'
    )
    parser.set_defaults(func=execute)
