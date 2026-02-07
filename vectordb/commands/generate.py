"""Generate command for creating random text embeddings."""
import random
import string
import time

import h5py
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def generate_random_text(min_length=100, max_length=500):
    """Generate random ASCII text of specified length range."""
    length = random.randint(min_length, max_length)
    chars = string.ascii_letters + string.digits + string.punctuation + ' ' * 10
    return ''.join(random.choice(chars) for _ in range(length))


def execute(args):
    """Execute the generate command."""
    # Set random seed
    if args.seed is None:
        seed = time.time_ns()
    else:
        seed = args.seed

    random.seed(seed)
    np.random.seed(seed % (2**32))
    print(f"Using random seed: {seed}")

    print(f"Loading BGE model: bge-large-en-v1.5...")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')

    print(f"Generating {args.num_vectors} random texts...")
    texts = []
    for _ in tqdm(range(args.num_vectors), desc="Generating texts"):
        texts.append(generate_random_text())

    print(f"Creating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"Saving to {args.output}...")
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('embeddings', data=embeddings)
        f.create_dataset('texts', data=np.array(texts, dtype=h5py.string_dtype()))
        f.attrs['num_vectors'] = args.num_vectors
        f.attrs['model'] = 'BAAI/bge-large-en-v1.5'
        f.attrs['embedding_dim'] = embeddings.shape[1]
        f.attrs['seed'] = seed

    print(f"Done! Generated {args.num_vectors} embeddings")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Output saved to: {args.output}")


def register_generate_command(subparsers):
    """Register the generate subcommand."""
    parser = subparsers.add_parser(
        'generate',
        help='Generate random text embeddings and save to HDF5'
    )
    parser.add_argument(
        '-n', '--num-vectors',
        type=int,
        default=1000,
        help='Number of vectors to generate (default: 1000)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output path for the HDF5 file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding (default: 32)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: current Unix nanosecond time)'
    )
    parser.set_defaults(func=execute)
