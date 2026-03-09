"""Insert or delete vectors to degrade an existing HNSW index."""

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import psycopg
from tqdm import tqdm

from commands.common import add_common_args, build_conninfo

_thread_local = threading.local()


def _get_conn(conninfo: str) -> psycopg.Connection:
    if not hasattr(_thread_local, "conn") or _thread_local.conn.closed:
        _thread_local.conn = psycopg.connect(conninfo, autocommit=True)
    return _thread_local.conn


def _insert_row(conninfo: str, table: str, text: str, vector_str: str) -> None:
    conn = _get_conn(conninfo)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {table} (text, embedding) VALUES (%s, %s)",
            (text, vector_str),
        )


def _delete_rows(conninfo: str, table: str, num_vectors: int, seed: int) -> None:
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {table}")
            all_ids = [row[0] for row in cur.fetchall()]

    if num_vectors > len(all_ids):
        print(f"Requested {num_vectors} deletions but only {len(all_ids)} rows exist, deleting all")
        num_vectors = len(all_ids)

    random.seed(seed)
    ids_to_delete = random.sample(all_ids, num_vectors)

    print(f"Deleting {num_vectors} rows (seed={seed}, {len(all_ids)} total rows)")
    start = time.monotonic()

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table} WHERE id = ANY(%s)", (ids_to_delete,))
        conn.commit()

    elapsed = time.monotonic() - start
    print(f"Done: {num_vectors} rows deleted in {elapsed:.2f}s")


def _insert_rows(conninfo: str, table: str, args) -> None:
    with h5py.File(args.input, "r") as f:
        total_available = int(f.attrs["num_vectors"])
        offset = args.offset or 0
        remaining = total_available - offset
        if remaining <= 0:
            print(f"Offset {offset} is beyond dataset size {total_available}")
            return
        num_vectors = min(args.num_vectors, remaining) if args.num_vectors else remaining
        embeddings = f["embeddings"][offset : offset + num_vectors]
        texts = f["texts"][offset : offset + num_vectors].astype(str)

    max_workers = max(args.qps * 2, 4)
    print(f"Inserting {num_vectors} vectors row-by-row (target {args.qps} rows/sec, {max_workers} workers)")
    start = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i in range(num_vectors):
            target_time = start + i / args.qps
            sleep_time = target_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

            vector_str = "[" + ",".join(map(str, embeddings[i])) + "]"
            futures.append(pool.submit(_insert_row, conninfo, args.table, texts[i], vector_str))

        for f in tqdm(as_completed(futures), total=num_vectors, desc="Inserting"):
            f.result()

    elapsed = time.monotonic() - start
    print(f"Done: {num_vectors} rows in {elapsed:.2f}s ({num_vectors / elapsed:.1f} rows/sec)")


def execute(args) -> None:
    conninfo = build_conninfo(args)

    if args.operation == "delete":
        if args.num_vectors is None:
            print("--num-vectors is required for delete operation")
            return
        _delete_rows(conninfo, args.table, args.num_vectors, args.seed)
    elif args.operation == "insert":
        if args.input is None:
            print("--input is required for insert operation")
            return
        _insert_rows(conninfo, args.table, args)
    else:
        raise ValueError(f"Unsupported operation: {args.operation}")


def register_degrade_command(subparsers) -> None:
    parser = subparsers.add_parser("degrade", help="Insert vectors row-by-row to degrade an HNSW index")
    parser.add_argument("--input", default=None, help="Path to HDF5 dataset (required for insert)")
    parser.add_argument("--num-vectors", type=int, default=None, help="Number of vectors to insert/delete (default: all for insert, required for delete)")
    parser.add_argument("--offset", type=int, default=None, help="Starting index in the HDF5 dataset (default: 0)")
    parser.add_argument("--qps", type=int, default=1, help="Target rows per second (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible deletes (default: 42)")
    parser.add_argument("--operation", choices=["insert", "delete"], default="insert", help="Operation to perform (default: insert)")
    add_common_args(parser)
    parser.set_defaults(func=execute)
