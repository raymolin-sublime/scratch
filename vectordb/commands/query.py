"""Query command for load-testing nearest neighbor queries with monitoring."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import numpy as np
import psycopg
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .common import (
    add_common_args,
    build_conninfo,
    poll_docker_stats,
    poll_shared_buffers,
    summarize_stats,
    write_stats,
)

if TYPE_CHECKING:
    SubParsersAction = argparse._SubParsersAction[argparse.ArgumentParser]  # type: ignore[type-arg]


class Embedding(NamedTuple):
    text: str
    vector: str
    raw: NDArray[np.float32]


class QueryRow(NamedTuple):
    id: int
    text: str
    distance: float


class QueryResult(NamedTuple):
    latency: float
    rows: List[QueryRow]
    recall: Optional[float] = None


class WindowResult(NamedTuple):
    window: float
    result: QueryResult


_thread_local = threading.local()


def _parse_period(period_str: str) -> int:
    """Parse a period string like '5m', '30s', '2h' to seconds."""
    match = re.fullmatch(r"(\d+)([smh])", period_str)
    if not match:
        raise ValueError(
            f"Invalid period format: {period_str}. Use e.g. '30s', '5m', '2h'"
        )
    value, unit = int(match.group(1)), match.group(2)
    return value * {"s": 1, "m": 60, "h": 3600}[unit]


def _latency_mstats(latencies: list[float]) -> dict[str, float]:
    """Compute latency percentile statistics."""
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    return {
        "avg": round(sum(sorted_lat) / n, 4),
        "p50": round(sorted_lat[n // 2], 4),
        "p95": round(sorted_lat[min(int(n * 0.95), n - 1)], 4),
        "p99": round(sorted_lat[min(int(n * 0.99), n - 1)], 4),
        "max": round(sorted_lat[-1], 4),
    }


def _run_single_query(
    conninfo: str,
    table: str,
    neighbors: int,
    vector_str: str,
    timeout_ms: float,
    expected_neighbors: Optional[List[str]] = None,
    ef_search: Optional[int] = None,
) -> QueryResult:
    """Run a single nearest-neighbor query. Returns (latency, rows, recall)."""
    if not hasattr(_thread_local, "conn") or _thread_local.conn.closed:
        _thread_local.conn = psycopg.connect(conninfo, autocommit=True)
        with _thread_local.conn.cursor() as cur:
            if timeout_ms:
                cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
            if ef_search is not None:
                cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")
    conn = _thread_local.conn

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        # <#> is negative dot product; smaller values are more similar
        cur.execute(
            f"""
            SELECT id, text, embedding <=> %s::vector AS distance
            FROM {table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """,
            (vector_str, vector_str, neighbors),
        )
        rows = [QueryRow(*r) for r in cur.fetchall()]
    latency_ms = (time.perf_counter() - t0) * 1000

    recall = None
    if expected_neighbors is not None and expected_neighbors:
        result_texts = {r.text for r in rows}
        truth_texts = set(expected_neighbors)
        recall = len(result_texts & truth_texts) / len(truth_texts)

    return QueryResult(latency_ms, rows, recall)


def _generate_random_text(min_length: int = 100, max_length: int = 500) -> str:
    """Generate random ASCII text of specified length range."""
    length = random.randint(min_length, max_length)
    chars = string.ascii_letters + string.digits + string.punctuation + " " * 10
    return "".join(random.choice(chars) for _ in range(length))


def _synthesize_dataset(size: int) -> List[Embedding]:
    print(f"Loading BGE model and pre-generating {size} query embedding(s)...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    texts = [_generate_random_text() for _ in range(size)]

    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=32)
    vectors = ["[" + ",".join(map(str, emb)) + "]" for emb in embeddings]

    return [
        Embedding(text, vector, emb)
        for text, vector, emb in zip(texts, vectors, embeddings)
    ]


def _load_dataset(path: str, num_samples: Optional[int] = None) -> List[Embedding]:
    import h5py

    with h5py.File(path, "r") as f:
        available = int(f.attrs["num_vectors"])
        n = min(num_samples, available) if num_samples is not None else available
        print(f"Loading {n} embeddings from {path} ({available} available)...")
        embeddings = f["embeddings"][:n]
        texts = f["texts"][:n].astype(str)

    vectors = ["[" + ",".join(map(str, emb)) + "]" for emb in embeddings]

    return [
        Embedding(text, vector, emb)
        for text, vector, emb in zip(texts, vectors, embeddings)
    ]


def _generate_stats(
    args: argparse.Namespace,
    results: List[WindowResult],
    db_samples: List,
    buffer_samples: List,
    conn_info: str,
    error_times: List[float],
    window_s: Optional[int],
    elapsed: float,
) -> List[dict]:
    latencies = [r.result.latency for r in results]
    errors = len(error_times)
    stats = {
        "qps_target": args.qps,
        "qps_actual": round(len(latencies) / elapsed, 2) if elapsed > 0 else 0,
        "period": args.period,
        "duration_s": round(elapsed, 2),
        "total_queries": len(latencies),
        "errors": errors,
        "neighbors": args.neighbors,
        "table": args.table,
        "ef_search": args.ef_search if args.ef_search is not None else 40,
    }

    if latencies:
        stats["latency_ms"] = _latency_mstats(latencies)

    recalls = [r.result.recall for r in results if r.result.recall is not None]
    if recalls:
        stats["recall_avg"] = round(sum(recalls) / len(recalls), 4)

    stats.update(summarize_stats(db_samples, buffer_samples, args.container, conn_info))

    if window_s:
        output = []
        num_windows = math.ceil(elapsed / window_s)
        configured_bytes = stats.get("shared_buffers", {}).get("configured_bytes")

        for i in range(num_windows):
            w_start = i * window_s
            w_end = (i + 1) * window_s
            w = {
                "window": i,
                "window_s": window_s,
                "start_s": round(w_start, 2),
                "end_s": round(min(w_end, elapsed), 2),
                "qps_target": args.qps,
                "period": args.period,
                "duration_s": round(elapsed, 2),
                "neighbors": args.neighbors,
                "table": args.table,
                "ef_search": args.ef_search if args.ef_search is not None else 40,
            }

            # Query latencies in this window
            w_lats = [r.result.latency for r in results if w_start <= r.window < w_end]
            w["queries"] = len(w_lats)
            if w_lats:
                actual_duration = min(w_end, elapsed) - w_start
                w["qps_actual"] = (
                    round(len(w_lats) / actual_duration, 2)
                    if actual_duration > 0
                    else 0
                )
                w["latency_ms"] = _latency_mstats(w_lats)

            # Recall in this window
            w_recalls = [
                r.result.recall
                for r in results
                if w_start <= r.window < w_end and r.result.recall is not None
            ]
            if w_recalls:
                w["recall_avg"] = round(sum(w_recalls) / len(w_recalls), 4)

            # Docker stats in this window
            w_db = [s for s in db_samples if w_start <= s["ts"] < w_end]
            if w_db:
                cpu_vals = [s["cpu_pct"] for s in w_db]
                mem_vals = [s["mem_pct"] for s in w_db]
                w["db_stats"] = {
                    "samples": len(w_db),
                    "cpu_pct": {
                        "avg": round(sum(cpu_vals) / len(cpu_vals), 2),
                        "max": round(max(cpu_vals), 2),
                    },
                    "mem_pct": {
                        "avg": round(sum(mem_vals) / len(mem_vals), 2),
                        "max": round(max(mem_vals), 2),
                    },
                }

            # Buffer stats in this window
            w_buf = [s for s in buffer_samples if w_start <= s["ts"] < w_end]
            if w_buf and configured_bytes:
                buf_bytes = [s["total_bytes"] for s in w_buf]
                avg_b = round(sum(buf_bytes) / len(buf_bytes))
                max_b = max(buf_bytes)
                w["shared_buffers"] = {
                    "samples": len(w_buf),
                    "bytes": {"avg": avg_b, "max": max_b},
                    "utilization_pct": {
                        "avg": round(avg_b / configured_bytes * 100, 2),
                        "max": round(max_b / configured_bytes * 100, 2),
                    },
                }

            output.append(w)
        return output
    else:
        return [stats]


def execute(args: argparse.Namespace):
    """Execute the query command."""
    if args.seed is not None:
        random.seed(args.seed)

    duration_s = _parse_period(args.period)
    window_s = _parse_period(args.window) if args.window else None
    total_queries = args.qps * duration_s
    conninfo = build_conninfo(args)

    # Pre-generate query embeddings
    if args.input:
        input_dataset = _load_dataset(args.input, args.samples)
    else:
        pool_size = min(total_queries, 10000)
        input_dataset = _synthesize_dataset(pool_size)

    # Compute ground truth if reference dataset provided
    ground_truth: Optional[dict[str, list[str]]] = None
    if args.reference:
        import h5py

        print(f"Calculating ground truth from reference dataset {args.reference}")
        with h5py.File(args.reference, "r") as f:
            ref_embeddings = f["embeddings"][:]
            ref_texts = f["texts"][:].astype(str)

        query_vectors = np.stack([e.raw for e in input_dataset])

        # Normalize as float32 to avoid silent upcast to float64 from np.linalg.norm
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True).astype(np.float32)
        ref_embeddings = ref_embeddings / np.linalg.norm(ref_embeddings, axis=1, keepdims=True).astype(np.float32)

        # Chunk queries to bound peak memory. Each chunk allocates:
        #   cosine_sim: chunk_size × num_ref × 4 bytes (float32)
        #   argpartition: chunk_size × num_ref × 8 bytes (int64)
        # Target ~500MB per chunk for the cosine_sim matrix.
        chunk_size = max(1, (500 * 1024 * 1024) // (ref_embeddings.shape[0] * 4))
        ground_truth = {}
        for i in range(0, len(query_vectors), chunk_size):
            chunk = query_vectors[i : i + chunk_size]
            cosine_sim = np.inner(chunk, ref_embeddings)
            top_n_indices = np.argpartition(-cosine_sim, args.neighbors, axis=1)[:, : args.neighbors]
            for j, indices in enumerate(top_n_indices):
                ground_truth[input_dataset[i + j].text] = [ref_texts[idx] for idx in indices]
        print(
            f"Computed ground truth for {len(ground_truth)} queries, top {args.neighbors} neighbors each"
        )

    # Start monitoring threads
    stop_event = threading.Event()
    db_samples = []
    buffer_samples = []
    t0 = time.monotonic()

    docker_thread = threading.Thread(
        target=poll_docker_stats,
        args=(args.container, db_samples, stop_event, t0),
        daemon=True,
    )
    docker_thread.start()

    buffer_thread = threading.Thread(
        target=poll_shared_buffers,
        args=(conninfo, args.table, buffer_samples, stop_event, t0),
        daemon=True,
    )
    buffer_thread.start()

    # Run queries at target QPS
    timeout_ms = args.timeout * 1000
    print(
        f"Running {args.qps} QPS for {args.period} ({duration_s}s), timeout={args.timeout}s..."
    )
    query_results: List[WindowResult] = []
    error_times: List[float] = []  # [relative_submit_time, ...]
    max_workers = max(args.qps * 2, 4)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []  # [(relative_submit_time, future), ...]
        start = time.monotonic()
        query_num = 0

        while query_num < total_queries:
            target_time = start + query_num / args.qps
            sleep_time = target_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

            query_input = input_dataset[query_num % len(input_dataset)]
            expected = ground_truth.get(query_input.text) if ground_truth else None
            rel_time = time.monotonic() - start
            futures.append(
                (
                    rel_time,
                    pool.submit(
                        _run_single_query,
                        conninfo,
                        args.table,
                        args.neighbors,
                        query_input.vector,
                        timeout_ms,
                        expected,
                        args.ef_search,
                    ),
                )
            )
            query_num += 1

        # Collect results
        for rel_time, f in futures:
            try:
                result = f.result(timeout=30)
                query_results.append(WindowResult(rel_time, result))
            except Exception as e:
                if not error_times:
                    print(f"First query error: {e}")
                error_times.append(rel_time)

    elapsed = time.monotonic() - start

    # Stop monitoring
    stop_event.set()
    docker_thread.join()
    buffer_thread.join()

    # Parse extra metadata
    meta = {}
    for item in args.meta:
        key, _, value = item.partition(":")
        if not _:
            raise ValueError(f"Invalid --meta format: {item!r} (expected key:value)")
        meta[key] = value

    # Aggregate / dump stats
    for datapoint in _generate_stats(
        args,
        query_results,
        db_samples,
        buffer_samples,
        conninfo,
        error_times,
        window_s,
        elapsed,
    ):
        datapoint.update(meta)
        print(json.dumps(datapoint))
        if args.stats_file:
            write_stats(datapoint, args.stats_file)


def register_query_command(subparsers: SubParsersAction) -> None:
    """Register the query subcommand."""
    parser = subparsers.add_parser(
        "query", help="Query nearest neighbors with load testing and monitoring"
    )
    add_common_args(parser)

    parser.add_argument(
        "-n",
        "--neighbors",
        type=int,
        default=10,
        help="Number of nearest neighbors per query (default: 10)",
    )
    parser.add_argument(
        "--qps", type=int, default=1, help="Target queries per second (default: 1)"
    )
    parser.add_argument(
        "--period",
        type=str,
        default="1s",
        help="Duration to run, e.g. 30s, 5m, 2h (default: 1s)",
    )
    parser.add_argument(
        "--window",
        type=str,
        default=None,
        help="Stats window size, e.g. 15s, 1m (buckets stats into time windows)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Query statement timeout in seconds (default: 1)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to an HDF5 file to use as the query dataset instead of synthesizing",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to use from the --input HDF5 file (defaults to all)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to an HDF5 file containing reference embeddings",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=None,
        help="HNSW ef_search parameter (default: pgvector default)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for text generation"
    )
    parser.add_argument(
        "--meta",
        type=str,
        action="append",
        default=[],
        help="Extra metadata as key:value (repeatable, e.g. --meta m:16 --meta ef_construction:64)",
    )
    parser.set_defaults(func=execute)
