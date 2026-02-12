"""Query command for load-testing nearest neighbor queries with monitoring."""
import json
import math
import random
import re
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import psycopg
from sentence_transformers import SentenceTransformer

from .common import (
    add_common_args, build_conninfo, poll_docker_stats, poll_shared_buffers,
    summarize_stats, write_stats,
)


def generate_random_text(min_length=100, max_length=500):
    """Generate random ASCII text of specified length range."""
    length = random.randint(min_length, max_length)
    chars = string.ascii_letters + string.digits + string.punctuation + ' ' * 10
    return ''.join(random.choice(chars) for _ in range(length))


def _parse_period(period_str):
    """Parse a period string like '5m', '30s', '2h' to seconds."""
    match = re.fullmatch(r'(\d+)([smh])', period_str)
    if not match:
        raise ValueError(f"Invalid period format: {period_str}. Use e.g. '30s', '5m', '2h'")
    value, unit = int(match.group(1)), match.group(2)
    return value * {'s': 1, 'm': 60, 'h': 3600}[unit]


def _latency_stats(latencies):
    """Compute latency percentile statistics."""
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    return {
        'avg': round(sum(sorted_lat) / n, 4),
        'p50': round(sorted_lat[n // 2], 4),
        'p95': round(sorted_lat[min(int(n * 0.95), n - 1)], 4),
        'p99': round(sorted_lat[min(int(n * 0.99), n - 1)], 4),
        'max': round(sorted_lat[-1], 4),
    }


_thread_local = threading.local()


def _run_single_query(conninfo, table, neighbors, vector_str):
    """Run a single nearest-neighbor query. Returns latency in seconds."""
    if not hasattr(_thread_local, 'conn') or _thread_local.conn.closed:
        _thread_local.conn = psycopg.connect(conninfo, autocommit=True)
    conn = _thread_local.conn

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT id, text, embedding <=> %s::vector AS distance
            FROM {table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (vector_str, vector_str, neighbors))
        cur.fetchall()
    return time.perf_counter() - t0


def execute(args):
    """Execute the query command."""
    if args.seed is not None:
        random.seed(args.seed)

    duration_s = _parse_period(args.period)
    window_s = _parse_period(args.window) if args.window else None
    total_queries = args.qps * duration_s
    conninfo = build_conninfo(args)

    # Pre-generate query embeddings
    pool_size = min(total_queries, 100)
    print(f"Loading BGE model and pre-generating {pool_size} query embedding(s)...")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')

    if args.query_text:
        texts = [args.query_text] * pool_size
    else:
        texts = [generate_random_text() for _ in range(pool_size)]

    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=32)
    vector_strs = ['[' + ','.join(map(str, emb)) + ']' for emb in embeddings]

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

    # Wait for at least one docker sample
    deadline = time.monotonic() + 15
    while not db_samples and time.monotonic() < deadline:
        time.sleep(0.1)

    # Run queries at target QPS
    print(f"Running {args.qps} QPS for {args.period} ({duration_s}s)...")
    query_results = []  # [(relative_submit_time, latency), ...]
    errors = 0
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

            vec = vector_strs[query_num % len(vector_strs)]
            rel_time = time.monotonic() - start
            futures.append((rel_time, pool.submit(
                _run_single_query, conninfo, args.table, args.neighbors, vec
            )))
            query_num += 1

        # Collect results
        for rel_time, f in futures:
            try:
                query_results.append((rel_time, f.result(timeout=30)))
            except Exception:
                errors += 1

    elapsed = time.monotonic() - start

    # Stop monitoring
    stop_event.set()
    docker_thread.join()
    buffer_thread.join()

    # Aggregate stats
    latencies = [lat for _, lat in query_results]
    stats = {
        'qps_target': args.qps,
        'qps_actual': round(len(latencies) / elapsed, 2) if elapsed > 0 else 0,
        'period': args.period,
        'duration_s': round(elapsed, 2),
        'total_queries': len(latencies),
        'errors': errors,
        'neighbors': args.neighbors,
        'table': args.table,
    }

    if latencies:
        stats['latency_s'] = _latency_stats(latencies)

    stats.update(summarize_stats(db_samples, buffer_samples, args.container, conninfo))

    if window_s:
        num_windows = math.ceil(elapsed / window_s)
        configured_bytes = stats.get('shared_buffers', {}).get('configured_bytes')

        for i in range(num_windows):
            w_start = i * window_s
            w_end = (i + 1) * window_s
            w = {
                'window': i,
                'window_s': window_s,
                'start_s': round(w_start, 2),
                'end_s': round(min(w_end, elapsed), 2),
                'qps_target': args.qps,
                'period': args.period,
                'duration_s': round(elapsed, 2),
                'neighbors': args.neighbors,
                'table': args.table,
            }

            # Query latencies in this window
            w_lats = [lat for t, lat in query_results if w_start <= t < w_end]
            w['queries'] = len(w_lats)
            if w_lats:
                actual_duration = min(w_end, elapsed) - w_start
                w['qps_actual'] = round(len(w_lats) / actual_duration, 2) if actual_duration > 0 else 0
                w['latency_s'] = _latency_stats(w_lats)

            # Docker stats in this window
            w_db = [s for s in db_samples if w_start <= s['ts'] < w_end]
            if w_db:
                cpu_vals = [s['cpu_pct'] for s in w_db]
                mem_vals = [s['mem_pct'] for s in w_db]
                w['db_stats'] = {
                    'samples': len(w_db),
                    'cpu_pct': {'avg': round(sum(cpu_vals)/len(cpu_vals), 2), 'max': round(max(cpu_vals), 2)},
                    'mem_pct': {'avg': round(sum(mem_vals)/len(mem_vals), 2), 'max': round(max(mem_vals), 2)},
                }

            # Buffer stats in this window
            w_buf = [s for s in buffer_samples if w_start <= s['ts'] < w_end]
            if w_buf and configured_bytes:
                buf_bytes = [s['total_bytes'] for s in w_buf]
                avg_b = round(sum(buf_bytes) / len(buf_bytes))
                max_b = max(buf_bytes)
                w['shared_buffers'] = {
                    'samples': len(w_buf),
                    'bytes': {'avg': avg_b, 'max': max_b},
                    'utilization_pct': {
                        'avg': round(avg_b / configured_bytes * 100, 2),
                        'max': round(max_b / configured_bytes * 100, 2),
                    },
                }

            print(json.dumps(w))
            if args.stats_file:
                write_stats(w, args.stats_file)
    else:
        print(json.dumps(stats, indent=2))
        if args.stats_file:
            write_stats(stats, args.stats_file)


def register_query_command(subparsers):
    """Register the query subcommand."""
    parser = subparsers.add_parser(
        'query',
        help='Query nearest neighbors with load testing and monitoring'
    )
    add_common_args(parser)

    parser.add_argument(
        '-n', '--neighbors',
        type=int,
        default=10,
        help='Number of nearest neighbors per query (default: 10)'
    )
    parser.add_argument(
        '--qps',
        type=int,
        default=1,
        help='Target queries per second (default: 1)'
    )
    parser.add_argument(
        '--period',
        type=str,
        default='1s',
        help='Duration to run, e.g. 30s, 5m, 2h (default: 1s)'
    )
    parser.add_argument(
        '--window',
        type=str,
        default=None,
        help='Stats window size, e.g. 15s, 1m (buckets stats into time windows)'
    )
    parser.add_argument(
        '--query-text',
        type=str,
        default=None,
        help='Query text (default: generate random text per query)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for text generation'
    )
    parser.set_defaults(func=execute)
