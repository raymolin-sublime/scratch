"""Shared monitoring functions for PostgreSQL and Docker."""

import json
import subprocess
import time

import psycopg


def poll_docker_stats(container, samples, stop_event, t0=None):
    """Background thread that polls docker stats every 2 seconds."""
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                [
                    "docker",
                    "stats",
                    container,
                    "--no-stream",
                    "--format",
                    "{{.CPUPerc}}\t{{.MemPerc}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split("\t")
                cpu = float(parts[0].strip("%"))
                mem = float(parts[1].strip("%"))
                sample = {"cpu_pct": cpu, "mem_pct": mem}
                if t0 is not None:
                    sample["ts"] = time.monotonic() - t0
                samples.append(sample)
        except Exception:
            pass
        stop_event.wait(2)


def poll_shared_buffers(conninfo, table_name, samples, stop_event, t0=None):
    """Background thread that polls shared buffer usage every 2 seconds."""
    try:
        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                while not stop_event.is_set():
                    try:
                        snap = snapshot_buffers(cur, table_name)
                        snap["ts"] = (
                            time.monotonic() - t0
                            if t0 is not None
                            else time.monotonic()
                        )
                        samples.append(snap)
                    except Exception:
                        pass
                    stop_event.wait(2)
    except Exception:
        pass


def snapshot_buffers(cur, table_name):
    """Query pg_buffercache for buffers belonging to the given table."""
    cur.execute(
        """
        SELECT count(*) AS buffer_count,
               count(*) * current_setting('block_size')::bigint AS total_bytes,
               pg_size_pretty(count(*) * current_setting('block_size')::bigint) AS total_size
        FROM pg_buffercache b
        JOIN pg_class c ON c.relfilenode = b.relfilenode
        WHERE c.relname = %s
    """,
        (table_name,),
    )
    row = cur.fetchone()
    return {"buffer_count": row[0], "total_bytes": row[1], "total_size": row[2]}


def build_conninfo(args) -> str:
    """Build a psycopg conninfo string from parsed args."""
    return f"host={args.host} port={args.port} dbname={args.database} user={args.user} password={args.password}"


def summarize_stats(db_samples, buffer_samples, container, conninfo):
    """Aggregate docker and shared buffer samples into a stats dict."""
    result = {}

    if db_samples:
        cpu_vals = [s["cpu_pct"] for s in db_samples]
        mem_vals = [s["mem_pct"] for s in db_samples]
        result["db_stats"] = {
            "container": container,
            "samples": len(db_samples),
            "cpu_pct": {
                "avg": round(sum(cpu_vals) / len(cpu_vals), 2),
                "max": round(max(cpu_vals), 2),
            },
            "mem_pct": {
                "avg": round(sum(mem_vals) / len(mem_vals), 2),
                "max": round(max(mem_vals), 2),
            },
        }

    if buffer_samples:
        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT setting::bigint * current_setting('block_size')::bigint
                    FROM pg_settings WHERE name = 'shared_buffers'
                """)
                shared_buffers_total_bytes = cur.fetchone()[0]

        buf_bytes = [s["total_bytes"] for s in buffer_samples]
        avg_bytes = round(sum(buf_bytes) / len(buf_bytes))
        max_bytes = max(buf_bytes)
        result["shared_buffers"] = {
            "configured_bytes": shared_buffers_total_bytes,
            "samples": len(buffer_samples),
            "bytes": {
                "avg": avg_bytes,
                "max": max_bytes,
            },
            "utilization_pct": {
                "avg": round(avg_bytes / shared_buffers_total_bytes * 100, 2),
                "max": round(max_bytes / shared_buffers_total_bytes * 100, 2),
            },
        }

    return result


def add_common_args(parser):
    """Add database/monitoring arguments shared by load and query commands."""
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="PostgreSQL host (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=5432, help="PostgreSQL port (default: 5432)"
    )
    parser.add_argument(
        "--database",
        type=str,
        default="postgres",
        help="Database name (default: postgres)",
    )
    parser.add_argument(
        "--user", type=str, default="postgres", help="Database user (default: postgres)"
    )
    parser.add_argument(
        "--password",
        type=str,
        default="postgres",
        help="Database password (default: postgres)",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="embeddings",
        help="Table name (default: embeddings)",
    )
    parser.add_argument(
        "--container",
        type=str,
        default="pgvector-db",
        help="Docker container name for stats polling (default: pgvector-db)",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default=None,
        help="File path to append stats JSON (one object per line)",
    )


def write_stats(stats, path):
    """Append a stats dict as a JSON line to the given file."""
    with open(path, "a") as f:
        f.write(json.dumps(stats) + "\n")
    print(f"Stats appended to {path}")
