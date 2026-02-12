# vectordb

A CLI tool for benchmarking [pgvector](https://github.com/pgvector/pgvector) performance. Generates random text embeddings, loads them into PostgreSQL, queries nearest neighbors, and plots benchmark results.

## Setup

Requires Python >= 3.13.

```
docker compose -f docker/pgvector.yml up -d
uv sync
```

## Usage

### generate

Create random text embeddings and save to HDF5.

```
vectordb generate -n 10000 -o 10k.hdf5
```

Uses the `BAAI/bge-large-en-v1.5` sentence-transformer model. Supports `--batch-size` and `--seed`.

### load

Load embeddings from HDF5 into PostgreSQL and optionally build an HNSW index.

```
vectordb load -i 10k.hdf5 --create-table --create-index
```

Monitors Docker container stats (CPU, memory, shared buffer usage) during index creation. Use `--stats-file` to save results as JSON for later plotting.

### query

Find k-nearest neighbors using cosine distance.

```
vectordb query --query-text "some text" -n 5
```

### plot

Visualize benchmark results from JSON files.

```
vectordb plot -d results/index_creation/v2 -x num_vectors -y index_creation_time_s
```

Supports log scale (`--xlog`, `--ylog`) and saving to file (`-o output.png`).
