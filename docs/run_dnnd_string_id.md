# Run DNND with String IDs

This guide shows how to build a kNN graph (kNNG) and run queries when input
points use string IDs.

## Point File Format

For point files with string IDs, DNND currently supports `wsv-id` and `tsv-id`.
Each row contains one point, and the first field is the point's string ID.

## Build a kNNG with String IDs (Distributed)

```shell
cd build
mpirun -np 4 ./examples/dnnd_advanced_index_build_str_id \
  -k 5 \
  -f l2 \
  -p wsv-id \
  -d /tmp/pm_datastore \
  ../examples/datasets/point_5-4_str-id.txt
```

To dump the constructed kNNG into text files, add:

```shell
-G /tmp/index-dump/
```

## Run Distributed Query with String IDs

Example: find 5 approximate nearest neighbors for each query point in
`../examples/datasets/query_5-4.txt`.

```shell
mpirun -np 4 ./examples/dnnd_advanced_query_str_id \
  -d /tmp/pm_datastore \
  -f l2 \
  -q ../examples/datasets/query_5-4.txt \
  -n 5 \
  -o /tmp/query-results
```

Query results are written to the text file given by `-o`.
If the query file has `Q` rows, the output has `2 * Q` rows:

1. Rows `0 ... Q-1`: neighbor string IDs for each query point, sorted by
distance (ascending).
2. Rows `Q ... 2Q-1`: corresponding distances.

For query index `i` (0-based), row `i` contains neighbor IDs, and row `i + Q`
contains the matching distances.

## Run Shared-Memory Query Against the Dumped kNNG

`run_query_knng_str_id_float_features` reads the kNNG dumped by DNND and runs
query search on a shared-memory machine.

```shell
./examples/run_query_knng_str_id_float_features \
  -i ../examples/datasets/point_5-4_str-id.txt \
  -p wsv-id \
  -g /tmp/index-dump/ \
  -f l2 \
  -q ../examples/datasets/query_5-4.txt \
  -n 5 \
  -o /tmp/out
```

The `-o` output format is the same as the distributed query output described
above.
