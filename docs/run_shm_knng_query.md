# Run Shared-Memory KNNG Query

We describe how to run a shared-memory query against a KNNG constructed by
DNND or NEO-DNND.

(NEO)-DNND dumps a KNNG to files after construction.
[run_query_knng](../examples/run_query_knng.cpp) runs nearest neighbor searches
against the dumped KNNG on a shared-memory system.

Here is an example:

```shell

# Construct KNNG using NEO-DNND (or jsut DNND)
mpirun -n 2 ./examples/neo_dnnd_bench_float_features -i ./examples/datasets/point_5-4.txt -p wsv -f l2 -k 2 -G ./knng  -v  -t 2

# Run query against the constructed KNNG
./examples/run_query_knng_float_features -i ./examples/datasets/point_5-4.txt -p wsv -g ./knng -f l2 -q ./examples/datasets/query_5-4.txt -n 4 -G ./examples/datasets/ground-truth_5-4.txt 

```