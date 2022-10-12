## Special Instructions for Installing HDF5 with Spack
``` bash
module load gcc/8.3.1
spack compiler add
# Install HDF5 v1.10.7 with C++ support using GCC v8.3.1 and without MPI support
spack install hdf5@1.10.7%gcc@8.3.1+cxx~mpi
```

## Building
These instructions assume you have a relatively modern C++ compiler (C++17 required, only tested on GCC) and Cereal
installed. Spack makes it easier to include Cereal in the project, but this can be done without Spack. Instructions for
each method will be provided.

### Generic steps
These steps are executed regardless of whether or not Spack is used.
``` bash
# Load an appropriate version of gcc (on LC systems)
module load gcc/8.3.1
git clone https://github.com/LLNL/saltatlas.git
cd saltatlas
mkdir build && cd build
```

### Option 1: Using Spack
If Spack is being used, HDF5 must be installed before the following
instructions. 
By default, HDF5 has a dependence on MPI, which will often cause issues.

The process for building with HDF5 is slightly finicky due to Spack's HDF5
library coming with an MPI library that will conflict with LC.

``` bash
spack load hdf5~mpi

cmake ..
make
```

When loading HDF5, the "~mpi" is only strictly necessary if you have multiple
versions of HDF5 installed, some of which have MPI support.

### Option 2: Not Using Spack
If Spack is not being used, we must tell CMake where to find Cereal before we begin the build process. If cereal is not
already installed and an internet connection is available, it can be obtained using
``` bash
git clone https://github.com/USCiLab/cereal.git
```

Once cereal and HDF5 are installed, we tell CMake where to find them and complete the build process. 

``` bash
export CMAKE_PREFIX_PATH=/path/to/cereal:/path/to/hdf5/:${CMAKE_PREFIX_PATH}
export CMAKE_PREFIX_PATH=/path/to/hdf5/:${CMAKE_PREFIX_PATH}

cmake ..
make
```

### Option 3: Dropping HDF5

One can also use saltatlas without the HDF5 dependency.
The procedure is similar to building with HDF5, while ignoring the spack or 
environment variable manipulation in the above text that refers to HDF5.
In addition, when invoking cmake one should run:
``` bash
cmake .. -DSALTATLAS_USE_HDF5=OFF
```

## Using HDF5 benchmark

The HDF5 benchmark provides 3 successive levels of benchmarking:
1. Speed of building a distributed index
2. Querying throughput
3. Measuring recall as an indication of performance when ground truth data is given.

Each level builds on the previous level, requiring all of the same inputs plus some additional ones. Here is a quick
summary of using each benchmark and the options required.

### Speed of index construction
To build an index, the only necessary components are the points used to build the index, the number of Voronoi cells to
use, and the Voronoi rank to use during construction (that is, how many pointers to other Voronoi cells each data point in the index stores).
These are given through the -i, -s, and -v flags, respectively.

An example of using this benchmark is:
```bash
srun -n 24 benchmark_hdf5 -v 2 -s 96 -i /path/to/index.hdf
```

### Querying throughput
To test the querying throughput, we need to build an index, and we need to provide a collection of data points to query
with, the number of nearest neighbors to search for, and the number of hops to take when querying. These parameters are
given using the -q, -k, and -p flags, respectively.

An example of using this benchmark is:
```bash
srun -n 24 benchmark_hdf5 -v 2 -s 96 -k 10 -p 2 -i /path/to/index.hdf -q /path/to/query.hdf
```

### Querying recall
To calculate recall when querying, we need to build an index and provide data points to query with. In addition, we need
to provide a file containing the ground truth nearest neighbors, given using the -g flag.

An example of using this benchmark is:
```bash
srun -n 24 benchmark_hdf5 -v 2 -s 96 -k 10 -p 2 -i /path/to/index.hdf -q /path/to/query.hdf -g /path/to/ground_truth.hdf
```

### Files provided to benchmark
The files provided to the benchmarking application can either be HDF5 files containing all necessary data, or a text
file containing a list of HDF5 files to use.

For example, if we run
`srun -n 24 benchmark_hdf5 -v 2 -s 96 -i /path/to/index.hdf`, then the benchmark attempts to open `/path/to/index.hdf`
as an HDF5 file. If it is unsuccessful, it assumes it is a text file containing a list of HDF5 files to use for building
the distributed index. If a collection of files are given, the files will be distributed across MPI ranks for reading.

## Running examples

### Running example without HDF5
Within a Slurm allocation run the example using
``` bash
srun -n 2 src/dknn_example
```

### Playing with HDF5
The example given in src/knn_dpockets_hdf5.cpp uses HDF5 files (assumes 8-D floating point data, like in dpockets). It
uses a single rank to read in the HDF5 and distribute it to all other ranks.

It is designed to sweep over a range of values for Voronoi rank and for number of hops. It can be run using
```bash
srun -n 24 src/knn_dpockets_hdf5 num_seeds min_Voronoi_rank max_Voronoi_rank min_hops max_hops /path/to/HDF5/data
```

### Examples
The basic example is given in src/dknn_example.cpp. This example contains two uses of this code, differing in the way
metric spaces are constructed.

Within hnswlib, a memcpy is performed on data when a point is added to an HNSW. They expect to be given a C-style array
of data points which they then copy for their own purposes. All of their built-in distance functions are working on data
of this form. 

Additionally, they have hard-coded an alias for their distance functions to be
```
template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);
```
As a user, creating your own distance function requires casting the first two `void *` arguments to the type of your
data points and using the third as any additional arguments needed for your distance function (such as dimension).

To avoid copying and make it easier to write distance functions with more interesting types (i.e. std::vector,
std::string, etc.) that don't necessarily have fixed sizes, we provide a wrapper that handles the casting of datatypes in
distance functions, so a user can write a simpler function that takes two std::vector's. When used in this way, an
std::vector gets added to hnswlib by performing a memcpy on the vector (copying a pointer, not the actual data). This
data is already stored in our distributed data structure, so we end up with a single copy instead of two.

In dknn_example.cpp, the first example uses this wrapper around a distance function working on std::vector's. The second
example uses a built-in distance function and std::array's for data points, more like how hnswlib appears to be designed
for.

While the second example should be faster, the first is easier to use and more easily supports variable length data
(i.e. strings).


## Running DNND (Distributed NNDescent) Example

```shell
# Usage
mpirun -n [#of procs] ./examples/dnnd_example (options) /path/to/point/file/0 /path/to/point/file/1 ...

# Show help menu (available options)
mpirun -n 1 ./examples/dnnd_example -h
```

### Running Example

```shell
cd build

# Construct a k-NN index
mpirun -n 2 ./examples/dnnd_example -k 4 -f l2 -p wsv ../examples/datasets/point_5-4.dat 

# Construct a k-NN index, query nearest neighbors, and show the accuracy.
mpirun -n 2 ./examples/dnnd_example -k 2 -f l2 \
  -n 4 -q ../examples/datasets/query_5-4.dat -g ../examples/datasets/neighbor_5-4.dat \ 
  -p wsv ../examples/datasets/point_5-4.dat
```


## Running DNND PM (persistent memory) Examples

### Required CMake Option

The DNND PM examples require Metall and Boost C++ Libraries.
Add `-DSALTATLAS_USE_METALL=ON` when running CMake.

Those libraries are automatically downloaded and set up properly.

### Executables

There are three examples executables for k-NN index construction,
k-NN index optimization, and query, respectively.

Use `-h` option to show the help menus. 

#### dnnd_pm_const_example

This program constructs a k-NN index.

```shell
mpirun -n [#of procs] ./examples/dnnd_pm_const_example (options) point_file_0 point_file_1...
```

#### dnnd_pm_optimize_example

This program optimizes an already constructed k-NN index.

```shell
mpirun -n [#of procs] ./examples/dnnd_pm_optimize_example (options)
```


#### dnnd_pm_query_example

This program performs queries against an already constructed index.

```shell
mpirun -n [#of procs] ./examples/dnnd_pm_query_example (options)
```


### Running Example

```shell
cd build

# Construct a k-NN index and store
mpirun -n 2 ./examples/dnnd_pm_const_example -z ./pindex -k 2 -f l2 -p wsv ../examples/datasets/point_5-4.dat 

# Optimize the k-NN index created above
mpirun -n 2 ./examples/dnnd_pm_optimize_example -z ./pindex -u -m 1.5

# Open the k-NN index created above, query nearest neighbors, and show the accuracy.
mpirun -n 2 ./examples/dnnd_pm_query_example -z ./pindex \
  -n 4 -q ../examples/datasets/query_5-4.dat -g ../examples/datasets/neighbor_5-4.dat
```

# License
saltatlas is distributed under the MIT license.

All new contributions must be made under the MIT license.

See [LICENSE-MIT](LICENSE-MIT), [NOTICE](NOTICE), and [COPYRIGHT](COPYRIGHT) for
details.

SPDX-License-Identifier: MIT

# Release
LLNL-CODE-833039
