## Building
These instructions assume you have a relatively modern C++ compiler (C++17 required, only tested on GCC) and Cereal installed.
Instructions for each method will be provided.

### Generic steps
These are the generic steps in install saltatlas.
saltalas's build system will automatically fetch all dependencies.

``` bash
# Load an appropriate version of gcc (on LC systems)
module load gcc/12.1.1-magic
git clone https://github.com/LLNL/saltatlas.git
cd saltatlas
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
# Builds all compile targets
make -j
```

#### Fetching Boost

saltatlas's CMake will automatically fetch a proper version of Boost by default.

There are two CMake options to change the behavior:

- `BOOST_SOURCE_DIR`
  - Path to Boost libraries directory already downloaded and uncompressed.
  - This option will copy the only required files to build this project from the original source directory. Useful for faster build and saving disk space.
- `BOOST_FETCH_URL`
  - URL or file path to an archived Boost source.

For both cases, Boost must be a version that supports CMake.
Using two options at the same time will result in an error.

DNND and NEO-DNND require Boost 1.87 or higher.

## Running examples

### Running example
Within a Slurm allocation run the example using
``` bash
srun -n 2 src/dknn_example
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
data is already stored in our distributed data structure, so we end up with
a single copy instead of two.

In dknn_example.cpp, the first example uses this wrapper around a distance function working on std::vector's. The second
example uses a built-in distance function and std::array's for data points, more like how hnswlib appears to be designed
for.

While the second example should be faster, the first is easier to use and more easily supports variable length data
(i.e. strings).

# DNND and NEO-DNND

DNND is a distributed NN-Descent code.
NEO-DNND is a communication-efficient version of DNND.
DNND has more APIs and flexibility, targeting a wider range of use cases.
For more details, please refer to the publications below.

Example codes are available in [./examples](./examples) directory.
They are MPI programs. To run them, for example, use the following command:

```bash
mpirun -n 2 ./examples/dnnd_simple
mpirun -n 2 ./examples/neo_dnnd_example
```

# Publications

- [DNND](https://dl.acm.org/doi/abs/10.1145/3624062.3625132)

- [NEO-DNND](https://ieeexplore.ieee.org/abstract/document/10820763)

# License
saltatlas is distributed under the MIT license.

All new contributions must be made under the MIT license.

See [LICENSE-MIT](LICENSE-MIT), [NOTICE](NOTICE), and [COPYRIGHT](COPYRIGHT) for
details.

SPDX-License-Identifier: MIT

# Release
LLNL-CODE-833039
