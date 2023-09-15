Distributed NN-Descent (DNND)
=======

Overview
--------

DNND is a distributed memory version of a well-known k-nearest neighbor graph (k-NNG) construction algorithm, `NN-Descent`_
(we use k-NNG and k-NN index interchangeably).

DNND is implemented in C++ and uses YGM (MPI) for communication.
In addition to constructing k-NNGs,
DNND provides k-NNG optimization and k-approximate neighbor neighbor search features.

DNND also provides a persistent memory (PM) mode, which enables users to store k-NNGs in files and reuse them later,
leveraging `Metall <https://github.com/LLNL/metall>`_ persistent memory allocator.
The non-persistent mode DNND class is :code:`saltatlas::dnnd` and the persistent mode DNND class is :code:`saltatlas::dnnd_pm`.

DNND Examples
--------

Here we describe how to build and run DNND examples.
To build DNND examples, please refer to the :doc:`Build <../build>` page.

Simple DNND Example
^^^^^^^

We provides a single example program, :code:`examples/dnnd_example.cpp`, which shows how to use DNND.
The program constructs a k-NNG, optimizes it, and performs k-approximate nearest neighbor search.

.. code-block:: shell
  :caption: Build Example

  # Assumes that we are in a build directory
  mpirun -n [#of procs] ./examples/dnnd_example


DNND PM (persistent memory mode) Examples
^^^^^^^

The example directory also contains three programs for DNND PM mode as follows:

* :code:`examples/dnnd_pm_example.cpp`: Constructs a k-NNG and stores it in files.
* :code:`examples/dnnd_pm_optimize_example.cpp`: Loads a k-NNG from files and optimizes it.
* :code:`examples/dnnd_pm_search_example.cpp`: Loads a k-NNG from files and performs a k-approximate nearest neighbor search.

.. code-block:: shell
  :caption: Build Example

  # Assumes that we are in a build directory

  # Construct a k-NNG
  srun -n 24 -N 1 ./examples/dnnd_pm_const_example -f l2 -p wsv -k 20 -r 0.8 -d 0.001 -e  -z ./index  -v  ./examples/datasets/point_5-4.txt

  # Optimize a k-NNG
  srun -n 24 -N 1 ./examples/dnnd_pm_optimize_example -z ./index -v -u -m 1.5

  # Perform a k-approximate nearest neighbor search
  srun -n 24 -N 1 ./examples/dnnd_pm_query_example -z ./index -q ./examples/datasets/query_5-4.txt -n 100 -g  ./examples/datasets/ground-truth_5-4.txt

For more details about the options, please run the programs with the :code:`-h` option.

.. _NN-Descent: https://dl.acm.org/doi/10.1145/3227609.3227643