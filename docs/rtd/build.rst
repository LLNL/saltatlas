Building saltatlas
=======

Required
-----

* CMake 3.14 or more

* GCC 8.3 or more

  * GCC 9.3 or more is recommended

* MPI

.. WARNING::
  Which version is required for YGM?


Option
-----

* HDF5 (C++ interface)

.. WARNING::
  should we turn off HDF5 by default?


Dependencies
-----

saltatlas depends on some libraries.
Those libraries are downloaded and set up automatically by saltatlas's CMake script.

Build
-----

.. code-block:: shell
  :caption: Build Example

  git clone https://github.com/LLNL/saltatlas.git
  cd saltatlas
  mkdir build && cd build
  cmake ../ -DCMAKE_BUILD_TYPE=Release
  make # add '-j' for parallel build

Some programs in saltatlas depend on additional libraries.
For more information, see the next section.

Often Used CMake Options
^^^^

* :code:`SALTATLAS_USE_HDF5=on/off`

  * If :code:`on` is specified, saltatlas is built with HDF5.
  * HDF5 must be installed (for more details, see [link]).
  * Default is off.

* :code:`SALTATLAS_USE_METALL=on/off`

  * If :code:`on` is specified, saltatlas is built with Metall.
  * DNND PM depends on Metall.
  * saltatlas's CMake automatically downloads and sets up Metall if this option is turned on.

* :code:`BOOST_ROOT=/path/to/boost`

  * Path to C++ Boost Libraries.
  * If this option is used, our CMake configure step uses the specified Boost instead of downloading.
  * Install is not required. Only header files are used.
