Developing saltatlas
===========

This page contains information for saltatlas developers.

Build Read the Docs (RTD)
-------

Here is how to build RTD document using Sphinx on your machine.

.. code-block:: shell
  :caption: How to build RTD docs locally

  # Install required software
  brew install doxygen graphviz
  pip install sphinx breathe sphinx_rtd_theme

  git clone https://github.com/LLNL/saltatlas.git
  cd saltatlas
  mkdir build && cd build

  # Run CMake
  cmake ../ -DSALTATLAS_RTD_ONLY=ON

  # Generate Read the Docs documents using Sphinx
  # This command runs Doxygen to generate XML files
  # before Sphinx automatically
  make sphinx
  # Open the following file using a web browser
  open docs/rtd/sphinx/index.html

  # For running doxygen only
  make doxygen
  # open the following file using a web browser
  open docs/html/index.html

Rerunning Build Command
^^^^^

Depending on what files are modified, one may need to rerun the CMake command and/or :code:`make sphinx`.
For instance:

* Require running the CMake command and :code:`make sphinx`:

  * Adding new RTD-related files, including configuration and .rst files
  * Modifying CMake files

* Require running only :code:`make sphinx`

  * **Existing** files (except CMake) are modified