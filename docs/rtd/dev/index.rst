Developing saltatlas
===========

This page contains information for saltatlas developers.

Build Read the Docs (RTD)
-------

Here is how to build RTD document using Sphinx on your machine.

.. code-block:: shell
  :caption: How to build RTD docs locally

  # Install Sphinx, Breathe, and Read the Docs Theme
  pip install sphinx breathe sphinx_rtd_theme

  git clone https://github.com/LLNL/saltatlas.git
  cd saltatlas
  mkdir build && cd build

  # Run CMake with SALTATLAS_DOXYGEN=ON
  # Need to rerun this command if any CMake or RTD related files are updated
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
