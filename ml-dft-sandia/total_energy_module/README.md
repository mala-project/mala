# Build Python Module to run QE "v of rho" subroutines

## Using Intel/MKL

### Initial Setup

1. Clone QE source: git clone https://github.com/QEF/q-e.git
   On blake, it may be necessary to 'module load git' first.
2. Change into the q-e directory and check out out the qe-6.4.1 tag:
    git checkout qe-6.4.1
3. Set up a virtual environment that includes numpy (f2py). On blake, you can
   load the python/3.7.3 module and run the script ../networks/setup\_python\_env.sh.
4. Patch the QE source so that the FoX project is built with -fPIC:
   1. change to the q-e/install/m4 folder
   2. patch < path/to/this/folder/x\_ac\_qe\_f90.m4.patch
5. Copy total_energy.f90 to q-e/PW/src. This file contains the Fortran 90 code
   that f2py will create a Python binding to.

### Build Quantum Espresso

1. If you already have a build of QE, go into the q-e folder and run 'make veryclean'.
2. Set up the environment. On blake, you can source blake\_qe\_env.sh. For other
   platforms, load appropriate compiler and MPI modules, then export/set 
   environment variables as shown in blake\_qe\_env.sh. You will not be able
   to build the Python module unless these environment variables are set when
   configure is run.
3. Run 'configure' and 'make all'

### Build the Python Module
1. Load a Python 3 module (on blake: module load python/3.7.3) 
2. Activate the virtual environment. If using the mlmm\_env environment, then 
  'source /path/to/mlmm\_env/bin/activate'.
3. Load other needed modules. On blake, you can source blake\_qe\_env.sh.
4. Change to q-e/PW/src and run the build\_total\_energy\_energy\_module.sh 
   script in this folder. If the build is successful, a file named 
   total_energy.cpython-37m-x86\_64-linux-gnu.so will be generated. It contains
   the Python module.

## Using GNU and the mlmm_env environment on Blake

### Initial Setup

1. Clone QE source: git clone https://github.com/QEF/q-e.git
   On blake, it may be necessary to 'module load git' first.
2. Change into the q-e directory and check out out the qe-6.4.1 tag:
    git checkout qe-6.4.1
3. Set up the mlmm_env virtual environment if you haven't already:
   1. cd ../networks/environment/
   2. source load\_mlmm\_blake\_modules.sh 
   3. ./setup\_python\_env.sh.
4. Patch the QE source so that the FoX project is built with -fPIC:
   1. change to the q-e/install/m4 folder
   2. patch < path/to/this/folder/x\_ac\_qe\_f90.m4.patch
5. Copy total_energy.f90 to q-e/PW/src. This file contains the Fortran 90 code
   that f2py will create a Python binding to.

### Build Quantum Espresso

1. If you already have a build of QE, go into the q-e folder and run 'make veryclean'.
2. Set up the environment.
   1. source networks/environment/load\_mlmm\_blake\_modules.sh
   2. source total\_energy/blake\_qe\_env\_gnu.sh.
3. Run 'configure' and 'make all'
4. Because the GNU build doesn't use MKL, QE builds its own linear algebra libraries. These
   don't get unpacked until the make step, which means we have to rebuild them
   after making a slight modification. Open q-e/LAPACK/make.inc in a text editor. On line 22, 
   add the option '-fPIC' to the end of the "NOOPT" setting.
5. In q-e/LAPACK, run 'make clean' and 'make'
6. In q-e/LAPACK/CBLAS, run 'make clean' and 'make'.

### Build the Python Module
1. Set up the environment.
    1. source networks/environment/load\_mlmm\_blake\_modules.sh
    2. Activate the mlmm_env virtual environment: source /path/to/mlmm\_env/bin/activate
2. Change to q-e/PW/src and run the build\_total\_energy\_energy\_module\_gnu.sh 
   script in the total\_energy folder. If the build is successful, a file named 
   total_energy.cpython-37m-x86\_64-linux-gnu.so will be generated. It contains
   the Python module.

## Use the Python Module
1. Add the folder that contains total_energy.cpython-37m-x86\_64-linux-gnu.so to
   your PYTHONPATH.
2. In the Python REPL or script: import total_energy
