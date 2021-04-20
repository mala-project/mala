Installation on HPC clusters: tips & tricks
===========================================
hemera5
*******************

- Tested by: Lenz Fiedler, 16.04.2021
- Scope of test: mala, torch, horovod; I ommited LAMMPS and QE for this test
- Test log:
    1. Create a conda environment:
        - ``conda create --name mala_train python=3.8.5``

    2. Installing requirements:
        - ``pip install -r requirements.txt``
            - I got "Requirement already satisfied: oapackage in /home/fiedle09/.local/lib/python3.8/site-packages (from -r requirements.txt (line 9)) (2.6.8)" - I don't know how I got to this point, but the experience might differ for other people

    3. Install Pytorch. I load CUDA for this.
        - ``module load cuda/10.2``
        - ``pip install torch``

    4. Horovod needs CMAKE to work, so we need to load it along with the other dependencies required for it:
        - ``module load gcc/7.3.0``
        - ``module load cmake``

    5. Then I install horovod as
        ``pip install horovod[pytorch]``

Summit
*******************

- Tested by: Vladyslav Oles, 16.04.2021
- Scope of test: mala, torch, horovod; I ommited LAMMPS and QE for this test
- Test log:
    1. Install a current version of SWIG (instead of Summit's 2.0.10) as per http://www.swig.org/svn.html:
        - ``git clone https://github.com/swig/swig.git``
        - ``cd swig``
        - ``./autogen.sh``
        - ``./configure --prefix=/.../swig`` (replace ``...`` with absolute path to ``swig`` directory)
        - ``make``
        - ``make install``
        - ``PATH = /.../swig/bin:$PATH`` (replace ``...`` with actual absolute path to ``swig`` directory)

    2. Clone conda environment with pytorch and horovod, and add oapackage to it:
        - ``module load open-ce/1.1.3-py38-0``
        - ``conda create --name mala-opence --clone open-ce-1.1.3-py38-0``
        - ``conda activate mala-opence``
        - ``conda install gxx_linux-ppc64le``
        - ``pip install oapackage``

    3. Install MALA (from the directory with cloned MALA repository):
        - ``pip install -e .`` (note that it will install required packages for MALA listed in ``requirements.txt``)



