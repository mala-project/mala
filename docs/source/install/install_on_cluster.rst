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




