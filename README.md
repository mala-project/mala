# Accelerating Electronic Structure Calculation with Atom-Decomposed Neural Modeling

## Overview
This library contains a PyTorch implementation of atom-decomposed neural modeling (ADOS) of the electronic density of states (DOS) of aluminum.
Atomic environment descriptors are learned during training using Concentrical Spherical GNN (CSGNN) model.
It was originally run using Python 3.8, PyTorch 1.9, and CUDA Toolkit 11.1 on NVIDIA V100 GPU.

Authors: James Fox (jfox43@gatech.edu), Normand A. Modine (namodin@sandia.gov), Siva Rajamanickam (srajama@sandia.gov)

## Dependencies
The following installs dependencies to Anaconda virtual environment. 
```bash
conda create --name csgnn python=3.8
conda activate csgnn
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia
conda install -c dglteam dgl-cuda11.1
pip install pymatgen
pip install scikit-learn
```

### Atom-centered Density of States for Aluminum
Use the calculate\_ADOS.py script to generate ADOS reference values for
training and evaluation, per snapshot.
The ADOS is generated from LDOS, and so the script requires location of
LDOS and relevant Quantum ESPRESSO output files ('input_dir').
The generated references are saved to output directory ('output_dir').
```bash
python -m ados.calculate_ADOS [input_dir] [output_dir]
```

To evaluate pre-trained model for DOS and band energy error:
```bash
python -m dos.test saved/csgnn-ados-933K.pkl -d [data_dir]
```
'data_dir' specifies location where snapshots, ADOS reference values, and
other relevant data are stored.


To train the model from scratch with default settings:
```bash
python -m dos.train -d [data_dir]
```

# Hyperparameters

### ADOS data generation
| Name | Description | Default | File |
| --- | --- | --- | --- |
| gauss_width | Controls local sensitivity of partition of unity | 1.3 | calculate_ADOS.py |

### Model and training
| Name | Description | Default | File |
| --- | --- | --- | --- |
| rcut | Radial cutoff of atom-centered environment | 7.0 angstrom | train.py |