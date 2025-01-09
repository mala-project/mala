# MALA example with ACE

Note that sharing the same MALA_DATA_REPO between ACE and SNAP may lead
to errors resulting from mismatched descriptor counts. To be safe,
a separate folder (this current directory) is specified for ACE examples.

## Usage

###preprocess data to get ACE descriptors 
python ex07_preprocess_ace.py
###train network (current example is for demonstration purposes only - not for predictive modeling of Be)
python ex08_train_ace.py
###test the model (evaluate the band energy)
python ex09_run_prediction_ace.py


## Requirements

Requires installation of LAMMPS with ACE grid computes (ML-PACE compile flag on)


