# ML-DFT Networks

------------------------------------------------
Description:
------------------------------------------------

Machine Learning models from predicting the Local Density 
of States(LDOS) from atomic configurations and their *fingerprints*. 

------------------------------------------------
To Run:
------------------------------------------------

###Tested Module/Package Combos
- python/3.7.3
- torch/1.3.0

###Install

1. Prepare python packages and env
  + Run `source load_mlmm_blake_modules.sh` (On Blake)
  + Run `./setup_python_env.sh`
  + The above will run `source ./mlmm_env/bin/activate` to set the python env
  + (Run `deactivate` if you wish to exit python env)

2. Allocate node with `salloc -N 4 -n 4 -t 60` for 4 node(`-N`), 4 ranks(`-n`), wall-limit of 1 hr
  + Make sure you are in the python env and correct modules are loaded
  + i.e. run `source mlmm_env/bin/activate` and `source load_mlmm_blake_modules.sh` again

3. Run `cd ldos_feedforward; python3 ldos_example.py --model <N>`
  + List of models `<N>`:
  + `1` Density prediction with activation fns
  + `2` Density prediction w/o activations fns
  + `3` LDOS prediction w/o LSTM
  + `4` LDOS prediction with LSTM
  + `5` LDOS prediction with bidirection LSTM (Rampi's network)

4. Run `python3 ldos_example.py --help` for more options

<!--
2. Prepare generated data
  + Move to ${minigan_dir}/data/2d or /3d 
  + Run `python generate_random_images.py`
  + Optional: Change dims of images in generate_random_images.py script

3. Run `python mlmm_dnn_driver.py -f <input_file>` 
-->

