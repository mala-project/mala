import mala
from mala import printout
from mala.network import Tester
import time
import os
import sys
import numpy as np
from torch.nn.functional import mse_loss
from torch import from_numpy as torch_from_numpy


def write_hist(vals, save_file, bot=-0.1, top=0.1, step=0.00005):
    bins = np.arange(bot, top+step, step)
    counts, bins = np.histogram(vals, bins=bins)
    if os.path.exists(save_file):
        existing_counts = np.load(save_file)
        new_counts = np.vstack([existing_counts,counts])
        with open(save_file, 'wb+') as f:
            np.save(f, new_counts)
    else:
        with open(save_file, 'wb+') as f:
            np.save(f, counts)

### CONSTANTS #################################################################
GEN0SIZE = int(GEN0SIZE_FLAG)
GENSIZE =  int(GENSIZE_FLAG)
# GEN0SIZE = 0 indicates we use the entire dataset
DATASET = "DATASET_FLAG"
RAND = RAND_FLAG
CUMU_FRAC = CUMU_FRAC_FLAG
###############################################################################

### Create fresh model
start = time.time()
printout("=== Constructing fresh model."); sys.stdout.flush()
# Load the parameters from the hyperparameter optimization
parameters = mala.Parameters()
parameters.manual_seed = SEED_FLAG # Set a manual seed to compare network runs
# Specify data restrictions
if not CLIP_LDOS_FLAG: parameters.targets.restrict_targets = None
# Specify the data scaling.
parameters.data.input_rescaling_type = "feature-wise-standard"
parameters.data.output_rescaling_type = "normal"
# LDOS parameters.
parameters.targets.ldos_gridsize = 400
parameters.targets.ldos_gridspacing_ev = 0.1
parameters.targets.ldos_gridoffset_ev = -20.0
parameters.network.layer_activations = ["ACTFN_FLAG"]
# Specify the training parameters, these are not completely the same as
# for the hyperparameter optimization.
parameters.running.early_stopping_epochs = 6
parameters.running.early_stopping_threshold = 1E-7
parameters.running.learning_rate = LR0_FLAG
parameters.running.trainingtype = "Adam"
parameters.running.learning_rate_scheduler = "ReduceLROnPlateau"
parameters.running.learning_rate_decay = 0.1
parameters.running.learning_rate_patience = 4
parameters.running.verbosity = True
parameters.running.mini_batch_size = 1024 
parameters.use_gpu = True
parameters.use_horovod = True
parameters.verbosity = 2
parameters.comment = "SiO2 trained to aq only"
parameters.running.checkpoint_name = "aq_training_checkpoint"
### Parameters important to active learning workflow ###########################
parameters.running.max_number_epochs = 20
parameters.running.checkpoints_each_epoch = 0
parameters.running.use_shuffling_for_samplers = False
parameters.data.use_lazy_loading = False
###############################################################################
parameters.save('aq.params.json')
parameters.show(); sys.stdout.flush()

### Constants
DO_REPARAM = True
poly = 'aqrtz'
DATA_LOC = '/gpfs/kdmille/SiO2/data_222'
shuffle_type = 'SHUFFLE_FLAG'
if shuffle_type == 's2':
    snap_path = f"{DATA_LOC}/splits_s2_snapshots12345678_seed8"
    ldos_path = f"{DATA_LOC}/splits_s2_snapshots12345678_seed8"
else:
    snap_path = f"{DATA_LOC}/splits"
    ldos_path = f"{DATA_LOC}/splits"


### Create datahandler
start = time.time()
# Check for existing scalers
iscaler_name = "aq.iscaler.pkl"
oscaler_name = "aq.oscaler.pkl"
if os.path.exists(iscaler_name) and os.path.exists(oscaler_name):
    data_handler = mala.DataHandler(parameters,
                                    input_data_scaler=mala.DataScaler.load_from_file(iscaler_name),
                                    output_data_scaler=mala.DataScaler.load_from_file(oscaler_name))
    DO_REPARAM = False
else:
    data_handler = mala.DataHandler(parameters)
    DO_REPARAM = True

### Set up constants for loading
snapshot = TRSNAP_FLAG
i = CHUNKSTART_FLAG
print(DATASET,flush=True)
if DATASET == 'small':
    ldos_name = f"test_aqrtz_snapshot-1_split0_ldos_0.npy"
    fp_name = f"test_aqrtz_snapshot-1_split0_fp_0.npy"
elif DATASET == 'tiny':
    ldos_name = f"tinytest_aqrtz_snapshot-1_split0_ldos_0.npy"
    fp_name = f"tinytest_aqrtz_snapshot-1_split0_fp_0.npy"
elif DATASET == 'big':
    if shuffle_type == 's2':
        ldos_name = f"{poly}_snapshot-{snapshot}_shuffled2_ldos_144x144x160grid_split{i}.npy"
        fp_name = f"{poly}_snapshot-{snapshot}_shuffled2_fp_144x144x160grid_split{i}.npy"
    else:
        ldos_name = f"{poly}_snapshot-{snapshot}_shuffled2_ldos_144x144x160grid_split{i}.npy"
        fp_name = f"{poly}_snapshot-{snapshot}_shuffled2_fp_144x144x160grid_split{i}.npy"
elif DATASET == 'full':
    if shuffle_type == 's2':
        ldos_name = f"{poly}_snapshot-{snapshot}_shuffled2_ldos_288x288x320grid.npy"
        fp_name = f"{poly}_snapshot-{snapshot}_shuffled2_fp_288x288x320grid.npy"
    else:
        pass
else:
    raise ValueError('Invalid DATASET flag')

### Select initial training & validation data (assumes already shuffled)
vector_len = int(np.prod(data_handler.descriptor_calculator.read_dimensions_from_numpy_file(f'{ldos_path}/{ldos_name}')[:-1]))

 # Check generation size for validity
if GEN0SIZE == 0: # use entire dataset (equivalent to no active learning)
    GEN0SIZE = vector_len//2
if GENSIZE > (vector_len/2) or GEN0SIZE > (vector_len/2):
    raise ValueError('Generation size must be less than or equal to half of dataset size bc GENSIZE examples are taken for both training and validation')

unseen_idxs = np.arange(vector_len)
 # One gen for training
tr_idxs = unseen_idxs[:GEN0SIZE]
tr_mask = np.zeros(vector_len, dtype=bool) 
tr_mask[tr_idxs] = True
 # One gen for validation
va_idxs = unseen_idxs[GEN0SIZE:2*GEN0SIZE]
va_mask = np.zeros(vector_len, dtype=bool) 
va_mask[va_idxs] = True
 # Use the rest of the data as testing
unseen_idxs = unseen_idxs[2*GEN0SIZE:]
te_mask = np.zeros(vector_len, dtype=bool) 
te_mask[unseen_idxs] = True

### Load training data
data_type = 'tr'
data_handler.add_snapshot(fp_name, snap_path, ldos_name, ldos_path,
                          add_snapshot_as=data_type, output_units="1/(Ry*Bohr^3)",
                          selection_mask=tr_mask)
### Load testing data
data_type = 'te'
data_handler.add_snapshot(fp_name, snap_path, ldos_name, ldos_path,
                          add_snapshot_as=data_type, output_units="1/(Ry*Bohr^3)",
                          selection_mask=te_mask)
### Load validation data
data_type = 'va'
data_handler.add_snapshot(fp_name, snap_path, ldos_name, ldos_path,
                          add_snapshot_as=data_type, output_units="1/(Ry*Bohr^3)",
                          selection_mask=va_mask)


### Load novel data (untouched data used only for testing)
data_type = 'te'
snapshot = VASNAP_FLAG
i = CHUNKSTART_FLAG
if DATASET == 'small':
    ldos_name = f"test_aqrtz_snapshot-1_split0_ldos_1.npy"
    fp_name = f"test_aqrtz_snapshot-1_split0_fp_1.npy"
elif DATASET == 'tiny':
    ldos_name = f"tinytest_aqrtz_snapshot-1_split0_ldos_1.npy"
    fp_name = f"tinytest_aqrtz_snapshot-1_split0_fp_1.npy"
elif DATASET == 'big':
    if shuffle_type == 's2':
        ldos_name = f"{poly}_snapshot-{snapshot}_shuffled2_ldos_144x144x160grid_split{i}.npy"
        fp_name = f"{poly}_snapshot-{snapshot}_shuffled2_fp_144x144x160grid_split{i}.npy"
    else:
        ldos_name = f"{poly}_snapshot-{snapshot}_shuffled2_ldos_144x144x160grid_split{i}.npy"
        fp_name = f"{poly}_snapshot-{snapshot}_shuffled2_fp_144x144x160grid_split{i}.npy"
elif DATASET == 'full':
    if shuffle_type == 's2':
        ldos_name = f"{poly}_snapshot-{snapshot}_shuffled2_ldos_288x288x320grid.npy"
        fp_name = f"{poly}_snapshot-{snapshot}_shuffled2_fp_288x288x320grid.npy"
    else:
        pass
else:
    raise ValueError('Invalid DATASET flag')

data_handler.add_snapshot(fp_name, snap_path, ldos_name, ldos_path,
                          add_snapshot_as=data_type, output_units="1/(Ry*Bohr^3)")#,
data_handler.prepare_data(reparametrize_scaler=DO_REPARAM) #only reparametrize if fresh data
data_handler.input_data_scaler.save("aq.iscaler.pkl")
data_handler.output_data_scaler.save("aq.oscaler.pkl")
printout(f"=== Time for loading: {time.time()-start} s"); sys.stdout.flush()
#printout(f"=== Time for loading_novel_data: {time.time()-start} s"); sys.stdout.flush()


### Instantiate network
parameters.network.layer_sizes = [data_handler._input_dimension, L1_FLAG, L2_FLAG, L3_FLAG, 250,
                                  data_handler._output_dimension]
printout(f'actual_layer_sizes : {parameters.network.layer_sizes}')
network = mala.Network(parameters)


### Active learning loop
generation = 0
converged = False
overall_start = time.time()

while not converged:
    gen_start = time.time()

    ### Train network.
    start = time.time()
    trainer = mala.Trainer(parameters, network, data_handler)
    trainer.train_network()
    
    ### Save parameters.
    #trainer.save_run(f'aq_{generation}', zip_run=False)
    network.save_network(f'aq_{generation}.network.pth')
    printout(f"=== Time for training: {time.time()-start} s"); sys.stdout.flush()
    start = time.time()

    ### Calculate and save unseen_test residuals
    resid_test_name = 'resid_means_unseen_test.npy'
    resid_train_name = 'resid_means_unseen_train.npy'
    tester = mala.Tester(parameters, network, data_handler)
    actual, predicted = tester.predict_targets(1)
    resid_means = np.mean(predicted - actual, axis=-1).flatten()
    #resid_stds = np.std(predicted - actual, axis=-1).flatten()
    print(f'len unseen_test:  {len(resid_means)}', flush=True)
    print(f'max unseen_test resid_mean:  {max(resid_means)}', flush=True)
    print(f'mean unseen_test resid_mean: {np.mean(resid_means)}', flush=True)
    print(f'min unseen_test resid_mean:  {min(resid_means)}', flush=True)
    print(f'std unseen_test resid_mean:  {np.std(resid_means)}', flush=True)
    if generation == 0:
        os.makedirs('past', exist_ok=True)
        if os.path.exists(resid_test_name): os.rename(resid_test_name, f'past/{resid_test_name}')
    write_hist(resid_means, resid_test_name)
    #write_hist(resid_stds, f'resid_stds_unseen_test.npy')
    loss = mse_loss(torch_from_numpy(predicted), torch_from_numpy(actual))
    print(f'Final novel data loss:  {loss}')

    ### Update initial parameters, reset changing parameters
    #if generation >= 1:
    #    parameters.running.learning_rate = LR_FLAG

    ### Decay Learning Rate
    parameters.running.learning_rate = parameters.running.learning_rate*LRDECAY_FLAG
    print(f'Updated LR: {parameters.running.learning_rate}')
    # WHY isn't this actually changing?
    


    ### Check for out of examples
    if unseen_idxs.size < 1: 
        converged = True
        printout(f"=== Finished: ran out of training examples.")
    else:
        ### Select new data
         # If we are running out of examples, use the rest
        if unseen_idxs.size < 2*GENSIZE:
            remaining = unseen_idxs.size
            if CUMU_FRAC > 0:
                rng = np.random.default_rng(seed=SEED_FLAG)
                retain_tr_idxs = tr_idxs[rng.permutation(tr_idxs.size)[:int(CUMU_FRAC*tr_idxs.size)]]
                retain_va_idxs = va_idxs[rng.permutation(va_idxs.size)[:int(CUMU_FRAC*va_idxs.size)]]
                tr_idxs = np.append(retain_tr_idxs,unseen_idxs[:remaining//2])
                va_idxs = np.append(retain_va_idxs,unseen_idxs[remaining//2:])
            else:
                tr_idxs = unseen_idxs[:remaining//2]
                va_idxs = unseen_idxs[remaining//2:]
            unseen_idxs = np.array([], dtype=int)

         # Otherwise find N examples with largest loss
        else:
            ### Calculate and save unseen_training residuals
            start = time.time()
            actual, predicted = tester.predict_targets(0)
            resid_means = np.mean(predicted - actual, axis=-1).flatten()
            #resid_stds = np.std(predicted - actual, axis=-1).flatten()
            print(f'max unseen_training resid_mean:  {max(resid_means)}', flush=True)
            print(f'mean unseen_training resid_mean: {np.mean(resid_means)}', flush=True)
            print(f'min unseen_training resid_mean:  {min(resid_means)}', flush=True)
            print(f'std unseen_training resid_mean:  {np.std(resid_means)}', flush=True)
            if generation == 0:
                os.makedirs('past', exist_ok=True)
                if os.path.exists(resid_train_name): os.rename(resid_train_name, f'past/{resid_train_name}')
            write_hist(resid_means, f'resid_means_unseen_training.npy')
            #write_hist(resid_stds, f'resid_stds_unseen_training.npy')

            ### Select highest-loss samples
            #losses = np.mean((predicted - actual)**2, axis=-1).flatten()
            losses = np.max((predicted - actual)**2, axis=-1).flatten()
            argpart = np.argpartition(losses, -2*GENSIZE, axis=0).astype(int)
            #print(argpart.dtype)
            #print(argpart.shape) 

            ### Turn off the active learning --> just sequential batch learning
            if RAND:
                rng = np.random.default_rng(seed=SEED_FLAG)
                argpart = rng.permutation(len(losses))
            #########################################################################
            #print(argpart[-2*GENSIZE+1::2].dtype)
            #print(f'tr_idxs:     {tr_idxs}')
            #print(f'unseen_idxs: {unseen_idxs}')
            #print(f'tr argpart:  {argpart[-2*GENSIZE+1::2]}')
            new_tr_idxs = unseen_idxs[argpart[-2*GENSIZE+1::2]]
            new_va_idxs = unseen_idxs[argpart[-2*GENSIZE::2]]
            unseen_idxs = unseen_idxs[argpart[:-2*GENSIZE]]
            #print(f'new_tr_idxs: {new_tr_idxs}')
            #print(f'new_va_idxs: {new_va_idxs}')
            if CUMU_FRAC > 0:
                print('=== Accumulating data')
                rng = np.random.default_rng(seed=SEED_FLAG)
                #print(f'pre-cumu len tr_idxs:      {len(tr_idxs)}, uniq: {len(set(tr_idxs))}')
                #print(f'pre-cumu len va_idxs:      {len(va_idxs)}, uniq: {len(set(va_idxs))}')
                retain_tr_idxs = tr_idxs[rng.permutation(tr_idxs.size)[:int(CUMU_FRAC*tr_idxs.size)]]
                retain_va_idxs = va_idxs[rng.permutation(va_idxs.size)[:int(CUMU_FRAC*va_idxs.size)]]
                tr_idxs = np.append(retain_tr_idxs, new_tr_idxs)
                va_idxs = np.append(retain_va_idxs, new_va_idxs)
                #tr_idxs = np.append(tr_idxs,new_tr_idxs)
                #va_idxs = np.append(va_idxs,new_va_idxs)
                #print(f'tr_idxs after:  {tr_idxs}')
                #print(f'va_idxs after:  {va_idxs}')
                #print(f'post-cumu len tr_idxs:     {len(tr_idxs)}, uniq: {len(set(tr_idxs))}')
                #print(f'post-cumu len va_idxs:     {len(va_idxs)}, uniq: {len(set(va_idxs))}')
            else:
                tr_idxs = new_tr_idxs
                va_idxs = new_va_idxs
            del new_tr_idxs, new_va_idxs
            

            #print(f'rand losses = {losses[:5]}', flush=True)
            #print(f'high losses = {sorted(losses[tr_idxs],reverse=True)[:5]}', flush=True)
            print(f'len tr_idxs:     {tr_idxs.size}')
            print(f'len va_idxs:     {va_idxs.size}')
            print(f'len unseen_idxs: {unseen_idxs.size}')

            printout(f"=== Time for prediction+selection: {time.time()-start} s"); sys.stdout.flush()
            start = time.time()

        ### Update Training mask
        printout(f"Old tr mask ({sum(tr_mask)}): {tr_mask[:10]}"); sys.stdout.flush()
        tr_mask = np.zeros(vector_len, dtype=bool) 
        tr_mask[tr_idxs] = True
        for snapshot in data_handler.parameters.snapshot_directories_list:
            if snapshot.snapshot_function == 'tr' and snapshot.selection_mask is not None:
                snapshot.set_selection_mask(tr_mask)
        printout(f"New tr mask ({sum(tr_mask)}): {tr_mask[:10]}"); sys.stdout.flush()
        ### Update Validation mask
        va_mask = np.zeros(vector_len, dtype=bool) 
        va_mask[va_idxs] = True
        for snapshot in data_handler.parameters.snapshot_directories_list:
            if snapshot.snapshot_function == 'va' and snapshot.selection_mask is not None:
                snapshot.set_selection_mask(va_mask)
        printout(f"New va mask ({sum(va_mask)}): {va_mask[:10]}"); sys.stdout.flush()
        ### Update Testing mask
        te_mask = np.zeros(vector_len, dtype=bool) 
        te_mask[unseen_idxs] = True
        for snapshot in data_handler.parameters.snapshot_directories_list:
            if snapshot.snapshot_function == 'te' and snapshot.selection_mask is not None:
                 snapshot.set_selection_mask(te_mask)
        printout(f"New te mask ({sum(te_mask)}): {te_mask[:10]}"); sys.stdout.flush()
        data_handler.prepare_data(reparametrize_scaler=False, refresh=True) #only reparametrize if fresh data
        printout(f"=== Time for data_refresh: {time.time()-start} s"); sys.stdout.flush()
        # if unseen_loss...: 
        #     converged = True
        #     printout(f"=== Finished: loss converged.")

    generation += 1
    printout(f"=== Time for generation: {time.time()-gen_start} s"); sys.stdout.flush()

### Save final model
trainer.save_run(f'final_aq', zip_run=False)
printout(f"=== Time for active_learning_loop: {time.time()-overall_start} s")
