import numpy as np

import os, sys
import re
from glob import glob
import matplotlib
import matplotlib.pyplot as plt

#import ldos_calc
import DFT_calculators
from functools import partial
import argparse

import horovod.torch as hvd

from scipy.stats import gaussian_kde

import torch

import pickle

sys.path.append("../fp_ldos_feedforward/src")
import fp_ldos_networks
import train_networks
import data_loaders

sys.path.append("../fp_ldos_feedforward/src/charm/clustering")
import cluster_fingerprints

sys.path.append("../fp_ldos_feedforward/src/charm")
import big_data


# Training settings
parser = argparse.ArgumentParser(description='FP-LDOS Feedforward Network')

# Training
parser.add_argument('--temp', type=str, default="298K", metavar='T',
                    help='temperature of snapshot to train on (default: "298K")')
parser.add_argument('--gcc', type=str, default="2.699", metavar='GCC',
                    help='density of snapshot to train on (default: "2.699")')
parser.add_argument('--elvls', type=int, default=250, metavar='ELVLS',
                    help='number of ldos energy levels (default: 250)')
parser.add_argument('--nxyz', type=int, default=200, metavar='GCC',
                    help='number of x/y/z elements (default: 200)')
parser.add_argument('--dos-snapshot', type=int, default=2, metavar='N',
                    help='snapshot of temp/gcc with DOS file for egrid (default: 2)')
parser.add_argument('--max-snapshots', type=int, default=10, metavar='N',
                    help='max number of inference snapshots due to memory restrictions (default: 10)')
parser.add_argument('--snapshot-offset', type=int, default=0, metavar='N',
                    help='which snapshot to begin inference (default: 0)')
#parser.add_argument('--num-atoms', type=int, default=256, metavar='N',
#                    help='number of atoms in the snapshot (default: 256)')
parser.add_argument('--num-slices', type=int, default=1, metavar='N',
                    help='number of density z slices to plot (default: 1)')
#parser.add_argument('--no_convert', action='store_true', default=False,
#                    help='Do not Convert units from Rydbergs')
#parser.add_argument('--log', action='store_true', default=False,
#                    help='Apply log function to densities')

parser.add_argument('--cpu', action='store_true', default=False,
                    help='Run inference of a GPU trained model on a CPU machine.')
parser.add_argument('--integration', type=str, default="analytic", metavar='T',
                    choices=["analytic", "trapz", "simps", "quad"],
                    help='type of integration from {"trapz", "simps", "quad", "analytic"} (default: "analytic")')

# Directory Locations
parser.add_argument('--pred-dir', type=str, \
            default="../fp_ldos_feedforward/output", \
            metavar="str", help='path to ldos prediction directory (default: ../fp_ldos_feedforward/output)')
parser.add_argument('--ldos-dir', type=str, \
            default="../../training_data/ldos_data", \
            metavar="str", help='path to ldos data directory (default: ../../training_data/ldos_data)')
parser.add_argument('--fp-dir', type=str, \
            default="../../training_data/fp_data", \
            metavar="str", help='path to fp data (QE .out files) directory (default: ../../training_data/fp_data)')
parser.add_argument('--output-dir', type=str, \
            default="./inference_figs", \
            metavar="str", help='path to output directory (default: ./inference_figs)')

args = parser.parse_args()

hvd.init()


if (hvd.rank() == 0):
    print("\n\nBegin LDOS inference\n\n")


# Liquid
#fig_snapshots = [9]

# Solid
#fig_snapshots = [19]

# Hybrid
#fig_snapshots = [0, 19]

# All
fig_snapshots = [0,9,10,19]

tempv = int(args.temp[:-1])

DFT_calculators.temp = tempv
DFT_calculators.gcc  = float(args.gcc)
#print("Tempv: %d" % tempv)

#DFT_calculators.set_temp(tempv)
#DFT_calculators.set_gcc(args.gcc)

# Conversion factor from Rydberg to eV
#Ry2eV = ldos_calc.get_Ry2eV()
#ar_en = ldos_calc.get_energies(args.temp, args.gcc, args.elvls)


args.output_dirs = glob(args.pred_dir + "/*/")

if (hvd.rank() == 0):
    print("\nProcessing these model directories: ")

    for idx, current_dir in enumerate(args.output_dirs):
        print("Directory %d:  %s" % (idx, current_dir))

# DFT
#args.dft_files = sorted(glob(args.fp_dir + "/%s/%sgcc/QE*.out" % (args.temp, args.gcc)))
args.dft_files = glob(args.fp_dir + "/%s/%sgcc/QE*.out" % (args.temp, args.gcc))
args.dft_files.sort(key=lambda f: int(re.sub('\D', '', f)))

if (hvd.rank() == 0):
    print("\nAvailable DFT files: ")

    for idx, dft_file in enumerate(args.dft_files):
        print("DFT %d:  %s" % (idx, dft_file))

# FP
#args.fp_files = sorted(glob(args.fp_dir + "/%s/%sgcc/*fp*.npy" % (args.temp, args.gcc)))
args.fp_files = glob(args.fp_dir + "/%s/%sgcc/*fp*.npy" % (args.temp, args.gcc))
args.fp_files.sort(key=lambda f: int(re.sub('\D', '', f)))

if (hvd.rank() == 0):
    print("\nAvailable FP files: ")

    for idx, fp_file in enumerate(args.fp_files):
        print("FP %d:  %s" % (idx, fp_file))

# LDOS
#args.ldos_files = sorted(glob(args.ldos_dir + "/%s/%sgcc/*ldos*.npy" % (args.temp, args.gcc)))
args.ldos_files = glob(args.ldos_dir + "/%s/%sgcc/*ldos*.npy" % (args.temp, args.gcc))
args.ldos_files.sort(key=lambda f: int(re.sub('\D', '', f)))

if (hvd.rank() == 0):
    print("\nAvailable LDOS files: ")

    for idx, ldos_file in enumerate(args.ldos_files):
        print("LDOS %d:  %s" % (idx, ldos_file))


args.snapshots = np.min([len(args.fp_files), len(args.ldos_files)])


#if (args.snapshots > args.max_snapshots):
#    args.snapshots = args.max_snapshots


# Testing
#args.snapshot_offset = 0;
#args.snapshot_offset = 10;
#args.snapshots = 10;
#exit(0);

for i in range(args.snapshot_offset + args.max_snapshots, args.snapshots):
    del args.dft_files[args.snapshot_offset + args.max_snapshots]
    del args.fp_files[args.snapshot_offset + args.max_snapshots]
    del args.ldos_files[args.snapshot_offset + args.max_snapshots]

for i in range(args.snapshot_offset):
    del args.dft_files[0]
    del args.fp_files[0]
    del args.ldos_files[0]
    
if (args.snapshots > args.max_snapshots):
    args.snapshots = args.max_snapshots

if (hvd.rank() == 0):
    print("\nProcessing these DFT files: ")

    for idx, dft_file in enumerate(args.dft_files):
        print("DFT %d:  %s" % (idx, dft_file))
    
    print("\nProcessing these FP files: ")

    for idx, fp_file in enumerate(args.fp_files):
        print("FP %d:  %s" % (idx, fp_file))

    print("\nProcessing these LDOS files: ")

    for idx, ldos_file in enumerate(args.ldos_files):
        print("LDOS %d:  %s" % (idx, ldos_file))

#exit(0);




dft_results = []

true_dos = []
true_eband = []
true_enum = []

target_ldos = []


for i in range(args.snapshots):

    if (hvd.rank() == 0):
        print("\nCalculating True and Target DFT results for snapshot%d" % (i + args.snapshot_offset))

#    dft_fname         = args.fp_dir + "/%s/%sgcc/QE_Al.scf.pw.snapshot%d.out" % \
#            (args.temp, args.gcc, i)

#    target_ldos_fname = args.ldos_dir + "/%s/%sgcc/Al_ldos_%dx%dx%dgrid_%delvls_snapshot%d.npy" % \
#            (args.temp, args.gcc, args.nxyz, args.nxyz, args.nxyz, args.elvls, i)

    qe_dos_fname      = args.ldos_dir + "/%s/%sgcc/Al_dos_%delvls_snapshot%d.txt" % \
            (args.temp, args.gcc, args.elvls, args.dos_snapshot)

    if (hvd.rank() == 0):
        print("Calculating DFT Eigen Results.")

    ### DFT Eigen Results ###
    dft_results.append(DFT_calculators.DFT_results(args.dft_files[i]))

    if (os.path.exists(qe_dos_fname)):
        # QE Dos Results to get dos_e_grid
        qe_dos = DFT_calculators.DOS.from_dos_file(dft_results[i], qe_dos_fname)

        dos_e_grid = qe_dos.e_grid

    sigma = dos_e_grid[1] - dos_e_grid[0]
    # Smearing of 2sigma
    wide_gaussian = partial(DFT_calculators.gaussian, sigma = 2.0 * sigma)

    true_dos.append(DFT_calculators.DOS.from_calculation(dft_results[i], dos_e_grid, wide_gaussian))

    true_efermi = DFT_calculators.dos_2_efermi(true_dos[i], tempv, integration=args.integration)

#    print("ef1: ", true_efermi)

    true_eband.append(DFT_calculators.dft_2_eband(dft_results[i], e_fermi='sc', temperature=tempv))
    true_enum.append(DFT_calculators.dft_2_enum(dft_results[i], e_fermi='sc', temperature=tempv))
    #true_enum  = DFT_calculators.dft_2_enum(dft_results, e_fermi=true_efermi, temperature=tempv)

    #exit(0)





#    print("Calculating QE DOS Results.")

    ### QE DOS Results ###
#    qe_dos_efermi = DFT_calculators.dos_2_efermi(qe_dos, tempv, integration=args.integration)
#    qe_dos_eband = DFT_calculators.dos_2_eband(qe_dos, e_fermi=qe_dos_efermi, \
#                                               temperature=tempv, integration=args.integration)
#    qe_dos_enum  = DFT_calculators.dos_2_enum(qe_dos, e_fermi=qe_dos_efermi, \
#                                              temperature=tempv, integration=args.integration)

    if (hvd.rank() == 0):
        print("Calculating Target LDOS Results.")

    ### Target LDOS Results ###
    ldos_e_grid = dos_e_grid[:-1]

    target_ldos.append(DFT_calculators.LDOS(dft_results[i], ldos_e_grid, args.ldos_files[i]))
    target_ldos[i].do_calcs()


    print("TRUE BE: %4.4f, TARGET LDOS BE: %4.4f, DIFF %4.4f" % (true_eband[i], target_ldos[i].eband, (target_ldos[i].eband - true_eband[i])))

#target_dos = qe_ldos.dos

#target_dos_efermi = DFT_calculators.dos_2_efermi(target_dos, tempv, integration=args.integration)
#target_eband = DFT_calculators.dos_2_eband(target_dos, e_fermi=target_dos_efermi, temperature=tempv, integration=args.integration)
#target_enum = DFT_calculators.dos_2_enum(target_dos, e_fermi=target_dos_efermi, temperature=tempv, integration=args.integration)

if (hvd.rank() == 0):
    print("\n\nCalculating ML Predicted LDOS Results.")


#hvd.shutdown();
#exit(0);


#models_idx = 0
#models_limit = 1

for idx, current_dir in enumerate(args.output_dirs):

    if (hvd.rank() == 0):
        print("\n\nPerforming inference for Model %d from output_dir: %s" % (idx, current_dir))

#    pred_ldos_fname = current_dir + "fp_ldos_predictions.npy"
    pred_ldos_fname = current_dir + "fp_ldos_model.pth"


    if ("fp_ldos_dir" in current_dir):
        print("Found dir: %s. Processing." % current_dir)
    elif (not os.path.exists(pred_ldos_fname)):
        if (hvd.rank() == 0):
            print("Skipped %s! No predictions." % current_dir)
        continue;

#    if (models_idx >= models_limit):
#        continue

#    models_idx += 1

    # ML Predictions
#    pred_ldos = np.load(current_dir + "fp_ldos_predictions.npy")

#    pred_ldos = DFT_calculators.LDOS(dft_results, ldos_e_grid, pred_ldos_fname)


#    if (pred_ldos.ldos.shape[3] != args.elvls):
#        if (hvd.rank() == 0):
#            print("Skipped %s! Bad elvls." % current_dir)
#        continue;

    fp_scaler_fname = glob(current_dir + "charm_input*")
    ldos_scaler_fname = glob(current_dir + "charm_output*")

#    fp_factor_fname = glob(current_dir + "fp_row*.npy") 
#    ldos_factor_fname = glob(current_dir + "ldos_*.npy")
#    shift_fname = glob(current_dir + "log_shift.npy")

    if (hvd.rank() == 0):
        print("Found fp scaler file: %s" % fp_scaler_fname[0])
        print("Found ldos scaler file: %s" % ldos_scaler_fname[0])

    # Normalization factors
    fp_scaler = torch.load(fp_scaler_fname[0])
    ldos_scaler = torch.load(ldos_scaler_fname[0])


#    fp_row_norms = "row" in fp_factor_fname[0]
#    fp_minmax_norms = "max" in fp_factor_fname[0]

#    ldos_row_norms = "row" in ldos_factor_fname[0]
#    ldos_minmax_norms = "max" in ldos_factor_fname[0]


    model_fpath = current_dir + "fp_ldos_model.pth"
    args_fpath = current_dir + "commandline_args.pth"
#    args_fpath = current_dir + "commandline_args.pkl"

    if (os.path.exists(model_fpath) and os.path.exists(args_fpath)):
        if (hvd.rank() == 0):
            print("Found model: %s, args: %s" % (model_fpath, args_fpath))
    else:
        raise ValueError('No model/args path found.')

    model_args = torch.load(args_fpath)
   
    if (args.cpu):
        model = torch.load(model_fpath, map_location='cpu')
        model_args.cuda = False
    else:
        model = torch.load(model_fpath)
    
    model = model.eval()

#    margs_file = open(args_fpath, "rb")
#    model_args = pickle.load(open(args_fpath, "rb"))
#    margs_file.close()


#    print(model_args.fp_length)

#    exit(0);


    old_fp = None
    old_ldos = None
    predictions = None
    inference_fp = None

    for i in range(args.snapshots):

        if (hvd.rank() == 0):
            print("\nWorking on Snapshot%d" % (i + args.snapshot_offset))

        old_fp = inference_fp

        # Loading FP and transforming
        inference_fp = np.load(args.fp_files[i])

        # Remove Coords
        inference_fp = inference_fp[:,:,:, 3:] 
#        inference_fp = np.reshape(inference_fp, [model_args.grid_pts, model_args.fp_length])
        inference_fp = np.reshape(inference_fp, [args.nxyz ** 3, 91])

#        print(fp_factors.shape)


        if (hvd.rank() == 0):
            print("\nTransforming model input FPs")
        
#        for row in range(model_args.fp_length):
#            inference_fp[:, row] = (inference_fp[:, row] - fp_factors[0, row]) / fp_factors[1, row]

        inference_fp = fp_scaler.do_scaling_sample(inference_fp)


        if (False and i > 0):
            print("Old/New FP Diffs: ")

            for row in range(model_args.fp_length):
                mind = np.min(abs(inference_fp[:, row] - old_fp[:, row]))
                mend = np.mean(abs(inference_fp[:, row] - old_fp[:, row]))
                maxd = np.max(abs(inference_fp[:, row] - old_fp[:, row]))
                
                print("MIN/MEAN/MAX DIFFS: %4.4f %4.4f %4.4f" % (mind, mend, maxd))



        inference_fp_dataset = torch.utils.data.TensorDataset(torch.tensor(inference_fp, dtype=torch.float32),
                                                              torch.ones([args.nxyz ** 3, 1], dtype=torch.float32))

        del inference_fp

#        inference_fp_dataset = torch.utils.data.TensorDataset(torch.tensor(inference_fp, dtype=torch.float32), 
#                                                              torch.ones([model_args.grid_pts, 1], dtype=torch.float32))

        if (hvd.rank() == 0):
            print("\nBuilding Sampler/Loader")
        
        inference_sampler = torch.utils.data.sampler.SequentialSampler(inference_fp_dataset)
        inference_loader = torch.utils.data.DataLoader(inference_fp_dataset, \
                                                       batch_size=model_args.test_batch_size, \
                                                       sampler=inference_sampler)

        hvd.allreduce(torch.tensor(0), name='barrier')

        # Running Model

        old_ldos = predictions

#        predictions = np.empty([model_args.grid_pts, model_args.ldos_length]) 
        predictions = np.empty([args.nxyz ** 3, model_args.ldos_length])
#        predictions = np.empty([200 ** 3, model_args.ldos_length])
        hidden_n = model.test_hidden
        data_idx = 0 

        for batch_idx, (data, target) in enumerate(inference_loader):
            
            if model_args.cuda:
                data, target = data.cuda(), target.cuda()

            output, hidden_n = model(data, hidden_n)

            hidden_n = hidden_n[0].detach(), hidden_n[1].detach()
            
            num_samples = output.shape[0]

            if (model_args.cuda):
                predictions[data_idx:data_idx + num_samples, :] = output.cpu().detach().numpy()
            else:
                predictions[data_idx:data_idx + num_samples, :] = output.detach().numpy()

            data_idx += num_samples

            if (batch_idx % (model_args.log_interval * 10) == 0 % (model_args.log_interval * 10) and hvd.rank() == 0):
                print("Test batch_idx %d of %d" % (batch_idx, len(inference_loader)))


        if (hvd.rank() == 0):
            print("Inference Done\n")

#        exit(0);

        if (False and i > 0):
            print("Old/New LDOS Diffs: ")

            for row in range(model_args.ldos_length):
                mind = np.min(abs(predictions[:, row] - old_ldos[:, row]))
                mend = np.mean(abs(predictions[:, row] - old_ldos[:, row]))
                maxd = np.max(abs(predictions[:, row] - old_ldos[:, row]))
                
                print("MIN/MEAN/MAX DIFFS: %4.4f %4.4f %4.4f" % (mind, mend, maxd))


#            print("Done")
#            exit(0);

        pred_ldos = DFT_calculators.LDOS(dft_results[i], ldos_e_grid, predictions)



        # Denormalizing LDOS
#        if (hvd.rank() == 0):
#            print("Denormalizing LDOS with row: %r, minmax: %r" % (ldos_row_norms, ldos_minmax_norms))

#        if (ldos_row_norms):
#            if(ldos_minmax_norms):
#                for row, (minv, maxv) in enumerate(np.transpose(ldos_factors)):
#                    pred_ldos.ldos[:, :, :, row] = (pred_ldos.ldos[:, :, :, row] * (maxv - minv)) + minv
#            else:
#                for row, (meanv, stdv) in enumerate(np.transpose(ldos_factors)):
#                    pred_ldos.ldos[:, :, :, row] = (pred_ldos.ldos[:, :, :, row] * stdv) + meanv
#        else:
#            if(ldos_minmax_norms):
#                for row, (minv, maxv) in enumerate(np.transpose(ldos_factors)):
#                    pred_ldos.ldos = (pred_ldos.ldos * (maxv - minv)) + minv
#            else:
#                for row, (meanv, stdv) in enumerate(np.transpose(ldos_factors)):
#                    pred_ldos.ldos = (pred_ldos.ldos * stdv) + meanv
#
#        if (len(shift_fname) != 0):
#
#            print("Reverting ldos log and shift")
#            log_shift = np.load(shift_fname[0])
#            pred_ldos.ldos = np.exp(pred_ldos.ldos) - log_shift


        if (hvd.rank() == 0):
            print("Denormalizing LDOS with charm_scaler")

        pred_ldos.ldos = ldos_scaler.undo_scaling_sample(pred_ldos.ldos)

        pred_ldos.do_calcs()


    #    pred_ldos = np.reshape(pred_ldos, target_ldos.shape)

        print("Pred LDOS shape: ", pred_ldos.ldos.shape)
        print("Target LDOS shape: ", target_ldos[i].ldos.shape)

        # Density predictions
        print("Pred_LDOS->Density")
        print("Pred_Density shape: ", pred_ldos.density.shape)
        print("Target_Density shape: ", target_ldos[i].density.shape)


        # DOS predictions
        print("Pred_LDOS->DOS") 
        print("Pred_DOS shape: ", pred_ldos.dos.dos.shape)
        print("Target_DOS shape: ", target_ldos[i].dos.dos.shape)


        # Band Energy predictions
    #    print("Pred_DOS->BandEnergy")
    #    pred_be = ldos_calc.dos_to_band_energy(pred_dos, args.temp, args.gcc, args.simple)
    #    print("Target_DOS->BandEnergy")
    #    target_be = ldos_calc.dos_to_band_energy(target_dos, args.temp, args.gcc, args.simple)


        # Convert to Rydbergs
    #    if (not args.no_convert):
    #
    #        pred_density = pred_density / Ry2eV
    #        true_density = true_density / Ry2eV
    #
    #        pred_dos = pred_dos / Ry2eV
    #        true_dos = true_dos / Ry2eV
    #
    #        pred_be = pred_be / (Ry2eV)
    #        true_be = true_be / (Ry2eV)

        # Pred/Target Error Results
        print("\nPred/Target Density Min: %f, %f" % (np.min(pred_ldos.density), np.min(target_ldos[i].density)))
        print("Pred/Target Density Max: %f, %f" % (np.max(pred_ldos.density), np.max(target_ldos[i].density)))
        print("Pred/Target Density Mean: %f, %f" % (np.mean(pred_ldos.density), np.mean(target_ldos[i].density)))

        print("\nError Density Min: ", np.min(abs(pred_ldos.density - target_ldos[i].density)))
        print("Error Density Max: ", np.max(abs(pred_ldos.density - target_ldos[i].density)))
        print("Error Density Mean: ", np.mean(abs(pred_ldos.density - target_ldos[i].density)))

        print("\nPred/Target Error DOS L1_norm: ", np.linalg.norm((target_ldos[i].dos.dos - pred_ldos.dos.dos), ord=1))
        print("Pred/Target Error DOS L2_norm: ", np.linalg.norm((target_ldos[i].dos.dos - pred_ldos.dos.dos), ord=2))
        print("Pred/Target Error DOS Linf_norm: ", np.linalg.norm((target_ldos[i].dos.dos - pred_ldos.dos.dos), ord=np.inf))
        
        #dos_rmse = 
        #print("Pred/Target Error DOS RMSE (Oxford): ", dos_rmse)

        print("\n\nBand Energy Comparisons")
        print("Pred BE: ", pred_ldos.eband)
        print("Target BE: ", target_ldos[i].eband)
    #    print("QE DOS BE: ", qe_dos_eband)
        print("DFT Eigen BE: ", true_eband[i])
        
        print("\nPred/Target BE diff: ", (target_ldos[i].eband - pred_ldos.eband))
        print("Pred/Target BE relative diff: ", (target_ldos[i].eband - pred_ldos.eband) / target_ldos[i].eband * 100)
        print("Pred/Target BE meV/Atom diff: ", (target_ldos[i].eband - pred_ldos.eband) / dft_results[i].num_atoms * 1000)

        print("\nPred/True Eigen BE diff: ", (true_eband[i] - pred_ldos.eband))
        print("Pred/True Eigen BE relative diff: ", (true_eband[i] - pred_ldos.eband) / true_eband[i] * 100)
        print("Pred/True Eigen BE meV/Atom diff: ", (true_eband[i] - pred_ldos.eband) / dft_results[i].num_atoms * 1000)

        print("\n\nElectron Num Comparisons")
        print("Pred ENUM: ", pred_ldos.enum)
        print("Target ENUM: ", target_ldos[i].enum)
    #    print("QE DOS ENUM: ", qe_dos_enum)
        print("DFT Eigen ENUM: ", true_enum[i])
        
        print("\nPred/Target ENUM diff: ", (target_ldos[i].enum - pred_ldos.enum))
        print("Pred/Target ENUM relative diff: ", (target_ldos[i].enum - pred_ldos.enum) / target_ldos[i].enum * 100)
        print("Pred/Target ENUM electron/Atom diff: ", (target_ldos[i].enum - pred_ldos.enum) / dft_results[i].num_atoms)

        print("\nPred/True Eigen ENUM diff: ", (true_enum[i] - pred_ldos.enum))
        print("Pred/True Eigen ENUM relative diff: ", (true_enum[i] - pred_ldos.enum) / true_enum[i] * 100)
        print("Pred/True Eigen ENUM electron/Atom diff: ", (true_enum[i] - pred_ldos.enum) / dft_results[i].num_atoms)


        file_format = 'jpg'

        if (i + args.snapshot_offset) in fig_snapshots:
            # Create output dir if it doesn't exist
            if not os.path.exists(args.output_dir):
                print("\nCreating output folder %s\n" % args.output_dir)
                os.makedirs(args.output_dir)
         
            matplotlib.rcdefaults()

            font = {'weight': 'normal', 'size': 13}
            matplotlib.rc('font', **font)
            
            # DOS and error plots
           
            # Truncate dos
            t_ldos_e_grid = ldos_e_grid[50:200]
            t_target_dos = target_ldos[i].dos.dos[50:200]
            t_pred_dos = pred_ldos.dos.dos[50:200]


            print("Fermi Energy: ", target_ldos[i].e_fermi)

            fig, (ax0, ax1) = plt.subplots(2,1)
            #ax0.plot(dos_e_grid, true_dos[i].dos, "-k")
            ax0.plot(t_ldos_e_grid, t_target_dos, "-k")
            ax0.plot(t_ldos_e_grid, t_pred_dos, "--r")
            ax0.plot([target_ldos[i].e_fermi, target_ldos[i].e_fermi], [80, 120], '-g')
            ax0.legend(["DFT LDOS Target", "ML-DFT Prediction", "Fermi Energy"])
            ax0.set_ylabel("DOS ($eV^{-1}$)")
            
            ax0.set_ylim([0, 200])
          
            #ax1.plot(ldos_e_grid, target_ldos[i].dos.dos - true_dos[i].dos[:-1], "-b")
            ax1.plot(t_ldos_e_grid, t_pred_dos - t_target_dos, "-r")
            ax1.legend(["ML-DFT Pred DOS Error"])
            ax1.set_xlabel("Energy (eV)")
            ax1.set_ylabel("DOS Error ($eV^{-1}$)")
            
            ax1.set_ylim([-2.0, 2.0])

            plt.tight_layout()
         
            dos_fname = "/pred_target_true_dos_model%d_snapshot%d.%s" % (idx, (i + args.snapshot_offset), file_format)

            plt.savefig(args.output_dir + dos_fname, format=file_format)
        
            print("\nDOS plot created, Model: %d, Snapshot: %d" % (idx, (i + args.snapshot_offset)))
       


            # Density Difference plots
            fig, ax = plt.subplots(1, 1)

            target_density = np.reshape(target_ldos[i].density, [args.nxyz ** 3])
            pred_density = np.reshape(pred_ldos.density, [args.nxyz ** 3])

            target_max = np.max(target_density)
            target_max = .055
           

            density_err_max = np.max(np.abs(target_density - pred_density))
            density_err_mean = np.mean(np.abs(target_density - pred_density))
            density_err_std = np.std(np.abs(target_density - pred_density))

            print("Density error max: ", density_err_max)
            print("Density error mean: ", density_err_mean)
            print("Density error std: ", density_err_std)


            # Color heat map
            idxs = np.arange(200**3)
            idxs = np.random.choice(idxs, 10000, replace=False)

            x_mod = target_density[idxs]
            y_mod = pred_density[idxs]

            xy = np.vstack([x_mod, y_mod])
            z = gaussian_kde(xy)(xy)
            



#            ax.plot(target_ldos.density, target_ldos.density / target_ldos.density, 'k-')
#            ax.scatter(target_density, pred_density, c=z, s=100, edgecolor='')
            im = ax.scatter(x_mod, y_mod, c=z, edgecolor='')
            ax.plot([0, target_max], [0, target_max], 'k-')
            ax.legend(['DFT LDOS Target Density', 'ML-DFT Pred Density Errors'])
            ax.set_xlabel('DFT LDOS Target Electron Density ($e^{-}$/$A^{3}$)')
            ax.set_ylabel('ML-DFT Pred Electron Density ($e^{-}$/$A^{3}$)')

#            cbar = fig.colorbar(im, ax=ax)

            ax.set_xlim([0, target_max])
            ax.set_ylim([0, target_max])

            plt.tight_layout()

            density_fname = "/pred_target_density_diffs_model%d_snapshot%d.%s" % (idx, (i + args.snapshot_offset), file_format)

            plt.savefig(args.output_dir + density_fname, format=file_format)

            print("\nDensity diff plot created, Model: %d, Snapshot: %d" % (idx, (i + args.snapshot_offset)))

#            exit(0);

          # Density Error Slice Plots
 
#          font = {'weight' : 'bold',
#                  'size'   : 40}
#
#          matplotlib.rc('font', **font)
#
#          z_slices = np.round(np.linspace(0, target_ldos.ldos.shape[2] - 1, args.num_slices)).astype(int)
#
#          fig, ax = plt.subplots(args.num_slices,1)
#
#          fig.set_figheight(20 * args.num_slices)
#          fig.set_figwidth(20)
#
#          cbar_err_max = np.max(np.abs(pred_ldos.density - target_ldos.density))
#
#          for i in range(args.num_slices):
#              
#              gs = pred_ldos.cell_volume ** (1/3.)
#
#              xgrid = np.linspace(0, target_ldos.ldos.shape[0], target_ldos.ldos.shape[0]) * gs
#              ygrid = np.linspace(0, target_ldos.ldos.shape[1], target_ldos.ldos.shape[1]) * gs
#
#              density_slice = np.abs(pred_ldos.density[:,:,z_slices[i]] - target_ldos.density[:,:,z_slices[i]])
#
#              if (args.log):
#                  density_slice = np.log(density_slice)
#
#              im = ax[i].contourf(xgrid, ygrid, density_slice, cmap="seismic")
#              ax[i].set_title("Absolute Density Error Z-Slice at %3.1f" % (z_slices[i] * gs))
#
#              cbar = fig.colorbar(im, ax=ax[i])
#              cbar.set_clim(0.0, cbar_err_max)
#
#          #plt.tight_layout()
#          
#          
#          if (args.log):
#              density_fname = "/pred_target_error_density_log_model%d_snapshot%d.eps" % (idx, i)
#          else:
#              density_fname = "/pred_target_error_density_model%d_snapshot%d.eps" % (idx, i)
#
#          plt.savefig(args.output_dir + density_fname, format='eps')
#
#
#
#          vmin_plot = 0.0
#          vmax_plot = np.max([np.max(target_ldos.density), np.max(pred_ldos.density)])
#
#          fig, ax = plt.subplots(args.num_slices,1)
#
#          fig.set_figheight(20 * args.num_slices)
#          fig.set_figwidth(20)
#
#          for i in range(args.num_slices):
#              
#              gs = pred_ldos.cell_volume ** (1/3.) 
#
#              xgrid = np.linspace(0, pred_ldos.ldos.shape[0], pred_ldos.ldos.shape[0]) * gs
#              ygrid = np.linspace(0, pred_ldos.ldos.shape[1], pred_ldos.ldos.shape[1]) * gs
#
#              density_slice = pred_ldos.density[:,:,z_slices[i]] 
#
#              if (args.log):
#                  density_slice = np.log(density_slice)
#
#              im = ax[i].contourf(xgrid, ygrid, density_slice, vmin=vmin_plot, vmax=vmax_plot, cmap="seismic")
#              ax[i].set_title("Pred Density Z-Slice at %3.1f" % (z_slices[i] * gs))
#
#              cbar = fig.colorbar(im, ax=ax[i])
#              cbar.set_clim(0.0, vmax_plot)
#
#          #plt.tight_layout()
#          
#
#          if (args.log):
#              density_fname = "/pred_density_log_model%d_snapshot%d.eps" % (idx, i)
#          else:
#              density_fname = "/pred_density_model%d_snapshot%d.eps" % (idx, i)
#
#          plt.savefig(args.output_dir + density_fname, format='eps')
#
#
#
            
#            fig, ax = plt.subplots(args.num_slices,1)

#            z_slices = np.array([100])

#            fig.set_figheight(20 * args.num_slices)
#            fig.set_figwidth(20)

#            for i in range(args.num_slices):
              
#                gs = target_ldos.cell_volume ** (1/3.) 

#                xgrid = np.linspace(0, target_ldos.ldos.shape[0], target_ldos.ldos.shape[0]) * gs
#                ygrid = np.linspace(0, target_ldos.ldos.shape[1], target_ldos.ldos.shape[1]) * gs

#                density_slice = target_ldos.density[:,:,z_slices[i]] 

          #    if (args.log):
                  #        density_slice = np.log(density_slice)

                #im = ax[i].contourf(xgrid, ygrid, density_slice, vmin=vmin_plot, vmax=vmax_plot, cmap="seismic")
#                im = ax[i].contourf(xgrid, ygrid, density_slice, cmap="seismic")
                #ax[i].set_title("Target Density Z-Slice at %3.1f" % (z_slices[i] * gs))

#                cbar = fig.colorbar(im, ax=ax[i])
#                cbar.set_clim(0.0, vmax_plot)

#            plt.tight_layout()
          
          
          #if (args.log):
              #              density_fname = "/target_density_log_model%d_snapshot%d.eps" % (idx, i)
#          else:
            
#            density_fname = "/target_density_model%d_snapshot%d.%s" % (idx, i, file_format)    

#            plt.savefig(args.output_dir + density_fname, format=file_format)

#          print("Density plots created")

            plt.close('all')

#            break


print("\n\nSuccess!\n\n")




hvd.shutdown()
