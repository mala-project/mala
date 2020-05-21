import numpy as np

import os
from glob import glob
import matplotlib
import matplotlib.pyplot as plt

#import ldos_calc
import DFT_calculators
from functools import partial
import argparse


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
parser.add_argument('--snapshot', type=int, default=2, metavar='N',
                    help='snapshot of temp/gcc pair to compare (default: 2)')
#parser.add_argument('--num-atoms', type=int, default=256, metavar='N',
#                    help='number of atoms in the snapshot (default: 256)')
parser.add_argument('--num-slices', type=int, default=4, metavar='N',
                    help='number of density z slices to plot (default: 4)')
#parser.add_argument('--no_convert', action='store_true', default=False,
#                    help='Do not Convert units from Rydbergs')
parser.add_argument('--log', action='store_true', default=False,
                    help='Apply log function to densities')
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
            default="./prediction_figs", \
            metavar="str", help='path to output directory (default: ./prediction_figs)')

args = parser.parse_args()



print("\n\nBegin Prediction/Target LDOS comparisons\n\n")

tempv = int(args.temp[:-1])
#print("Tempv: %d" % tempv)

#DFT_calculators.set_temp(tempv)
#DFT_calculators.set_gcc(args.gcc)

# Conversion factor from Rydberg to eV
#Ry2eV = ldos_calc.get_Ry2eV()
#ar_en = ldos_calc.get_energies(args.temp, args.gcc, args.elvls)


args.output_dirs = glob(args.pred_dir + "/*/")

print("Processing these directories: ")

for idx, current_dir in enumerate(args.output_dirs):
    print("Directory %d:  %s" % (idx, current_dir))

#exit(0)


dft_fname         = args.fp_dir + "/%s/%sgcc/QE_Al.scf.pw.snapshot%d.out" % \
        (args.temp, args.gcc, args.snapshot)

target_ldos_fname = args.ldos_dir + "/%s/%sgcc/Al_ldos_%dx%dx%dgrid_%delvls_snapshot%d.npy" % \
        (args.temp, args.gcc, args.nxyz, args.nxyz, args.nxyz, args.elvls, args.snapshot)

qe_dos_fname      = args.ldos_dir + "/%s/%sgcc/Al_dos_%delvls_snapshot%d.txt" % \
        (args.temp, args.gcc, args.elvls, args.snapshot)


print("\nCalculating DFT Eigen Results.")

### DFT Eigen Results ###
dft_results = DFT_calculators.DFT_results(dft_fname)

# QE Dos Results to get dos_e_grid
qe_dos = DFT_calculators.DOS.from_dos_file(dft_results, qe_dos_fname)

dos_e_grid = qe_dos.e_grid

sigma = dos_e_grid[1] - dos_e_grid[0]
wide_gaussian = partial(DFT_calculators.gaussian, sigma = 2.0 * sigma)

true_dos   = DFT_calculators.DOS.from_calculation(dft_results, dos_e_grid, wide_gaussian)

true_efermi = DFT_calculators.dos_2_efermi(true_dos, tempv, integration=args.integration)

print("ef1: ", true_efermi)

true_eband = DFT_calculators.dft_2_eband(dft_results, e_fermi='sc', temperature=tempv)
true_enum  = DFT_calculators.dft_2_enum(dft_results, e_fermi=true_efermi, temperature=tempv)

#exit(0)





print("Calculating QE DOS Results.")

### QE DOS Results ###
qe_dos_efermi = DFT_calculators.dos_2_efermi(qe_dos, tempv, integration=args.integration)
qe_dos_eband = DFT_calculators.dos_2_eband(qe_dos, e_fermi=qe_dos_efermi, temperature=tempv, integration=args.integration)
qe_dos_enum  = DFT_calculators.dos_2_enum(qe_dos, e_fermi=qe_dos_efermi, temperature=tempv, integration=args.integration)

print("Calculating Target LDOS Results.")

### Target LDOS Results ###
ldos_e_grid = dos_e_grid[:-1]

target_ldos = DFT_calculators.LDOS(dft_results, ldos_e_grid, target_ldos_fname)
target_ldos.do_calcs()

#target_dos = qe_ldos.dos

#target_dos_efermi = DFT_calculators.dos_2_efermi(target_dos, tempv, integration=args.integration)
#target_eband = DFT_calculators.dos_2_eband(target_dos, e_fermi=target_dos_efermi, temperature=tempv, integration=args.integration)
#target_enum = DFT_calculators.dos_2_enum(target_dos, e_fermi=target_dos_efermi, temperature=tempv, integration=args.integration)

print("Calculating ML Predicted LDOS Results.")

for idx, current_dir in enumerate(args.output_dirs):

    print("\n\nGenerating comparison %d from output_dir: %s" % (idx, current_dir))

    pred_ldos_fname = current_dir + "fp_ldos_predictions.npy"


    if (not os.path.exists(pred_ldos_fname)):
        print("Skipped! No predictions.")
        continue;

    # ML Predictions
#    pred_ldos = np.load(current_dir + "fp_ldos_predictions.npy")

    pred_ldos = DFT_calculators.LDOS(dft_results, ldos_e_grid, pred_ldos_fname)


    if (pred_ldos.ldos.shape[3] != args.elvls):
        print("Skipped! Bad elvls.")
        continue;


    factor_fname = glob(current_dir + "ldos_*.npy")
    shift_fname = glob(current_dir + "log_shift.npy")

    print("Found ldos factor file: %s" % factor_fname[0])

    # Normalization factors
    ldos_factors = np.load(factor_fname[0])

    row_norms = "row" in factor_fname[0]
    minmax_norms = "max" in factor_fname[0]

    print("Denormalizing LDOS with row: %r, minmax: %r" % (row_norms, minmax_norms))

    if (row_norms):
        if(minmax_norms):
            for row, (minv, maxv) in enumerate(np.transpose(ldos_factors)):
                pred_ldos.ldos[:, :, :, row] = (pred_ldos.ldos[:, :, :, row] * (maxv - minv)) + minv
        else:
            for row, (meanv, stdv) in enumerate(np.transpose(ldos_factors)):
                pred_ldos.ldos[:, :, :, row] = (pred_ldos.ldos[:, :, :, row] * stdv) + meanv
    else:
        if(minmax_norms):
            for row, (minv, maxv) in enumerate(np.transpose(ldos_factors)):
                pred_ldos.ldos = (pred_ldos.ldos * (maxv - minv)) + minv
        else:
            for row, (meanv, stdv) in enumerate(np.transpose(ldos_factors)):
                pred_ldos.ldos = (pred_ldos.ldos * stdv) + meanv

    if (len(shift_fname) != 0):

        print("Reverting ldos log and shift")
        log_shift = np.load(shift_fname[0])
        pred_ldos.ldos = np.exp(pred_ldos.ldos) - log_shift

    
    pred_ldos.do_calcs()


#    pred_ldos = np.reshape(pred_ldos, target_ldos.shape)

    print("Pred LDOS shape: ", pred_ldos.ldos.shape)
    print("Target LDOS shape: ", target_ldos.ldos.shape)

    # Density predictions
    print("Pred_LDOS->Density")
    print("Pred_Density shape: ", pred_ldos.density.shape)
    print("Target_Density shape: ", target_ldos.density.shape)


    # DOS predictions
    print("Pred_LDOS->DOS") 
    print("Pred_DOS shape: ", pred_ldos.dos.dos.shape)
    print("Target_DOS shape: ", target_ldos.dos.dos.shape)


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
    print("\nPred/Target Density Min: %f, %f" % (np.min(pred_ldos.density), np.min(target_ldos.density)))
    print("Pred/Target Density Max: %f, %f" % (np.max(pred_ldos.density), np.max(target_ldos.density)))
    print("Pred/Target Density Mean: %f, %f" % (np.mean(pred_ldos.density), np.mean(target_ldos.density)))

    print("\nError Density Min: ", np.min(abs(pred_ldos.density - target_ldos.density)))
    print("Error Density Max: ", np.max(abs(pred_ldos.density - target_ldos.density)))
    print("Error Density Mean: ", np.mean(abs(pred_ldos.density - target_ldos.density)))

    print("\nPred/Target Error DOS L1_norm: ", np.linalg.norm((target_ldos.dos.dos - pred_ldos.dos.dos), ord=1))
    print("Pred/Target Error DOS L2_norm: ", np.linalg.norm((target_ldos.dos.dos - pred_ldos.dos.dos), ord=2))
    print("Pred/Target Error DOS Linf_norm: ", np.linalg.norm((target_ldos.dos.dos - pred_ldos.dos.dos), ord=np.inf))

    print("\n\nBand Energy Comparisons")
    print("Pred BE: ", pred_ldos.eband)
    print("Target BE: ", target_ldos.eband)
    print("QE DOS BE: ", qe_dos_eband)
    print("DFT Eigen BE: ", true_eband)
    
    print("\nPred/Target BE diff: ", abs(pred_ldos.eband - target_ldos.eband))
    print("Pred/Target BE relative diff: ", abs(pred_ldos.eband - target_ldos.eband) / target_ldos.eband)
    print("Pred/Target BE meV/Atom diff: ", (target_ldos.eband - pred_ldos.eband) / dft_results.num_atoms)

    print("\nPred/True Eigen BE diff: ", abs(pred_ldos.eband - true_eband))
    print("Pred/True Eigen BE relative diff: ", abs(pred_ldos.eband - true_eband) / true_eband)
    print("Pred/True Eigen BE meV/Atom diff: ", (true_eband - pred_ldos.eband) / dft_results.num_atoms)

    print("\n\nElectron Num Comparisons")
    print("Pred ENUM: ", pred_ldos.enum)
    print("Target ENUM: ", target_ldos.enum)
    print("QE DOS ENUM: ", qe_dos_enum)
    print("DFT Eigen ENUM: ", true_enum)
    
    print("\nPred/Target ENUM diff: ", (target_ldos.enum - pred_ldos.enum))
    print("Pred/Target ENUM relative diff: ", (target_ldos.enum - pred_ldos.enum) / target_ldos.enum * 100)
    print("Pred/Target ENUM electron/Atom diff: ", (target_ldos.enum - pred_ldos.enum) / dft_results.num_atoms)

    print("\nPred/True Eigen ENUM diff: ", (true_enum - pred_ldos.enum))
    print("Pred/True Eigen ENUM relative diff: ", (true_enum - pred_ldos.enum) / true_enum * 100)
    print("Pred/True Eigen ENUM electron/Atom diff: ", (true_enum - pred_ldos.enum) / dft_results.num_atoms)


    # Create output dir if it doesn't exist
    if not os.path.exists(args.output_dir):
        print("\nCreating output folder %s\n" % args.output_dir)
        os.makedirs(args.output_dir)


    # DOS and error plots

    fig, (ax0, ax1) = plt.subplots(2,1)
    ax0.plot(dos_e_grid, true_dos.dos, "-k")
    ax0.plot(ldos_e_grid, target_ldos.dos.dos, ":b")
    ax0.plot(ldos_e_grid, pred_ldos.dos.dos, "--g")
    ax0.legend(["True Eigen", "ML Target", "ML Prediction"])
    ax0.set_ylabel("DOS")

    ax1.plot(ldos_e_grid, target_ldos.dos.dos - true_dos.dos[:-1], "-b")
    ax1.plot(ldos_e_grid, pred_ldos.dos.dos - true_dos.dos[:-1], "-r")
    ax1.legend(["Target DOS Error", "Pred DOS Error"])
    ax1.set_xlabel("Energy (eV)")
    ax1.set_ylabel("DOS Error")

    plt.savefig(args.output_dir + "/pred_target_true_dos%d.eps" % idx, format='eps')

    print("\nDOS plot created")


    # Density Error Slice Plots

    font = {'weight' : 'bold',
            'size'   : 50}

    matplotlib.rc('font', **font)

    z_slices = np.round(np.linspace(0, target_ldos.ldos.shape[2] - 1, args.num_slices)).astype(int)

    fig, ax = plt.subplots(args.num_slices,1)

    fig.set_figheight(20 * args.num_slices)
    fig.set_figwidth(20)

    for i in range(args.num_slices):
        
        gs = pred_ldos.cell_volume ** (1/3.)

        xgrid = np.linspace(0, target_ldos.ldos.shape[0], target_ldos.ldos.shape[0]) * gs
        ygrid = np.linspace(0, target_ldos.ldos.shape[1], target_ldos.ldos.shape[1]) * gs

        density_slice = np.abs(pred_ldos.density[:,:,z_slices[i]] - target_ldos.density[:,:,z_slices[i]])

        if (args.log):
            density_slice = np.log(density_slice)

        im = ax[i].contourf(xgrid, ygrid, density_slice, cmap="seismic")
        ax[i].set_title("Absolute Density Error Z-Slice at %3.1f" % (z_slices[i] * gs))

        cbar = fig.colorbar(im, ax=ax[i])

    #plt.tight_layout()

    if (args.log):
        density_fname = "/pred_target_error_density_log%d.eps" % idx
    else:
        density_fname = "/pred_target_error_density%d.eps" % idx

    plt.savefig(args.output_dir + density_fname, format='eps')



    vmin_plot = 0.0
    vmax_plot = np.max([np.max(target_ldos.density), np.max(pred_ldos.density)])

    fig, ax = plt.subplots(args.num_slices,1)

    fig.set_figheight(20 * args.num_slices)
    fig.set_figwidth(20)

    for i in range(args.num_slices):
        
        gs = pred_ldos.cell_volume ** (1/3.) 

        xgrid = np.linspace(0, pred_ldos.ldos.shape[0], pred_ldos.ldos.shape[0]) * gs
        ygrid = np.linspace(0, pred_ldos.ldos.shape[1], pred_ldos.ldos.shape[1]) * gs

        density_slice = pred_ldos.density[:,:,z_slices[i]] 

        if (args.log):
            density_slice = np.log(density_slice)

        im = ax[i].contourf(xgrid, ygrid, density_slice, vmin=vmin_plot, vmax=vmax_plot, cmap="seismic")
        ax[i].set_title("Pred Density Z-Slice at %3.1f" % (z_slices[i] * gs))

        cbar = fig.colorbar(im, ax=ax[i])

    #plt.tight_layout()

    if (args.log):
        density_fname = "/pred_density_log%d.eps" % idx
    else:
        density_fname = "/pred_density%d.eps" % idx

    plt.savefig(args.output_dir + density_fname, format='eps')



    fig, ax = plt.subplots(args.num_slices,1)

    fig.set_figheight(20 * args.num_slices)
    fig.set_figwidth(20)

    for i in range(args.num_slices):
        
        gs = target_ldos.cell_volume ** (1/3.) 

        xgrid = np.linspace(0, target_ldos.ldos.shape[0], target_ldos.ldos.shape[0]) * gs
        ygrid = np.linspace(0, target_ldos.ldos.shape[1], target_ldos.ldos.shape[1]) * gs

        density_slice = target_ldos.density[:,:,z_slices[i]] 

        if (args.log):
            density_slice = np.log(density_slice)

        im = ax[i].contourf(xgrid, ygrid, density_slice, vmin=vmin_plot, vmax=vmax_plot, cmap="seismic")
        ax[i].set_title("Target Density Z-Slice at %3.1f" % (z_slices[i] * gs))

        cbar = fig.colorbar(im, ax=ax[i])

    #plt.tight_layout()

    if (args.log):
        density_fname = "/target_density_log%d.eps" % idx
    else:
        density_fname = "/target_density%d.eps" % idx    

    plt.savefig(args.output_dir + density_fname, format='eps')

    print("Density plots created")

    plt.close('all')

#    break


print("\n\nSuccess!\n\n")




