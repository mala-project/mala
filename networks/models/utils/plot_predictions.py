import numpy as np

import os
import matplotlib
import matplotlib.pyplot as plt

import ldos_calc
import argparse


# Training settings
parser = argparse.ArgumentParser(description='FP-LDOS Feedforward Network')

# Training
parser.add_argument('--temp', type=str, default="298K", metavar='T',
                    help='temperature of snapshot to train on (default: "298K")')
parser.add_argument('--gcc', type=str, default="2.699", metavar='GCC',
                    help='density of snapshot to train on (default: "2.699")')
parser.add_argument('--snapshot', type=int, default=2, metavar='N',
                    help='snapshot of temp/gcc pair to compare (default: 2)')
parser.add_argument('--num-slices', type=int, default=4, metavar='N',
                    help='number of density z slices to plot (default: 4)')
parser.add_argument('--convert', action='store_true', default=False,
                    help='Convert units to Rydbergs? (need to confirm)')
parser.add_argument('--simple', action='store_true', default=False,
                    help='Use trapezoid rule instead of simps rule')

# Directory Locations
parser.add_argument('--pred-dir', type=str, \
            default=".", \
            metavar="str", help='path to ldos prediction directory (default: .)')
parser.add_argument('--ldos-dir', type=str, \
            default="../../training_data/ldos_data", \
            metavar="str", help='path to ldos data directory (default: ../../training_data/ldos_data)')
parser.add_argument('--output-dir', type=str, \
            default="./figs", \
            metavar="str", help='path to output directory (default: ./figs)')

args = parser.parse_args()



print("\n\nBegin Prediction/True LDOS comparisons\n\n")


# Conversion factor from Rydberg to eV
Ry2eV = ldos_calc.get_Ry2eV()
ar_en = ldos_calc.get_energies(args.temp, args.gcc)


# ML Predictions
pred_snp = np.load(args.pred_dir + "/fp_ldos_predictions.npy")

# "True" LDOS values
true_snp = np.load(args.ldos_dir + "/%s/%sgcc/ldos_200x200x200grid_128elvls_snapshot%d.npy" % (args.temp, args.gcc, args.snapshot))

pred_snp = np.reshape(pred_snp, true_snp.shape)

print("Pred LDOS shape: ", pred_snp.shape)
print("True LDOS shape: ", true_snp.shape)

# Energy grid
ar_en = ldos_calc.get_energies(args.temp, args.gcc, e_lvls=true_snp.shape[-1])


# Density predictions
print("Pred_LDOS->Density")
pred_density = ldos_calc.ldos_to_density(pred_snp, args.temp, args.gcc, args.simple)
print("Pred_Density shape: ", pred_density.shape)

print("True_LDOS->Density")
true_density = ldos_calc.ldos_to_density(true_snp, args.temp, args.gcc, args.simple)
print("True_Density shape: ", true_density.shape)


# DOS predictions
print("Pred_LDOS->DOS")
pred_dos = ldos_calc.ldos_to_dos(pred_snp, args.temp, args.gcc, args.simple)
print("Pred_DOS shape: ", pred_dos.shape)

print("True_LDOS->DOS")
true_dos = ldos_calc.ldos_to_dos(true_snp, args.temp, args.gcc, args.simple)
print("True_DOS shape: ", true_dos.shape)


# Band Energy predictions
print("Pred_DOS->BandEnergy")
pred_be = ldos_calc.dos_to_band_energy(pred_dos, args.temp, args.gcc, args.simple)
print("True_DOS->BandEnergy")
true_be = ldos_calc.dos_to_band_energy(true_dos, args.temp, args.gcc, args.simple)


# Convert to Rydbergs
if (args.convert):

    pred_density = pred_density / Ry2eV
    true_density = true_density / Ry2eV

    pred_dos = pred_dos / Ry2eV
    true_dos = true_dos / Ry2eV

    pred_be = pred_be / (Ry2eV ** 2)
    true_be = true_be / (Ry2eV ** 2)

# Error Results
print("Error Density Min: ", np.min(abs(pred_density - true_density)))
print("Error Density Max: ", np.max(abs(pred_density - true_density)))
print("Error Density Mean: ", np.mean(abs(pred_density - true_density)))

print("Error DOS L1_norm: ", np.linalg.norm((true_dos - pred_dos), ord=1))
print("Error DOS L2_norm: ", np.linalg.norm((true_dos - pred_dos), ord=2))
print("Error DOS Linf_norm: ", np.linalg.norm((true_dos - pred_dos), ord=np.inf))

print("Pred BE: ", pred_be)
print("True BE: ", true_be)
print("BE diff: ", abs(pred_be - true_be))
print("BE relative diff: ", abs(pred_be - true_be)/true_be)


# Create output dir if it doesn't exist
if not os.path.exists(args.output_dir):
    print("\nCreating output folder %s\n" % args.output_dir)
    os.makedirs(args.output_dir)


# DOS and error plots

fig, (ax0, ax1) = plt.subplots(2,1)
ax0.plot(ar_en, true_dos, "-k")
ax0.plot(ar_en, pred_dos, "--r")
ax0.legend(["True", "Pred"])
ax0.set_ylabel("Integrated DOS")

ax1.plot(ar_en, pred_dos - true_dos, "-r")
ax1.legend(["Pred DOS Error"])
ax1.set_xlabel("Energy (eV)")
ax1.set_ylabel("Error")

plt.savefig(args.output_dir + "/pred_true_dos.png")

print("DOS plot created")


# Density Error Slice Plots

font = {'weight' : 'bold',
        'size'   : 50}

matplotlib.rc('font', **font)

z_slices = np.round(np.linspace(0, true_snp.shape[2] - 1, args.num_slices)).astype(int)

fig, ax = plt.subplots(args.num_slices,1)

fig.set_figheight(20 * args.num_slices)
fig.set_figwidth(20)

for i in range(args.num_slices):
    
    gs = ldos_calc.get_grid_spacing()

    xgrid = np.linspace(0, true_snp.shape[0], true_snp.shape[0]) * gs
    ygrid = np.linspace(0, true_snp.shape[1], true_snp.shape[1]) * gs

    density_slice = pred_density[:,:,z_slices[i]] - true_density[:,:,z_slices[i]] 

    ax[i].contourf(xgrid, ygrid, density_slice)
    ax[i].set_title("Density Z-Slice at %3.1f" % (z_slices[i] * gs))

#plt.tight_layout()

plt.savefig(args.output_dir + "/pred_true_density.png")

print("Density plots created")

print("\n\nSuccess!\n\n")




