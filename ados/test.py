import torch
import torch.nn.functional as F
import argparse
import os
import pathlib
import logging
import sys
import pickle
from pprint import pprint
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from sklearn.metrics import mean_squared_error
from scipy import interpolate, integrate
from collections import defaultdict
from .dataset import AtomicConfigurations, load_snapshots_list
from .fp_spherical import ConfigurationFingerprint
from .train import model_setup, worker_init_fn
from .model import enable_dropout, disable_dropout
import ados.DFT_calculators as DFT_calculators
from functools import partial
from .common import E_LVLS
from mesh.mesh import icosphere


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#@torch.inference_mode()
def test_step(model, data, target):
    model.eval()
    if model.dropout > 0.0:
        enable_dropout(model)
    data = data.cuda()
    pred_ados = model(data)
    ref_ados = target.cuda()
    loss_ados = F.mse_loss(pred_ados, ref_ados)

    ados_npy = pred_ados.detach().cpu().numpy()
    ados_ref_npy = ref_ados.detach().cpu().numpy()

    return ados_npy, ados_ref_npy, loss_ados.item() 


def dos_plot(ref_DOS, pred_DOS, dos_e_grid, save_path, fermi_ref=None):
    e_start = 50
    e_end = 200
    ref_DOS = ref_DOS[e_start:e_end]
    pred_DOS = pred_DOS[e_start:e_end]
    dos_e_grid = dos_e_grid[e_start:e_end]

    y_lim = 200    
    fig, (ax0, ax1) = plt.subplots(2,1)
    ax0.set_ylim([0, y_lim])
    ax0.plot(dos_e_grid, ref_DOS, "-k")
    ax0.plot(dos_e_grid, pred_DOS, "--r")
    if fermi_ref:
        ax0.axvline(fermi_ref, color='g', label='Fermi Energy DFT', ymin=0.4, ymax=0.6)
    ax0.legend(["DFT DOS", "ML Prediction", "DFT Fermi Energy"], loc="upper left")
    ax0.set_ylabel("DOS")

    ax1.set_ylim([-2, 2])
    ax1.plot(dos_e_grid, pred_DOS - ref_DOS, "-r")
    ax1.legend(["Pred DOS Error"], loc="upper left")
    ax1.set_xlabel("Energy (eV)")
    ax1.set_ylabel("DOS Error")

    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.close()

    print("\nDOS plot created") 
    return plt


def evaluate_all(args, model, output_table_path=None, snapshot_results_dir=None, **kwargs):

    fp_transform = ConfigurationFingerprint(args.rcut, args.R, args.l, args.m, args.scaling)
    num_workers = args.workers
    test_snapshots = load_snapshots_list(args.test_snapshots)

    data_path = os.path.join(args.data_dir, args.dataset)
    test_data = AtomicConfigurations(data_path, test_snapshots, args.rcut, data_transform=fp_transform, rotate=args.rotate_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    log_dir = args.out
    
    dos_results = []
    fermi_results = []
    energy_results = []

    true_dos = []
    true_efermi = []
    true_eband = []
    true_enum = []

    dos_e_grid = np.arange(-10.0,15.0,0.1)

    # compute ADOS entries from dataset
    # ordering of ADOS entries for each snapshot not enforced
    ref_ados_by_config = defaultdict(list)
    pred_ados_by_config = defaultdict(list)
    for batch_idx, (input_data, target_data, sample_idxs) in enumerate(test_loader):
        ados_pred, ados_ref, loss = test_step(model, input_data, target_data)
        for i in range(len(input_data)):
            sample_idx = sample_idxs[i].item()
            snapshot = test_data.index_config_map[sample_idx]
            ados_pred_inst = ados_pred[i]
            ados_ref_inst = ados_ref[i]
            pred_ados_by_config[snapshot].append(ados_pred_inst)
            ref_ados_by_config[snapshot].append(ados_ref_inst)

    total_rmse_dos = 0
    total_error_fermi = 0
    total_error_eband = 0
    BE_err = defaultdict(list)
    band_energies = {}
    total_dos = {}

    for i, snapshot in enumerate(test_data.snapshots):
        temp, density, ss_id, phase = snapshot
        temp = int(temp)
        print(f"Temp: {temp}K, Density: {density}gcc, ID: {ss_id}")
        pred_ados = np.stack(pred_ados_by_config[snapshot])
        ref_ados = np.stack(ref_ados_by_config[snapshot])
        assert len(pred_ados) == len(ref_ados)

        # DOS
        pred_dos_calc = DFT_calculators.DOS.from_ados_data(None, dos_e_grid, pred_ados)
        ref_dos_calc = DFT_calculators.DOS.from_ados_data(None, dos_e_grid, ref_ados)
        total_dos[snapshot] = {
            "prediction": pred_dos_calc,
            "reference": ref_dos_calc
        }
        n_atoms = test_data.dft_results[i].num_atoms
        rmse_dos = np.sqrt(mean_squared_error(pred_dos_calc.dos, ref_dos_calc.dos))/n_atoms
        total_rmse_dos += rmse_dos

        # fermi level and band energy
        n_elec = DFT_calculators.dft_2_enum(test_data.dft_results[i], e_fermi='sc', temperature=temp)
        fermi_pred = DFT_calculators.dos_2_efermi(pred_dos_calc, temp, n_electrons=n_elec)
        fermi_ref = DFT_calculators.dos_2_efermi(ref_dos_calc, temp, n_electrons=n_elec)
        total_error_fermi += abs(fermi_ref-fermi_pred)

        pred_eband = DFT_calculators.dos_2_eband(pred_dos_calc, e_fermi=fermi_pred, temperature=temp)
        ref_eband = DFT_calculators.dos_2_eband(ref_dos_calc, e_fermi=fermi_ref, temperature=temp)
        band_energies[snapshot] = {
            "prediction": pred_eband,
            "reference": ref_eband
        }
        error_eband = abs(ref_eband - pred_eband)/n_atoms
        total_error_eband += error_eband
        BE_err[phase].append(error_eband)

        if args.plot:
            save_dos_path = os.path.join(log_dir, f"dos-aluminum-{temp}K_{density}gcc_id{ss_id}.png")
            dos_plot(ref_dos_calc.dos, pred_dos_calc.dos, dos_e_grid, save_dos_path, fermi_ref)

    n_configs = len(test_data.snapshots)
    avg_rmse_dos = total_rmse_dos / n_configs
    avg_error_fermi = total_error_fermi / n_configs
    avg_error_eband = total_error_eband / n_configs

    all_results = dict()
    all_results["dos"] = avg_rmse_dos
    all_results["fermi"] = avg_error_fermi
    for phase, errors in BE_err.items():
        all_results[f"{phase}_BE_avg"] = np.average(errors)
        all_results[f"{phase}_BE_max"] = np.max(errors)
    error_msg = ''
    all_results["custom"] = error_msg
    pprint(all_results)

    if output_table_path:
        metric_vals = [str(v) for k, v in all_results.items()]
        model_vals = [args.batch_size, args.lr, args.l, args.R, args.rcut, args.weight_decay, args.dropout, args.scaling, error_msg]
        metric_fields = ["dos", "fermi"] + list(all_results.keys()) 
        model_fields = ["B", "lr", "L", "R", "rcut", "w", "wd", "dp", "scaling", "custom"]
        entries = model_vals + metric_vals
        fields = model_fields + metric_fields
        write_results_table(output_table_path, fields, entries)

    if snapshot_results_dir:
        write_results_files(snapshot_results_dir, band_energies, total_dos)

    return all_results


def write_results_table(outpath, fields, values):
    """
    Writes header (if file doesn't exist) and results row containing
    both experiment hyperparameters and result metrics.
    """
    try:
        with open(outpath, 'x') as f:
            header_writer = csv.writer(f, delimiter=",")
            header_writer.writerow(fields)
    except FileExistsError:
        print('File already exists. Skipping writing of header')

    with open(outpath, "a+", newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        csvwriter.writerow(values)


def write_results_files(output_dir, band_energies, total_dos):
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    for snapshot in band_energies:
        temp, density, id, _ = snapshot
        filename = f"result-{temp}K-{density}gcc-{id}.pkl"
        with open(os.path.join(output_dir, filename), "wb") as f:
            pickle.dump(band_energies[snapshot], f)
            pickle.dump(total_dos[snapshot], f)


def mkdir_p(dirname):
    p = pathlib.Path(dirname)
    p.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to saved model")
    parser.add_argument("-dataset", default="aluminum", help="Dataset name")
    parser.add_argument("-d", "--data_dir", default=".", help="Data directory")
    parser.add_argument("--test_snapshots", help="File containing list of training snapshots", default="test_data.txt")
    parser.add_argument("-o", "--out", default='ados_out', help="Name of output directory for writing results")
    parser.add_argument("-t", "--test-out", default='test.csv', help="Name of test results file")
    parser.add_argument('-l', '--level', help='Level of mesh refinement', type=int, dest='l', default=3)
    parser.add_argument('-R', type=int, default=16, help='Number of radial levels')
    parser.add_argument('-rcut', help='Neighborhood radius for molecular environment', type=float, default=7)
    parser.add_argument("-p", default="mean", help="Pooling type: [max, sum, mean]")
    parser.add_argument("-m", default="linear", choices=["linear"], help="Data mapping type")
    parser.add_argument("-scaling", default="inverse_sq", choices=["none", "inverse", "inverse_sq"], help="Distance scaling type")
    parser.add_argument("-k_radial", type=int, help="Radial kernel size", default=3)
    parser.add_argument("-g", help="Graph convolution type", choices=["gcn", "graphsage"], default="gcn")
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("-F", "--param-factor", type=float, help="Parameter increase factor", default=4)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, help='random seed', default=24)
    parser.add_argument("--rotate-train", action='store_true')
    parser.add_argument("--rotate-test", action='store_true')
    parser.add_argument('--device', default=0, type=int, help='GPU device ID')
    parser.add_argument("--workers", type=int, default=8, help="Number of processes for data loading")
    parser.add_argument("--debug", action='store_true', help="Debug mode: single batch, single epoch")
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--identifier", help="user-defined string to add to log name")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    icosphere = icosphere(args.l)
    model = model_setup(args, mesh=icosphere)
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True) 
    snapshot_results_dir = out_dir / "snapshots"
    snapshot_results_dir.mkdir(exist_ok=True)
    output_table_path = out_dir / args.test_out

    evaluate_all(args, model, output_table_path=output_table_path, snapshot_results_dir=snapshot_results_dir)
