import torch
import torch.nn.functional as F
import argparse
import os
import logging
import sys
from pprint import pprint
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from sklearn.metrics import mean_squared_error
from scipy import interpolate, integrate
from collections import defaultdict
from .dataset import AtomicConfigurations
from .fp_spherical import ConfigurationFingerprint
from .train import model_setup, worker_init_fn
import ados.DFT_calculators as DFT_calculators
from functools import partial
from .common import E_LVLS
from mesh.mesh import icosphere


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.inference_mode()
def test_step(model, data, target):
    model.eval()
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


def evaluate_all(args, model, data_mode="test", config_data=None, write_path=None, **kwargs):

    fp_transform = ConfigurationFingerprint(args.rcut, args.R, args.l, args.m, args.scaling)
    num_workers = args.workers

    data_path = os.path.join(args.data_dir, args.dataset)
    test_data = AtomicConfigurations(data_path, data_mode, args.temp, args.rcut, data_transform=fp_transform, rotate=args.rotate_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    log_dir = os.path.join(args.data_dir, "ados_out/logs")
    if args.out:
        log_dir = os.path.join(log_dir, args.out)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    model_fields = ["B", "lr", "L", "R", "rcut", "w", "wd", "dp", "scaling", "custom"]
    metric_fields = ["dos", "fermi", "liquid_BE_avg", "liquid_BE_max", "solid_BE_avg", "solid_BE_max"]
    fields = model_fields + metric_fields

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
    for batch_idx, (input_data, target_data, config_ids) in enumerate(test_loader):
        ados_pred, ados_ref, loss = test_step(model, input_data, target_data)
        for sample_idx in range(len(input_data)):
            config_id = config_ids[sample_idx].item()
            ados_pred_inst = ados_pred[sample_idx]
            ados_ref_inst = ados_ref[sample_idx]
            pred_ados_by_config[config_id].append(ados_pred_inst)
            ref_ados_by_config[config_id].append(ados_ref_inst)

    total_rmse_dos = 0
    total_error_fermi = 0
    total_error_eband = 0
    BE_err = defaultdict(list)
    for i, config_id in enumerate(test_data.config_ids):
        print("config id: {}".format(config_id))
        if config_id < 10:
            config_type = "liquid"
        else:
            config_type = "solid"
        pred_ados = np.stack(pred_ados_by_config[config_id])
        ref_ados = np.stack(ref_ados_by_config[config_id])
        assert len(pred_ados) == len(ref_ados)

        # DOS
        pred_dos_calc = DFT_calculators.DOS.from_ados_data(None, dos_e_grid, pred_ados)
        ref_dos_calc = DFT_calculators.DOS.from_ados_data(None, dos_e_grid, ref_ados)
        n_atoms = test_data.dft_results[i].num_atoms
        rmse_dos = np.sqrt(mean_squared_error(pred_dos_calc.dos, ref_dos_calc.dos))/n_atoms
        total_rmse_dos += rmse_dos

        # fermi level and band energy
        n_elec = DFT_calculators.dft_2_enum(test_data.dft_results[i], e_fermi='sc', temperature=args.temp)
        fermi_pred = DFT_calculators.dos_2_efermi(pred_dos_calc, args.temp, n_electrons=n_elec)
        fermi_ref = DFT_calculators.dos_2_efermi(ref_dos_calc, args.temp, n_electrons=n_elec)
        total_error_fermi += abs(fermi_ref-fermi_pred)

        pred_eband = DFT_calculators.dos_2_eband(pred_dos_calc, e_fermi=fermi_pred, temperature=args.temp)
        ref_eband = DFT_calculators.dos_2_eband(ref_dos_calc, e_fermi=fermi_ref, temperature=args.temp)
        error_eband = abs(ref_eband - pred_eband)/n_atoms
        total_error_eband += error_eband
        BE_err[config_type].append(error_eband)

        if args.plot:
            save_dos_path = os.path.join(log_dir, "dos-aluminum-{}K_{}_id{}.png".format(args.temp, config_type, config_id))
            dos_plot(ref_dos_calc.dos, pred_dos_calc.dos, dos_e_grid, save_dos_path, fermi_ref)

    n_configs = len(test_data.config_ids)
    avg_rmse_dos = total_rmse_dos / n_configs
    avg_error_fermi = total_error_fermi / n_configs
    avg_error_eband = total_error_eband / n_configs

    all_results = dict()
    all_results["dos"] = avg_rmse_dos
    all_results["fermi"] = avg_error_fermi
    all_results["liquid_BE_avg"] = np.average(BE_err["liquid"])
    all_results["liquid_BE_max"] = np.max(BE_err["liquid"])
    all_results["solid_BE_avg"] = np.average(BE_err["solid"])
    all_results["solid_BE_max"] = np.max(BE_err["solid"])
    error_msg = ''
    all_results["custom"] = error_msg
    pprint(all_results)

    if write_path:
        metric_vals = [str(all_results[k]) for k in metric_fields]
        model_vals = [args.batch_size, args.lr, args.l, args.R, args.rcut, args.weight_decay, args.dropout, args.scaling, error_msg]
        entries = model_vals + metric_vals
        write_results_all(write_path, fields, entries)

    return all_results


def write_results_all(outpath, fields, values):
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



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to saved model")
    parser.add_argument("-dataset", default="aluminum", help="Dataset name")
    parser.add_argument("-d", "--data_dir", default=".", help="Data directory")
    parser.add_argument("-o", "--out", default='', help="Name of output directory for writing results")
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
    parser.add_argument("--temp", type=int, default=933, choices=[298, 933], help="Temperature of snapshots")
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

    write_path = None
    if args.test_out:
        log_dir = os.path.join(args.data_dir, "ados_out/logs")
        if args.out:
            log_dir = os.path.join(log_dir, args.out)
        write_path = os.path.join(log_dir, args.test_out)

    evaluate_all(args, model, write_path=write_path)
