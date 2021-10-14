import numpy as np
import time
import argparse
import os
import .DFT_calculators


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate atom-centered DOS values')
    parser.add_argument('input-dir', \
                help='path to input data directory')
    parser.add_argument('output-dir', \
                help='path to input data directory')
    parser.add_argument('--temp', type=int, default=933,
                        help='temperature of snapshot to train on')
    parser.add_argument('--gcc', type=float, default=2.699,
                        help='density of snapshot to train on')
    parser.add_argument('--snapshot', type=int, default=0, metavar='N',
                        help='aluminum snapshot id')
    args = parser.parse_args()


    snap0_dft_file = '{}/fp/{}K/{}gcc/QE_Al.scf.pw.snapshot{}.out'.format(args.input_dir, args.temp, args.gcc, args.snapshot)
    snap0_ldos_file = '/{}/ldos/{}K/{}gcc/Al_ldos_200x200x200grid_250elvls_snapshot{}.npy'.format(args.input_dir, args.temp, args.gcc, args.snapshot)

    e_grid = np.arange(-10.0,15.0,0.1)
    snap0_DFT = DFT_calculators.DFT_results(snap0_dft_file)

    gauss_width = 1.3

    print("Constructing LDOS")
    tic = time.perf_counter()
    snap0_LDOS = DFT_calculators.LDOS(snap0_DFT,e_grid,snap0_ldos_file,temperature = 933.0)
    snap0_LDOS.do_calcs()
    toc = time.perf_counter()
    print("LDOS time: {}".format(toc-tic))

    print("Constructing ADOS from LDOS")
    tic = time.perf_counter()
    snap0_ADOS = DFT_calculators.ADOS(snap0_LDOS,sigma = gauss_width)
    toc = time.perf_counter()
    print("ADOS time: {}".format(toc-tic))

    # Save ADOS
    out_dir ="{}/{}K/{}gcc".format(args.ouput_dir, args.temp, args.gcc)
    fname = "Al_ados_250elvls_sigma{}_snapshot{}.npy".format(gauss_width, args.snapshot)
    save_path = os.path.join(out_dir, fname)
    np.save(save_path, snap0_ADOS.ados)
