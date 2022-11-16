from argparse import ArgumentParser
import collections
import glob
import itertools
import json
import pandas as pd
import platform
import pathlib
import random
import shutil
import subprocess
import sys
from mpi4py import MPI


# Modes:
# train a single model
# hyperparameter opt
# Train a bunch of models with the same hyper parameters (deep ensembles)
# Inference:
#  * single model
#  * dropout (single model, multiple passes)
#  * deep sensembles (multiple models, single pass)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_ranks = comm.Get_size()

def get_losses(filename, mode):
    losses = []
    searchstr = f"mode={mode}"
    with open(filename, "r") as f:
        for line in f:
            if searchstr in line:
                loss = float(line.split()[3].split("=")[1])
                losses.append(loss)
    return losses


def record_training_losses(training_filename, result_filename):
    validation_losses = get_losses(training_filename, "validation")
    training_losses = get_losses(training_filename, "training")
    df = pd.DataFrame({"training": training_losses, "validation": validation_losses})
    df.to_csv(result_filename)


def make_relative(filename):
    p = pathlib.Path(filename)
    if not p.is_absolute():
        p = ".." / p
    return p.as_posix()

def  ensemble_infer(hidden_output_layers: int, param_factor: int,
                   ensemble_size: int, test_snapshots: str,
                   dataset: str, data_dir: str, out: str, model_dir: str):
    test_file = make_relative(test_snapshots)
    model_dir_rel = make_relative(model_dir)
    for i in range(rank, ensemble_size, num_ranks):
        label = f"{i:03d}"
        output_dir = pathlib.Path(label)
        output_dir.mkdir(parents=True, exist_ok=True)
        model = get_model_name(odel_dir_rel + "/" + label)
        gpu_id = rank % 4
        cmd = get_test_cmd(gpu_id, 1337, hidden_output_layers, param_factor,
                test_file, dataset, data_dir, out, model, 0.0)
        with open(output_dir / "node.txt", "w") as f:
            f.write(f"Rank {rank} is running model {i} on {platform.node()} using gpu {gpu_id}")
        with open(output_dir / "stdout.txt", "w") as outf, open(output_dir / "stderr.txt", "w") as errf:
            subprocess.run(cmd, stdout=outf, stderr=errf, cwd=output_dir)

 
def dropout_infer(hidden_output_layers: int, param_factor: int, dropout: float, 
                   ensemble_size: int, resume: bool, test_snapshots: str,
                   dataset: str, data_dir: str, out: str, model: str):
    test_file = make_relative(test_snapshots)
    model_rel = make_relative(model)
    labels = [f"{i:03d}" for i in range(ensemble_size)]
    if resume:
        labels = [label for label in labels if not (pathlib.Path(label) / "out" /"test.csv").exists()]
    for i in range(rank, len(labels), num_ranks):
       label = labels[i]
       output_dir = pathlib.Path(label)
       output_dir.mkdir(parents=True, exist_ok=True)
       gpu_id = rank % 4
       seed = get_seed(i) 
       cmd = get_test_cmd(gpu_id, seed, hidden_output_layers, param_factor,
               test_file, dataset, data_dir, out, model_rel, dropout) 
       with open(output_dir / "node.txt", "w") as f:
           f.write(f"Rank {rank} is running model {i} on {platform.node()} using gpu {gpu_id}")
       with open(output_dir / "stdout.txt", "w") as outf, open(output_dir / "stderr.txt", "w") as errf:
           subprocess.run(cmd, stdout=outf, stderr=errf, cwd=output_dir)
    

def get_model_name(dir_num):
    model_file = glob.glob(dir_num + "/out/logs/*.pkl")
    if len(model_file) == 1:
        return model_file[0]
    else:
        return ""


def ensemble_train(hidden_output_layers: int, param_factor: int, lr: float, epochs: int, start: int,
                   ensemble_size: int, resume: bool, training_snapshots: str, validation_snapshots: str,
                   dataset: str, data_dir: str, out: str):

    val_file = make_relative(validation_snapshots)
    train_file = make_relative(training_snapshots)
    
    labels = [f"{i:03d}" for i in range(ensemble_size)]
    if resume:
        labels = [label for label in labels if not (pathlib.Path(label) / "training_validation_losses.text").exists()]
        

    for i in range(rank, len(labels), num_ranks):
        if i < start:
            continue
        label = labels[i]
        gpu_id = rank % 4
        seed = get_seed(i)
        cmd = get_training_cmd(gpu_id, seed, hidden_output_layers, param_factor,
            lr, epochs, train_file,
            val_file, dataset, data_dir, out, 0.0)

        model_dir = pathlib.Path(label)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "node.txt", "w") as f:
            f.write(f"Rank {rank} is running model {i} on {platform.node()} using gpu {gpu_id}")
        with open(model_dir / "stdout.txt", "w") as outf, open(model_dir / "stderr.txt", "w") as errf:
            subprocess.run(cmd, stdout=outf, stderr=errf, cwd=model_dir)
        record_training_losses(model_dir / "stdout.txt", model_dir / "training_validation_losses.txt")


def hyperparameter_opt(hidden_output_layers, param_factor, lr, epochs,
                     dropout, ensemble_start, snapshots,
                     validation_fraction, dataset, data_dir, out, replicates, resume):
    snapshots_file = make_relative(snapshots)
    configs = generate_configs(hidden_output_layers, param_factor, lr, epochs, dropout, replicates)
    
    if resume:
        configs = [c for c in configs if not already_complete(c)]

    num_configs = len(configs)
    for i in range(rank, num_configs, num_ranks):
        if i < ensemble_start:
            continue
        config = configs[i]
        label = generate_label(config)
        seed = get_seed(i)
        gpu_id = rank % 4
        cmd = get_training_cmd(gpu_id, seed, config.hidden_output_layers, config.param_factor,
                            config.lr, config.epochs, snapshots_file, validation_fraction, dataset, data_dir,
                            out, config.dropout)
        model_dir = pathlib.Path(label)
        shutil.rmtree(str(model_dir), ignore_errors=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "node.txt", "w") as f:
            f.write(f"Rank {rank} is running model {i} on {platform.node()} using gpu {gpu_id}")
        with open(model_dir / "stdout.txt", "w") as outf, open(model_dir / "stderr.txt", "w") as errf:
            p = subprocess.run(cmd, stdout=outf, stderr=errf, cwd=model_dir)
        write_config(config, model_dir)
        record_training_losses(model_dir / "stdout.txt", model_dir / "training_validation_losses.txt")


def already_complete(config):
    p = pathlib.Path(generate_label(config)) / "training_validation_losses.txt"
    return p.exists()


def write_config(config, model_dir):
    cd = {
        "hidden_output_layers": config.hidden_output_layers,
        "param_factor": config.param_factor,
        "lr": config.lr,
        "epochs": config.epochs,
        "dropout": config.dropout,
        "replicate": config.replicate
    }
    with open(model_dir / "params.json", "w") as pf:
        json.dump(cd, pf)
            

def generate_configs(hidden_output_layers, param_factor, lr, epochs, dropout, replicates):
    TrainingConfig = collections.namedtuple('TrainingConfig',
        ["hidden_output_layers", "param_factor", "lr", "epochs", "dropout", "replicate"])
    hol_list = make_list(hidden_output_layers)
    pf_list = make_list(param_factor)
    lr_list = make_list(lr)
    e_list = make_list(epochs)
    d_list = make_list(dropout)
    r_list = [i for i in range(replicates)]
    configs = [TrainingConfig(hol, pf, lr, e, d, r) for hol, pf, lr, e, d, r in \
        itertools.product(hol_list, pf_list, lr_list, e_list, d_list, r_list)]
    return configs


def model_exists(label):
    return bool(get_model_name(label))

 
def single_train(hidden_output_layers: int, param_factor: int, lr: float, epochs: int, dropout: float,
        snapshots: str, validation_fraction: float, dataset: str, data_dir: str, out: str):
    if rank == 0:
        gpu_id = 0
        cmd = get_training_cmd(gpu_id, 1337, hidden_output_layers, param_factor,
            lr, epochs, snapshots,
            validation_fraction, dataset, data_dir, out, dropout)

        with open("stdout.txt", "w") as outf, open("stderr.txt", "w") as errf:
            subprocess.run(cmd, stdout=outf, stderr=errf)
        record_training_losses("stdout.txt", "training_validation_losses.txt")

def  single_infer(hidden_output_layers: int, param_factor: int, dropout: float,
                   test_snapshots: str, dataset: str, data_dir: str, out: str, model: str):
    if rank == 0:
        gpu_id = 0
        cmd = get_test_cmd(gpu_id, 1337, hidden_output_layers, param_factor,
                test_snapshots, dataset, data_dir, out, model, dropout)
        with open("stdout.txt", "w") as outf, open("stderr.txt", "w") as errf:
            subprocess.run(cmd, stdout=outf, stderr=errf)




def generate_label(config):
    label = f"ohl-{config.hidden_output_layers}_pf-{config.param_factor}_lr-{config.lr}_" + \
            f"e-{config.epochs}_d-{config.dropout}_r-{config.replicate}"
    return label


def make_list(d):
    if isinstance(d, list):
        return d
    else:
        return [d]


def get_seed(idx):
    seed_seed = 1337
    random.seed(seed_seed)
    for i in range(idx+1):
        seed = random.randint(100000, 999999)
    return seed


def get_test_cmd(gpu_id: int, seed: int, hidden_output_layers: int, param_factor: int,
            test_snapshots: str, dataset: str, data_dir: str, out: str, model: str,
            dropout: float):
    cmd = ["python3",
           "-m",
           "ados.test",
           "-dataset",
           dataset,
           "--data_dir",
           data_dir,
           "--test_snapshots",
           test_snapshots,
           "--out",
           out,
           "--device",
           str(gpu_id),
           "--seed",
           str(seed),
           "--output-hidden-layers",
           str(hidden_output_layers),
           "--param-factor",
           str(param_factor)]

    if dropout > 0.0:
        cmd.extend(["--dropout", str(dropout)])
    cmd.append(model)
    return cmd


def get_training_cmd(gpu_id: int, seed: int, hidden_output_layers: int, param_factor: int,
            lr: float, epochs: int, snapshots: str,
            validation_fraction: float, dataset: str, data_dir: str, out: str,
            dropout: float):
    cmd = ["python3",
           "-m",
           "ados.train",
           "-dataset",
           dataset,
           "--data_dir",
           data_dir,
           "--snapshots",
           snapshots,
           "--validation_fraction",
           str(validation_fraction),
           "--out",
           out,
           "--epochs",
           str(epochs),
           "--device",
           str(gpu_id),
           "--seed",
           str(seed),
           "--output-hidden-layers",
           str(hidden_output_layers),
           "--param-factor",
           str(param_factor),
           "--lr",
           str(lr)]
    if dropout is not None and dropout > 0.0:
        cmd += ["--dropout", str(dropout)]
    return cmd


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser_ensemble_train = subparsers.add_parser("ensemble-train")
    parser_ensemble_train.add_argument("--hidden-output-layers", default=2, type=int)
    parser_ensemble_train.add_argument("--param-factor", default=4, type=int)
    parser_ensemble_train.add_argument("--lr", default=1e-2, type=float)
    parser_ensemble_train.add_argument("--training-snapshots", default="training_data.txt")
    parser_ensemble_train.add_argument("--validation-snapshots", default="validation_data.txt")
    parser_ensemble_train.add_argument("--out", default="out")
    parser_ensemble_train.add_argument("--epochs", default=200, type=int)
    parser_ensemble_train.add_argument("--dataset", default="aluminum")
    parser_ensemble_train.add_argument("--data-dir", default="/g/g20/jasteph/ados_dataset")
    parser_ensemble_train.add_argument("--start", default=0, type=int)
    parser_ensemble_train.add_argument("--resume", action="store_true", default=False)
    parser_ensemble_train.add_argument("num", type=int)

    parser_dropout_infer = subparsers.add_parser("dropout-infer")
    parser_dropout_infer.add_argument("--hidden-output-layers", default=2, type=int)
    parser_dropout_infer.add_argument("--param-factor", default=4, type=int)
    parser_dropout_infer.add_argument("--dropout", default=0.1, type=float)
    parser_dropout_infer.add_argument("--test-snapshots", default="test_data.txt")
    parser_dropout_infer.add_argument("--out", default="out")
    parser_dropout_infer.add_argument("--dataset", default="aluminum")
    parser_dropout_infer.add_argument("--data-dir", default="/g/g20/jasteph/ados_dataset")
    parser_dropout_infer.add_argument("--resume", action="store_true", default=False)
    parser_dropout_infer.add_argument("num", type=int)
    parser_dropout_infer.add_argument("model")


    parser_ensemble_infer = subparsers.add_parser("ensemble-infer")
    parser_ensemble_infer.add_argument("--hidden-output-layers", default=2, type=int)
    parser_ensemble_infer.add_argument("--param-factor", default=4, type=int)
    parser_ensemble_infer.add_argument("--test-snapshots", default="test_data.txt")
    parser_ensemble_infer.add_argument("--out", default="out")
    parser_ensemble_infer.add_argument("--dataset", default="aluminum")
    parser_ensemble_infer.add_argument("--data-dir", default="/g/g20/jasteph/ados_dataset")
    parser_ensemble_infer.add_argument("num", type=int)
    parser_ensemble_infer.add_argument("model-dir")


    parser_hyper_opt = subparsers.add_parser("hyperparameter-opt")
    parser_hyper_opt.add_argument("--hidden-output-layers", nargs='+', type=int)
    parser_hyper_opt.add_argument("--param-factor", nargs='+', type=int)
    parser_hyper_opt.add_argument("--lr", nargs='+', type=float)
    parser_hyper_opt.add_argument("--dropout", nargs='*', type=float)
    parser_hyper_opt.add_argument("--snapshots", default="training_data.txt")
    parser_hyper_opt.add_argument("--validation-fraction", default=0.3, type=float)
    parser_hyper_opt.add_argument("--out", default="out")
    parser_hyper_opt.add_argument("--epochs", default=200, type=int)
    parser_hyper_opt.add_argument("--dataset", default="aluminum")
    parser_hyper_opt.add_argument("--start", default=0, type=int)
    parser_hyper_opt.add_argument("--data-dir", default="/g/g20/jasteph/ados_dataset")
    parser_hyper_opt.add_argument("--replicates", default=1, type=int)
    parser_hyper_opt.add_argument("--resume", action="store_true", default=False)

    parser_single_train = subparsers.add_parser("single-train")
    parser_single_train.add_argument("--hidden-output-layers", default=2)
    parser_single_train.add_argument("--param-factor", default=4)
    parser_single_train.add_argument("--lr", default=1e-2)
    parser_single_train.add_argument("--dropout", default=0.0, type=float)  
    parser_single_train.add_argument("--snapshots", default="training_data.txt")
    parser_single_train.add_argument("--validation-fraction", default=0.3)
    parser_single_train.add_argument("--out", default="out")
    parser_single_train.add_argument("--epochs", default=200)
    parser_single_train.add_argument("--dataset", default="aluminum")
    parser_single_train.add_argument("--start", default=0)
    parser_single_train.add_argument("--data-dir", default="/g/g20/jasteph/ados_dataset")

    parser_single_infer = subparsers.add_parser("single-infer")
    parser_single_infer.add_argument("--hidden-output-layers", default=2)
    parser_single_infer.add_argument("--param-factor", default=4)
    parser_single_infer.add_argument("--dropout", default=0.0)
    parser_single_infer.add_argument("--test-snapshots", default="training_data.txt")
    parser_single_infer.add_argument("--out", default="out")
    parser_single_infer.add_argument("--dataset", default="aluminum")
    parser_single_infer.add_argument("--data-dir", default="/g/g20/jasteph/ados_dataset")
    parser_single_infer.add_argument("model")

    return parser


def main():
    parser = create_parser()
    args = vars(parser.parse_args())
    print(args)
    if args["command"] == 'ensemble-train':
        ensemble_train(args["hidden_output_layers"], args["param_factor"], args["lr"], args["epochs"],
                   args["start"],
                   args["num"], args["resume"], args["training_snapshots"], args["validation_snapshots"],
                   args["dataset"], args["data_dir"], args["out"])
    elif args["command"] == 'ensemble-infer':
        ensemble_infer(args["hidden_output_layers"], args["param_factor"], args["num"],
                   args["test_snapshots"], args["dataset"], args["data_dir"], args["out"],
                   args["model-dir"])
    elif args["command"] == 'dropout-infer':
        print(f"resume flag is {args['resume']}") 
        dropout_infer(args["hidden_output_layers"], args["param_factor"], args["dropout"], args["num"], args["resume"],
                   args["test_snapshots"], args["dataset"], args["data_dir"], args["out"],
                   args["model"])
    elif args["command"] == "hyperparameter-opt":
        hyperparameter_opt(args["hidden_output_layers"], args["param_factor"], args["lr"], args["epochs"],
                     args["dropout"], args["start"], args["snapshots"],
                     args["validation_fraction"], args["dataset"], args["data_dir"],
                     args["out"], args["replicates"], args["resume"])
    elif args["command"] == 'single-train':
        single_train(args["hidden_output_layers"], args["param_factor"], args["lr"], args["epochs"], args["dropout"],
                   args["snapshots"], args["validation_fraction"],
                   args["dataset"], args["data_dir"], args["out"])
    elif args["command"] == 'single-infer':
        single_infer(args["hidden_output_layers"], args["param_factor"], args["dropout"],
                   args["test_snapshots"], args["dataset"], args["data_dir"], args["out"],
                   args["model"])

if __name__ == '__main__':
    main()



