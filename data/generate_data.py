"""Downloads a dataset and splits data for federated simulation

Split a classification dataset, e.g., `MNIST`, among `n_clients`.

A splitting strategy is available: `iid_split`

If 'iid_split' is selected, the dataset is split in an IID fashion.

Default usage is ''iid_split'

"""
import argparse

from utils import *
from constants import *


def parse_arguments(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset_name",
        help="name of dataset to use, possible are {'synthetic', 'mnist', 'cifar10'}",
        required=True,
        type=str
    )
    parser.add_argument(
        "--n_clients",
        help="number of clients",
        required=True,
        type=int
    )
    parser.add_argument(
        "--iid",
        help="if selected, data are split iid",
        action='store_true'
    )
    parser.add_argument(
        "--non_iid",
        help="if selected, data are split non-iid",
        action='store_true'
    )
    parser.add_argument(
        '--n_classes_per_client',
        help='number of classes given to each clients; ignored if `--non_iid` is not used;'
             'default is 2',
        type=int,
        default=2
    )
    parser.add_argument(
        '--frac',
        help='fraction of the dataset to be used; default: 1.0;',
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help='path of the directory to save data;'
             'the directory will be created if not already created;'
             'if not specified the data is saved to "./{dataset_name}";',
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '--seed',
        help='seed for the random number generator;'
             'if not specified the system clock is used to generate the seed;',
        type=int,
        default=argparse.SUPPRESS,
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args_ = parse_arguments()

    seed = (args_.seed if (("seed" in args_) and (args_.seed >= 0)) else int(time.time()))
    rng = np.random.default_rng(seed)

    if "save_dir" in args_:
        save_dir = args_.save_dir
    else:
        save_dir = os.path.join(".", args_.dataset_name)
        warnings.warn(f"'--save_dir' is not specified, results are saved to {save_dir}!", RuntimeWarning)

    os.makedirs(save_dir, exist_ok=True)

    if args_.dataset_name == "mnist":
        if args_.non_iid:
            split_type = "non_iid"
        elif args_.iid:
            split_type = "iid"
        else:
            warnings.warn("split type is automatically set to 'iid'")
            split_type = "iid"

        dataset = get_dataset(
            dataset_name=args_.dataset_name,
            raw_data_path=os.path.join(save_dir, "raw_data")
        )

        generate_data(
            dataset=dataset,
            split_type=split_type,
            n_train_samples=N_TRAIN_SAMPLES[args_.dataset_name],
            n_clients=args_.n_clients,
            n_classes=N_CLASSES[args_.dataset_name],
            n_classes_per_client=args_.n_classes_per_client,
            frac=args_.frac,
            save_dir=os.path.join(save_dir, "all_data"),
            rng=rng
        )

    else:
        error_message = f"{args_.dataset_name} is not available, possible datasets are:"
        for n in DATASETS:
            error_message += f" {n},"
        error_message = error_message[:-1]

        raise NotImplementedError(error_message)
