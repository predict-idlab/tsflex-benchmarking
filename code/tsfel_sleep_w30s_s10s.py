import argparse
import sys

import tsfel
from viztracer import VizTracer

from utils import get_data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# parse the args
parser = argparse.ArgumentParser(description='benchmark tsflex')
parser.add_argument('--index', required=True, type=int,
                    help='an integer for the accumulator')
parser.add_argument('--multiprocessing', required=True, type=str2bool,
                    help='whether to use multiprocessing variant')
parser.add_argument('--numb-window-strides', type=int, default=1,
                    help='the number of window-stride combinations')
parser.add_argument('--save-dir', type=str, default='benchmark_jsons',
                    help='whether  the directory in which the benchmark will be saved')

args = parser.parse_args()
print(args)

# load our library
sys.path.append('../../')

# ----------------------------------------------------------------------------
fs = 1000
feat_dict = (
    {
        "my_custom_set_of_features": {
            "Min": {
                "function": "tsfel.calc_min",
                "parameters": "",
                "n_features": 1,
                "use": "yes",
            },
            "Max": {
                "function": "tsfel.calc_max",
                "parameters": "",
                "n_features": 1,
                "use": "yes",
            },
            "Mean": {
                "function": "tsfel.calc_mean",
                "parameters": "",
                "n_features": 1,
                "use": "yes",
                "tag": "inertial",
            },
            "Standard deviation": {
                "function": "tsfel.calc_std",
                "parameters": "",
                "n_features": 1,
                "use": "yes",
            },
            "Variance": {
                "function": "tsfel.calc_var",
                "parameters": "",
                "n_features": 1,
                "use": "yes",
            },
            "Skewness": {
                "function": "tsfel.skewness",
                "parameters": "",
                "n_features": 1,
                "use": "yes",
            },
            "Kurtosis": {
                "function": "tsfel.kurtosis",
                "parameters": "",
                "n_features": 1,
                "use": "yes",
            },
            "Slope": {
                "function": "tsfel.slope",
                "parameters": "",
                "n_features": 1,
                "use": "yes",
            },
            "Root mean square": {
                "function": "tsfel.rms",
                "parameters": "",
                "n_features": 1,
                "use": "yes",
            },
            "Total energy": {
                "function": "tsfel.total_energy",
                "parameters": {"fs": fs},
                "n_features": 1,
                "use": "yes",
            },
        }
    }
)

# -------------- get the data
df_emg = get_data()
windows_s = [(20, 10), (30, 10), (60, 10), (90, 10)]

# ----------------------------------------------------------------------------
if args.multiprocessing:
    with VizTracer(
            log_gc=False,
            log_async=True,
            output_file=f"{args.save_dir}/tsfel_mp{'' if args.numb_window_strides == 1 else '_numb_ws=' + str(args.numb_window_strides)}_{args.index}.json",
            max_stack_depth=0,
            plugins=["vizplugins.cpu_usage", "vizplugins.memory_usage"],
    ):
        for window_s, stride_s in windows_s[:args.numb_window_strides]:
            out = tsfel.time_series_features_extractor(
                dict_features=feat_dict,
                signal_windows=df_emg,  # ["emg"],#.values,
                fs=fs,
                window_size=window_s * fs,
                overlap=(window_s - 10) / window_s,
                header_names=df_emg.columns,
                n_jobs=16
            )
            del out
else:
    with VizTracer(
            log_gc=False,
            output_file=f"{args.save_dir}/tsfel_sequential{'' if args.numb_window_strides == 1 else '_numb_ws=' + str(args.numb_window_strides)}_{args.index}.json",
            max_stack_depth=0,
            plugins=["vizplugins.cpu_usage", "vizplugins.memory_usage"],
    ):
        for window_s, stride_s in windows_s[:args.numb_window_strides]:
            out = tsfel.time_series_features_extractor(
                dict_features=feat_dict,
                signal_windows=df_emg,  # ["emg"],#.values,
                fs=fs,
                window_size=window_s * fs,
                overlap=(window_s - stride_s) / window_s,
                header_names=df_emg.columns,
                n_jobs=1
            )
            del out
