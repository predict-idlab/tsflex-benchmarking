import argparse

import numpy as np
import scipy.stats as ss
from viztracer import VizTracer

from utils import get_data

# load our library
# import sys
# sys.path.append('/users/jonvdrdo/jonas/projects/time_series')
from tsflex.features import FeatureCollection, FuncWrapper
from tsflex.features import MultipleFeatureDescriptors

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

# ----------------------------------------------------------------------------
# quantiles = [0.25, 0.5, 0.75]


def type_wrapper(x: np.ndarray, type_wrapped_func, **kwargs):
    return type_wrapped_func(x, **kwargs).astype(x.dtype)


# -- 2. in-line functions
#    You can define your functions locally; these will serialize flawlessly
def slope(x):
    return ss.linregress(np.arange(0, len(x)), x)[0]


f_slope = FuncWrapper(type_wrapper, output_names="slope", type_wrapped_func=slope)


# -- 3. Lambda's
#    Or even use lambda's and other modules' functions
def rms(x): return np.sqrt(np.mean(x ** 2))

def std_var(x):
    var = np.var(x)
    return np.sqrt(var), var

def sum_mean(x):
    s = np.sum(x)
    return s, s / len(x)


f_rms = FuncWrapper(rms, output_names="rms")
# More computationally efficient as you can reuse already calculated values
f_std_var = FuncWrapper(std_var, output_names=["std", "var"])
f_sum_mean = FuncWrapper(sum_mean, output_names=["area", "mean"])

# (For convenience) we store the constructed `NumpyFuncWrappers` in a list
segment_funcs = [
    np.min,
    np.max,
    f_std_var,
    f_sum_mean,
    ss.skew,
    ss.kurtosis,
    f_slope,
    f_rms,
]

# -------------- get the data
df_emg = get_data()
windows_s = ['30s', '60s', '90s']

# ----------------------------------------------------------------------------
if args.multiprocessing:
    with VizTracer(
            log_gc=False,
            log_async=True,
            output_file=f"{args.save_dir}/tsflex_mp{'' if args.numb_window_strides == 1 else '_numb_ws=' + str(args.numb_window_strides)}_{args.index}.json",
            max_stack_depth=0,
            plugins=["vizplugins.cpu_usage", "vizplugins.memory_usage"],
    ):
        fc = FeatureCollection(
            feature_descriptors=[
                MultipleFeatureDescriptors(
                    functions=segment_funcs,
                    series_names=["emg", "eog", "lso", "rio", "m1-a1"],
                    windows=windows_s[:args.numb_window_strides],
                    strides=["10s"],
                )
            ]
        )
        out = fc.calculate(data=df_emg, n_jobs=None, return_df=True)
        del out
        del fc
else:
    with VizTracer(
            log_gc=False,
            log_async=True,
            output_file=f"{args.save_dir}/tsflex_sequential{'' if args.numb_window_strides == 1 else '_numb_ws=' + str(args.numb_window_strides)}_{args.index}.json",
            max_stack_depth=0,
            plugins=["vizplugins.cpu_usage", "vizplugins.memory_usage"],
    ):
        fc = FeatureCollection(
            feature_descriptors=[
                MultipleFeatureDescriptors(
                    functions=segment_funcs,
                    series_names=["emg", "eog", "lso", "rio", "m1-a1"],
                    windows=windows_s[:args.numb_window_strides],
                    strides=["10s"],
                )
            ]
        )
        out = fc.calculate(data=df_emg, n_jobs=0, return_df=True)
        del out
        del fc
