import argparse

import numpy as np
import pandas as pd
import scipy.stats as ss
from seglearn.transform import Segment, FeatureRep, FeatureRepMix
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
parser.add_argument('--save-dir', type=str, default='benchmark_jsons',
                    help='whether  the directory in which the benchmark will be saved')

args = parser.parse_args()
print(args)


def slope(x):
    return np.apply_along_axis(
        lambda x_: np.polyfit(np.arange(0, len(x_)), x_, 1)[0], arr=x, axis=1
    )


# -------------- get the data
df_emg = get_data()

# ----------------------------------------------------------------------------
with VizTracer(
        log_gc=False,
        output_file=f"{args.save_dir}/seglearn_{args.index}.json",
        max_stack_depth=0,
        plugins=["vizplugins.cpu_usage", "vizplugins.memory_usage"],
):
    union = FeatureRepMix(
        [
            (sig + '_' + k, FeatureRep(features={"": v}), i)
            for k, v in {
            "min": lambda x: np.min(x, axis=1),
            "max": lambda x: np.max(x, axis=1),
            "mean": lambda x: np.mean(x, axis=1),
            "std": lambda x: np.std(x, axis=1),
            "var": lambda x: np.var(x, axis=1),
            "skew": lambda x: ss.skew(x, axis=1),
            "kurt": lambda x: ss.kurtosis(x, axis=1),
            "rms": lambda x: np.sqrt(np.mean(np.square(x), axis=1)),
            # **{f"quantile_{q}": lambda x: np.quantile(x, q=q) for q in quantiles},
            "slope": slope,  # ["emg", "eog", "lso", "rio", "m1-a1"]),
            "area": lambda x: np.sum(x, axis=1),
        }.items()
            for i, sig in enumerate(["emg", "eog", "lso", "rio", "m1-a1"])
        ]
    )
    fs = 1000  # the sample frequency
    segment = Segment(width=int(30 * fs), step=int(10 * fs))
    X, y, _ = segment.fit_transform(X=[df_emg.values], y=[[True] * len(df_emg)])
    X = union.fit_transform(X, y)
    df_feat = pd.DataFrame(data=X, columns=union.f_labels)
    del df_feat
    del segment
    del union
