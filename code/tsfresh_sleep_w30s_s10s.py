import argparse

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series
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
parser.add_argument('--save-dir', type=str, default='benchmark_jsons',
                    help='whether  the directory in which the benchmark will be saved')

args = parser.parse_args()
print(args)

# -------------- get the data
df_emg = get_data()

kind_to_fc_parameters = {
    colname: {
        x[0]: (None if len(x) == 1 else x[1])
        for x in [
            ("minimum", None),
            ("maximum",),
            ("mean",),
            ("standard_deviation",),
            ("variance",),
            ("skewness",),
            ("kurtosis",),
            ("root_mean_square",),
            ("linear_trend", [{"attr": "slope"}]),
            ("abs_energy",),
        ]
    }
    for colname in df_emg.columns
}

fs = 1000
window_s = 30
stride_s = 10

df_emg['id'] = 0

# ----------------------------------------------------------------------------
if args.multiprocessing:
    with VizTracer(
            log_gc=False,
            output_file=f"{args.save_dir}/tsfresh_mp_{args.index}.json",
            max_stack_depth=0,
            plugins=["vizplugins.cpu_usage", "vizplugins.memory_usage"],
    ):
        df_rolled = roll_time_series(
            df_emg.reset_index().rename(columns={'index': 'timestamp'}),
            column_id='id',
            column_sort="timestamp",
            min_timeshift=window_s * fs,
            max_timeshift=window_s * fs,
            rolling_direction=stride_s * fs,
            n_jobs=16
        )
        df_feat = extract_features(
            df_rolled,
            column_id="id",
            column_sort="timestamp",
            kind_to_fc_parameters=kind_to_fc_parameters,
            n_jobs=16
        )
        del df_rolled, df_feat
else:
    with VizTracer(
            log_gc=False,
            output_file=f"{args.save_dir}/tsfresh_sequential_{args.index}.json",
            max_stack_depth=0,
            plugins=["vizplugins.cpu_usage", "vizplugins.memory_usage"],
    ):
        df_rolled = roll_time_series(
            df_emg.reset_index().rename(columns={'index': 'timestamp'}),
            column_id='id',
            column_sort="timestamp",
            min_timeshift=window_s * fs,
            max_timeshift=window_s * fs,
            rolling_direction=stride_s * fs,
            n_jobs=0
        )
        df_feat = extract_features(
            df_rolled,
            column_id="id",
            column_sort="timestamp",
            kind_to_fc_parameters=kind_to_fc_parameters,
            n_jobs=0
        )
        del df_rolled, df_feat
