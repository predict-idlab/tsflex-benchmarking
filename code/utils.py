import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd


def process_viztrace_json(json_path: Union[str, Path]) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    with open(json_path, 'r') as f:
        df_viz = pd.json_normalize(json.load(f).get("traceEvents", {}))

    df_cpu = df_viz[df_viz.name == 'cpu_usage']
    df_cpu = df_cpu.set_index('ts').sort_index()

    df_mem = df_viz[df_viz.name == 'memory_usage']
    df_mem = df_mem.set_index('ts').sort_index()

    t_start = min(df_cpu.index[0], df_mem.index[0])

    df_cpu.index = pd.to_timedelta((df_cpu.index - t_start), unit='us')
    df_mem.index = pd.to_timedelta((df_mem.index - t_start), unit='us')

    df_mem['args.rss'] -= df_mem['args.rss'].min()
    df_mem['args.vms'] -= df_mem['args.vms'].min()

    return df_cpu.dropna(axis=1, how='all'), df_mem.dropna(axis=1, how='all')


def get_data():
    df_acc = pd.read_parquet(
        "https://github.com/predict-idlab/tsflex/raw/main/examples/data/empatica/acc.parquet",
        engine="fastparquet"
    ).set_index("timestamp")

    fs = 1000  # the sample frequency
    duration_s = 1 * 60 * 60  # 1 hour of data
    size = int(duration_s * fs)

    df_emg = pd.DataFrame(
        index=pd.date_range(
            start=datetime.now(), periods=size, freq=pd.Timedelta(seconds=1 / fs)
        ),
        data=np.array(
            [
                np.repeat(df_acc.values[:, idx % 3] / 64, np.ceil(size / len(df_acc)))[
                :size
                ]
                for idx in range(5)
            ]
        ).astype(np.float32).transpose(),
        columns=["emg", "eog", "lso", "rio", "m1-a1"],
    )
    print("memory usage: ", round(sum(df_emg.memory_usage(deep=True) / (2 ** 20)), 2),
          "MB")
    return df_emg
