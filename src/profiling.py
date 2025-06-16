import os
import json

import pandas as pd

def compile_profiled_stats():
    csvs = [file for file in os.listdir(".profiling/") if "prof_stats" in file]
    csv_paths = [f".profiling/{csv}" for csv in csvs]

    df_names = []
    for ix, _ in enumerate(csvs):
        df_names.append("df"+str(ix+1))

    df_list = []
    for df_name, csv_path in zip(df_names,csv_paths):
        df_name = pd.read_csv(csv_path)
        df_list.append(df_name)

    compiled_df = pd.DataFrame(columns=["tot_cumtime", "avg_tottime_call", "avg_percall", ])

    for ix, df in enumerate(df_list):
        stats = []
        stats.append(df["cumtime"].max())
        stats.append(df["tottime"].mean())
        stats.append(df["percall"].mean())
        compiled_df.loc[ix] = stats

    compiled_df.to_csv(".profiling/compiled_df.csv", index=False, header=True)

    compiled_stats = {"avg_time_per_question": compiled_df["tot_cumtime"].mean(),
                    "avg_avg_tottime_call": compiled_df["avg_tottime_call"].mean(),
                    "avg_avg_percall": compiled_df["avg_percall"].mean()}

    with open(".profiling/avg_compiled_stats.json", "w") as f:
        json.dump(compiled_stats, f)