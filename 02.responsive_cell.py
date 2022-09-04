#%% imports and definitions
import os

import numpy as np
import pandas as pd
import seaborn as sns

from routine.utilities import df_roll, df_set_metadata, iter_ds, norm, q_score

IN_SS = "./metadata/sessions.csv"
PARAM_AGG_SUB = (-60, 60)
PARAM_NSHUF = 10
PARAM_SIG_THRES = 0.05
OUT_PATH = "./intermediate/responsive_cell"
FIG_PATH = "./figs/responsive_cell"

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


def agg_cue(df):
    trace = (
        df[df["cue_fm"].notnull()]
        .groupby(["unit_id", "cue_lab", "cue_fm"])
        .mean()
        .reset_index()
    )
    trace = trace.groupby(["unit_id", "cue_lab"]).apply(agg_baseline)
    return trace


def agg_baseline(df):
    df["C"] = norm(df["C"]) * 99 + 1
    if df["C"].notnull().all():
        tlab = np.sign(df["cue_fm"])
        baseline = df.loc[tlab < 0, "C"].mean()
        assert baseline > 0
        df["C"] = (df["C"] - baseline) / baseline
    return df


def agg_dff(df):
    tlab = np.sign(df["cue_fm"])
    return df.loc[tlab > 0, "C"].sum()


def classify_cells(df, sig):
    dff = df.groupby("ishuf").apply(agg_dff)
    q = q_score(dff.loc[0:], dff.loc[-1])
    if q > 1 - sig:
        lab = "activated"
    elif q < sig:
        lab = "inhibited"
    else:
        lab = "non-modulated"
    return pd.Series({"cell_type": lab, "qscore": q, "dff": dff.loc[-1]})


#%% load data and compute shuffled mean
Cmean_full = pd.DataFrame()
for (anm, ss), ps_ds in iter_ds(subset_reg=True):
    C_df = ps_ds["C"].to_dataframe().reset_index()
    Cmean = agg_cue(C_df)
    Cmean = df_set_metadata(Cmean, {"ishuf": -1, "animal": anm, "session": ss})
    Cmean_full = pd.concat([Cmean_full, Cmean], ignore_index=True, copy=False)
    for ishuf in range(PARAM_NSHUF):
        C_shuf = C_df.groupby("unit_id").apply(df_roll)
        Cmean = agg_cue(C_shuf)
        Cmean = df_set_metadata(Cmean, {"ishuf": ishuf, "animal": anm, "session": ss})
        Cmean_full = pd.concat([Cmean_full, Cmean], ignore_index=True, copy=False)
Cmean_full = Cmean_full.astype({"cue_fm": int, "frame": int}).drop(
    columns=["unit_labels"]
)
Cmean_full.to_feather(os.path.join(OUT_PATH, "Cmean.feat"))

#%% load shuffled mean and classify cells
Cmean = pd.read_feather(os.path.join(OUT_PATH, "Cmean.feat"))
Cmean = Cmean[Cmean["cue_fm"].between(*PARAM_AGG_SUB)].copy()
cell_lab = (
    Cmean.groupby(["animal", "session", "unit_id", "cue_lab"])
    .apply(classify_cells, sig=PARAM_SIG_THRES)
    .reset_index()
)
cell_lab.to_csv(os.path.join(OUT_PATH, "cell_lab.csv"), index=False)

#%% plot activity
Cmean = pd.read_feather(os.path.join(OUT_PATH, "Cmean.feat"))
cell_lab = pd.read_csv(os.path.join(OUT_PATH, "cell_lab.csv"))
sessions = pd.read_csv(IN_SS).drop(columns=["date", "data"])
sessions["day"] = sessions["session"].apply(lambda s: s.split("_")[1])
cell_df = cell_lab.merge(sessions, how="left", on=["animal", "session"])
Cmean["C"] = Cmean["C"] * 100
Cmean["time"] = Cmean["cue_fm"] / 20
plot_df = (
    Cmean[Cmean["ishuf"] == -1]
    .drop(columns=["frame", "ishuf"])
    .merge(cell_df, on=["animal", "session", "unit_id", "cue_lab"])
)
plot_df["trace_type"] = "Day " + plot_df["day"] + " " + plot_df["cue_lab"]
g = sns.relplot(
    data=plot_df,
    x="time",
    y="C",
    row="cell_type",
    col="group",
    kind="line",
    hue="trace_type",
    style="trace_type",
    facet_kws={"sharey": "row"},
    ci=68,
    height=3.5,
    aspect=1.2,
    palette={
        "Day 1 GO trials": "green",
        "Day 1 NOGO trials": "red",
        "Day 19 GO trials": "green",
        "Day 19 NOGO trials": "red",
    },
    dashes={
        "Day 1 GO trials": "",
        "Day 1 NOGO trials": "",
        "Day 19 GO trials": (5, 5),
        "Day 19 NOGO trials": (5, 5),
    },
    hue_order=[
        "Day 1 GO trials",
        "Day 1 NOGO trials",
        "Day 19 GO trials",
        "Day 19 NOGO trials",
    ],
)
for ax in g.axes.flat:
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("DeltaF/F (%baseline)")
    ax.set_xticks([-3, 0, 3, 10])
    ax.axvline(0, color="gray")
    ax.axhline(0, color="gray")
    ax.axvspan(0, 3, color="gray", alpha=0.4)
g.savefig(os.path.join(FIG_PATH, "cells.svg"))
g.savefig(os.path.join(FIG_PATH, "cells.png"), dpi=300)
