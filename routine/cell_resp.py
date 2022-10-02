import numpy as np
import pandas as pd
from scipy.signal import medfilt

from .utilities import norm, q_score


def agg_cue(df, fm_name, lab_name, agg_method, smooth_ksize=None, act_name="C"):
    trace = (
        df[df[fm_name].notnull()]
        .groupby(["unit_id", lab_name, fm_name])
        .mean()
        .reset_index()
    )
    if smooth_ksize is not None:
        trace[act_name] = trace.groupby(["unit_id", lab_name])[act_name].transform(
            medfilt, kernel_size=smooth_ksize
        )
    trace = (
        trace.groupby(["unit_id", lab_name])
        .apply(agg_baseline, fm_name=fm_name, method=agg_method, act_name=act_name)
        .rename(columns={fm_name: "evt_fm", lab_name: "evt_lab"})
    )
    return trace


def agg_baseline(df, fm_name, act_name, method="minmax"):
    tlab = np.sign(df[fm_name])
    if method == "minmax":
        df[act_name] = norm(df[act_name]) * 99 + 1
        if df[act_name].notnull().all():
            baseline = df.loc[tlab < 0, act_name].mean()
            assert baseline > 0
            df[act_name] = (df[act_name] - baseline) / baseline
        else:
            df[act_name] = np.nan
    elif method == "zscore":
        baseline = df.loc[tlab < 0, act_name].mean()
        std = np.std(df[act_name])
        if std > 0:
            df[act_name] = (df[act_name] - baseline) / std
        else:
            df[act_name] = np.nan
    return df


def agg_dff(df, fm_name, act_name):
    tlab = np.sign(df[fm_name])
    return df.loc[tlab > 0, act_name].sum()


def classify_cells(df, sig, fm_name="evt_fm", act_name="C"):
    dff = df.groupby("ishuf").apply(agg_dff, fm_name=fm_name, act_name=act_name)
    q = q_score(dff.loc[0:], dff.loc[-1])
    if q > 1 - sig:
        lab = "activated"
    elif q < sig:
        lab = "inhibited"
    else:
        lab = "non-modulated"
    return pd.Series({"cell_type": lab, "qscore": q, "dff": dff.loc[-1]})


def standarize_df(df, act_name="C"):
    return (
        df[
            [
                "animal",
                "session",
                "unit_id",
                "by_event",
                "agg_method",
                "ishuf",
                "evt_lab",
                "evt_fm",
                act_name,
            ]
        ]
        .astype({"evt_fm": int})
        .copy()
    )


def separate_resp(r):
    parts = r.split("-")
    try:
        return parts[1]
    except IndexError:
        return parts[0]
