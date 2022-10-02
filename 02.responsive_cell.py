#%% imports and definitions
import itertools as itt
import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import medfilt

from routine.utilities import df_roll, df_set_metadata, iter_ds, norm, q_score

IN_SS = "./metadata/sessions.csv"
PARAM_AGG_SUB = (-60, 60)
PARAM_NSHUF = 10
PARAM_SIG_THRES = 0.05
PARAM_EVT = {"cue": ("cue_fm", "cue_lab"), "resp": ("resp_fm", "resp_lab")}
PARAM_AGG_METHOD = ["minmax", "zscore"]
PARAM_ITEM_SIZE = {
    "animal": 10,
    "session": 10,
    "by_event": max([len(e) for e in PARAM_EVT]),
    "agg_method": max([len(m) for m in PARAM_AGG_METHOD]),
    "evt_lab": 20,
}
PARAM_SMOOTH = 7
OUT_PATH = "./intermediate/responsive_cell"
FIG_PATH = "./figs/responsive_cell"

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


def agg_cue(df, fm_name, lab_name, agg_method):
    trace = (
        df[df[fm_name].notnull()]
        .groupby(["unit_id", lab_name, fm_name])
        .mean()
        .reset_index()
    )
    trace["C"] = trace.groupby(["unit_id", lab_name])["C"].transform(
        medfilt, kernel_size=PARAM_SMOOTH
    )
    trace = (
        trace.groupby(["unit_id", lab_name])
        .apply(agg_baseline, fm_name=fm_name, method=agg_method)
        .rename(columns={fm_name: "evt_fm", lab_name: "evt_lab"})
    )
    return trace


def agg_baseline(df, fm_name, method="minmax"):
    tlab = np.sign(df[fm_name])
    if method == "minmax":
        df["C"] = norm(df["C"]) * 99 + 1
        if df["C"].notnull().all():
            baseline = df.loc[tlab < 0, "C"].mean()
            assert baseline > 0
            df["C"] = (df["C"] - baseline) / baseline
        else:
            df["C"] = np.nan
    elif method == "zscore":
        baseline = df.loc[tlab < 0, "C"].mean()
        std = np.std(df["C"])
        if std > 0:
            df["C"] = (df["C"] - baseline) / std
        else:
            df["C"] = np.nan
    return df


def agg_dff(df, fm_name):
    tlab = np.sign(df[fm_name])
    return df.loc[tlab > 0, "C"].sum()


def classify_cells(df, sig, fm_name="evt_fm"):
    dff = df.groupby("ishuf").apply(agg_dff, fm_name=fm_name)
    q = q_score(dff.loc[0:], dff.loc[-1])
    if q > 1 - sig:
        lab = "activated"
    elif q < sig:
        lab = "inhibited"
    else:
        lab = "non-modulated"
    return pd.Series({"cell_type": lab, "qscore": q, "dff": dff.loc[-1]})


def standarize_df(df):
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
                "C",
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


#%% load data and compute shuffled mean
Cmean_store = pd.HDFStore(os.path.join(OUT_PATH, "Cmean.h5"), mode="w")
for method, ename in itt.product(PARAM_AGG_METHOD, PARAM_EVT):
    print("processing by {} using {}".format(ename, method))
    evars = PARAM_EVT[ename]
    for (anm, ss), ps_ds in iter_ds(subset_reg=True):
        C_df = ps_ds["C"].to_dataframe().reset_index()
        Cmean = agg_cue(C_df, fm_name=evars[0], lab_name=evars[1], agg_method=method)
        Cmean = standarize_df(
            df_set_metadata(
                Cmean,
                {
                    "ishuf": -1,
                    "animal": anm,
                    "session": ss,
                    "by_event": ename,
                    "agg_method": method,
                },
            )
        )
        Cmean_store.append(
            "Cmean", Cmean, format="t", data_columns=True, min_itemsize=PARAM_ITEM_SIZE
        )
        for ishuf in range(PARAM_NSHUF):
            C_shuf = C_df.groupby("unit_id").apply(df_roll)
            Cmean = agg_cue(
                C_shuf, fm_name=evars[0], lab_name=evars[1], agg_method=method
            )
            Cmean = standarize_df(
                df_set_metadata(
                    Cmean,
                    {
                        "ishuf": ishuf,
                        "animal": anm,
                        "session": ss,
                        "by_event": ename,
                        "agg_method": method,
                    },
                )
            )
            Cmean_store.append(
                "Cmean",
                Cmean,
                format="t",
                data_columns=True,
                min_itemsize=PARAM_ITEM_SIZE,
            )
Cmean_store.close()

#%% load shuffled mean and classify cells
Cmean = dd.read_hdf(os.path.join(OUT_PATH, "Cmean.h5"), "Cmean", chunksize=10000000)
Cmean = Cmean[Cmean["evt_fm"].between(*PARAM_AGG_SUB)]
cell_lab = (
    Cmean.groupby(["animal", "session", "by_event", "agg_method", "unit_id", "evt_lab"])
    .apply(
        classify_cells,
        sig=PARAM_SIG_THRES,
        meta=pd.DataFrame([{"cell_type": "active", "qscore": 0.1, "dff": 0.1}]),
    )
    .compute()
    .reset_index()
)
cell_lab.to_csv(os.path.join(OUT_PATH, "cell_lab.csv"), index=False)

#%% plot activity
plt_dict = {
    "cue": {
        "Day 1 GO trials": "green",
        "Day 1 NOGO trials": "red",
        "Day 19 GO trials": "green",
        "Day 19 NOGO trials": "red",
    },
    "resp": {
        "Day 1 correct": "green",
        "Day 1 incorrect": "red",
        "Day 19 correct": "green",
        "Day 19 incorrect": "red",
    },
}
dash_dict = {
    "cue": {
        "Day 1 GO trials": "",
        "Day 1 NOGO trials": "",
        "Day 19 GO trials": (5, 5),
        "Day 19 NOGO trials": (5, 5),
    },
    "resp": {
        "Day 1 correct": "",
        "Day 1 incorrect": "",
        "Day 19 correct": (5, 5),
        "Day 19 incorrect": (5, 5),
    },
}
sns.set_theme(font="Calibri", font_scale=1.2)
sns.set_style("ticks")
grp_map = {"active": "Active Solvers", "passive": "Passive Solvers"}
yrange_pad_fac = 1.1
Cmean = dd.read_hdf(os.path.join(OUT_PATH, "Cmean.h5"), "Cmean", chunksize=10000000)
cell_lab = pd.read_csv(os.path.join(OUT_PATH, "cell_lab.csv"))
Cmean = Cmean[Cmean["ishuf"] == -1].compute()
sessions = pd.read_csv(IN_SS).drop(columns=["date", "data"])
sessions["day"] = sessions["session"].apply(lambda s: s.split("_")[1])
sessions["group"] = sessions["group"].map(grp_map)
cell_df = cell_lab.merge(sessions, how="left", on=["animal", "session"])
Cmean["time"] = Cmean["evt_fm"] / 20
Cmean["C"] = Cmean["C"] * 100
Cmean["evt_lab"] = Cmean["evt_lab"].map(separate_resp)
cell_df["evt_lab"] = cell_df["evt_lab"].map(separate_resp)
plot_df = Cmean.drop(columns=["ishuf"]).merge(
    cell_df, on=["animal", "session", "unit_id", "by_event", "agg_method", "evt_lab"]
)
plot_df["trace_type"] = "Day " + plot_df["day"] + " " + plot_df["evt_lab"]
for (evt, method), subdf in plot_df.groupby(["by_event", "agg_method"]):
    range_df = (
        subdf.groupby(["cell_type", "group", "trace_type", "time"])["C"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    range_df = range_df[range_df["cell_type"].isin(["inhibited", "non-modulated"])]
    yrange = np.array(
        [
            (range_df["mean"] - range_df["sem"]).min(),
            (range_df["mean"] + range_df["sem"]).max(),
        ]
    )
    yrange = yrange * yrange_pad_fac
    g = sns.relplot(
        data=subdf,
        x="time",
        y="C",
        row="cell_type",
        col="group",
        kind="line",
        hue="trace_type",
        style="trace_type",
        facet_kws={"sharey": "row"},
        errorbar="se",
        row_order=["activated", "inhibited", "non-modulated"],
        height=3.5,
        aspect=1.2,
        palette=plt_dict[evt],
        dashes=dash_dict[evt],
        hue_order=list(plt_dict[evt].keys()),
    )
    for ax in g.axes.flat:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("DeltaF/F (%baseline)")
        ax.set_xticks([-3, 0, 3, 10])
        ax.axvline(0, color="gray")
        ax.axhline(0, color="gray")
        ax.axvspan(0, 3, color="gray", alpha=0.4)
        tt = ax.title.get_text()
        if "inhibited" in tt or "non-modulated" in tt:
            ax.set_ylim(yrange)
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.savefig(os.path.join(FIG_PATH, "{}-{}.svg".format(evt, method)))
    g.savefig(os.path.join(FIG_PATH, "{}-{}.png".format(evt, method)), dpi=300)
