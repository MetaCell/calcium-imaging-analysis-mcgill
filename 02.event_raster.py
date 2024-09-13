#%% imports and definitions
import itertools as itt
import os

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from routine.cell_resp import agg_cue, classify_cells
from routine.utilities import df_roll, df_set_metadata, iter_ds, norm, thres_gmm

IN_SS = "./metadata/sessions.csv"
PARAM_SUB_ANM = ["p3.2.1", "q1.4.1"]
PARAM_ACT_VARS = ["C", "S", "S_bin"]
PARAM_SESS = ["Day2_1", "Day3_8", "Day4_1", "Day4_9", "Day4_19"]
PARAM_CLASS_BY = "Day4_19"
PARAM_AGG_SUB = (-60, 60)
PARAM_NSHUF = 50
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
PARAM_SMOOTH = None
OUT_PATH = "./intermediate/event_raster"
FIG_PATH = "./figs/event_raster"

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


def standarize_df(df, act_name="C"):
    return (
        df[
            [
                "animal",
                "session",
                "unit_id",
                "master_uid",
                "by_event",
                "agg_method",
                "ishuf",
                "evt_lab",
                "evt_fm",
                act_name,
            ]
        ]
        .astype({"evt_fm": int, "master_uid": int})
        .copy()
    )


#%% load data and compute shuffled mean
for act_name in PARAM_ACT_VARS:
    act_store = pd.HDFStore(
        os.path.join(OUT_PATH, "{}mean.h5".format(act_name)), mode="w"
    )
    for method, ename in itt.product(PARAM_AGG_METHOD, PARAM_EVT):
        print("processing {} by {} using {}".format(act_name, ename, method))
        evars = PARAM_EVT[ename]
        for (anm, ss), ps_ds in iter_ds(sub_anm=PARAM_SUB_ANM, subset_reg=PARAM_SESS):
            if act_name == "S_bin":
                act = xr.apply_ufunc(
                    thres_gmm,
                    ps_ds["S"],
                    input_core_dims=[["frame"]],
                    output_core_dims=[["frame"]],
                    vectorize=True,
                ).rename("S_bin")
                act_df = act.to_dataframe().reset_index()
            else:
                act_df = ps_ds[act_name].to_dataframe().reset_index()
            act_mean = agg_cue(
                act_df,
                fm_name=evars[0],
                lab_name=evars[1],
                agg_method=method,
                act_name=act_name,
            )
            act_mean = standarize_df(
                df_set_metadata(
                    act_mean,
                    {
                        "ishuf": -1,
                        "animal": anm,
                        "session": ss,
                        "by_event": ename,
                        "agg_method": method,
                    },
                ),
                act_name=act_name,
            )
            act_store.append(
                "mean",
                act_mean,
                format="t",
                data_columns=True,
                min_itemsize=PARAM_ITEM_SIZE,
            )
            if ss == PARAM_CLASS_BY:
                for ishuf in range(PARAM_NSHUF):
                    act_shuf = act_df.groupby("unit_id").apply(
                        df_roll, col_name=act_name
                    )
                    act_mean = agg_cue(
                        act_shuf,
                        fm_name=evars[0],
                        lab_name=evars[1],
                        agg_method=method,
                        act_name=act_name,
                    )
                    act_mean = standarize_df(
                        df_set_metadata(
                            act_mean,
                            {
                                "ishuf": ishuf,
                                "animal": anm,
                                "session": ss,
                                "by_event": ename,
                                "agg_method": method,
                            },
                        ),
                        act_name=act_name,
                    )
                    act_store.append(
                        "mean",
                        act_mean,
                        format="t",
                        data_columns=True,
                        min_itemsize=PARAM_ITEM_SIZE,
                    )
    act_store.close()

#%% load shuffled mean and classify cells
for act_name in PARAM_ACT_VARS:
    act_mean = dd.read_hdf(
        os.path.join(OUT_PATH, "{}mean.h5".format(act_name)),
        "mean",
        chunksize=10000000,
    )
    act_mean = act_mean[
        (act_mean["evt_fm"].between(*PARAM_AGG_SUB))
        & (act_mean["session"] == PARAM_CLASS_BY)
    ]
    cell_lab = (
        act_mean.groupby(["animal", "master_uid", "by_event", "agg_method", "evt_lab"])
        .apply(
            classify_cells,
            sig=PARAM_SIG_THRES,
            meta=pd.DataFrame([{"cell_type": "active", "qscore": 0.1, "dff": 0.1}]),
            act_name=act_name,
        )
        .compute()
        .reset_index()
    )
    cell_lab.to_csv(
        os.path.join(OUT_PATH, "cell_lab-{}.csv".format(act_name)), index=False
    )

#%% plot raster
cmap = {"C": "GnBu", "S": "Blues", "S_bin": "Blues"}
sns.set(font_scale=1.6)


def act_heat(data, cell_lab, act_name, **kwargs):
    idx_dims = ["animal", "master_uid", "evt_lab"]
    data = data[idx_dims + ["evt_fm", act_name]].merge(
        cell_lab[idx_dims], on=idx_dims, how="inner"
    )
    if len(data) > 0:
        data["id"] = data["animal"] + "-" + data["master_uid"].astype(str)
        data["evt_fm"] = data["evt_fm"] / 20
        data[act_name] = data.groupby("id")[act_name].transform(norm)
        data = data.pivot(index="id", columns="evt_fm", values=act_name)
        ax = plt.gca()
        sns.heatmap(
            data,
            cmap=cmap[act_name],
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
        )
        ax.axvline(x=60, linestyle=":", color="grey")
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_edgecolor("darkgray")


for act_name in PARAM_ACT_VARS:
    cell_lab = pd.read_csv(os.path.join(OUT_PATH, "cell_lab-{}.csv".format(act_name)))
    act = dd.read_hdf(
        os.path.join(OUT_PATH, "{}mean.h5".format(act_name)), "mean", chunksize=10000000
    )
    act = act[act["ishuf"] == -1].compute()
    for (agg, typ), cur_lab in cell_lab.groupby(["agg_method", "cell_type"]):
        cur_act = act[act["agg_method"] == agg]
        g = sns.FacetGrid(
            cur_act,
            col="session",
            row="evt_lab",
            sharey="row",
            sharex=True,
            margin_titles=True,
            col_order=PARAM_SESS,
            despine=False,
        )
        g.map_dataframe(act_heat, cell_lab=cur_lab, act_name=act_name)
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.set_axis_labels(x_var="time", y_var="cells")
        fig = g.figure
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(
            os.path.join(FIG_PATH, "{}-{}-{}.svg".format(act_name, agg, typ)), dpi=500
        )
        plt.close(fig)
