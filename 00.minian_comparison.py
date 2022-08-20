"""
Script to compare minian results.
environment: minian.yml
"""
#%% imports and definitions
import os

import pandas as pd
import plotly.express as px
import xarray as xr
from minian.utilities import open_minian
from minian.visualization import norm

from routine.plotting import imshow
from routine.utilities import df_set_metadata

IN_ORG_DS = [
    "./data/raw/Cohort 8-3.6/p1.4/2021_03_22/13_18_44/My_V4_Miniscope/minian",
    "./data/raw/Cohort 8-3.6/q1.1/2021_03_22/11_52_16/My_V4_Miniscope/minian",
]
IN_NEW_DS = ["./intermediate/p1.4/minian_ds", "./intermediate/q1.1/minian_ds"]
FIG_PATH = "./figs/comparison"
os.makedirs(FIG_PATH, exist_ok=True)


#%% generate spatial footprints
A_df = []
ncell_df = []
for org_path, new_path in zip(IN_ORG_DS, IN_NEW_DS):
    ds_org = open_minian(org_path)
    ds_new = open_minian(new_path)
    anm, ss = (
        ds_org.coords["animal"].values.item(),
        ds_org.coords["session"].values.item(),
    )
    A_org = xr.apply_ufunc(
        norm,
        ds_org["A"],
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
    )
    A_org = A_org.max("unit_id").compute().to_dataframe().reset_index()
    A_new = ds_new["A"].max("unit_id").compute().to_dataframe().reset_index()
    max_proj = norm(ds_new["max_proj"]).rename("A").to_dataframe().reset_index()
    A_org = df_set_metadata(A_org, {"animal": anm, "session": ss, "run": "org"})
    A_new = df_set_metadata(A_new, {"animal": anm, "session": ss, "run": "new"})
    max_proj = df_set_metadata(max_proj, {"animal": anm, "session": ss, "run": "max"})
    A_df.extend([A_org, A_new, max_proj])
    ncell_df.extend(
        [
            pd.Series(
                {
                    "animal": anm,
                    "session": ss,
                    "run": "org",
                    "ncell": ds_org.sizes["unit_id"],
                }
            )
            .to_frame()
            .T,
            pd.Series(
                {
                    "animal": anm,
                    "session": ss,
                    "run": "new",
                    "ncell": ds_new.sizes["unit_id"],
                }
            )
            .to_frame()
            .T,
        ]
    )
A_df = pd.concat(A_df, ignore_index=True)
ncell_df = pd.concat(ncell_df, ignore_index=True)
A_df["ss"] = A_df["animal"] + "-" + A_df["session"]
ncell_df["ss"] = ncell_df["animal"] + "-" + ncell_df["session"]

#%% plot spatial footprints
fig_A = imshow(
    A_df,
    facet_row="ss",
    facet_col="run",
    x="width",
    y="height",
    z="A",
    coloraxis="coloraxis",
    subplot_args={"shared_xaxes": "rows", "shared_yaxes": "rows"},
)
fig_A.write_html(os.path.join(FIG_PATH, "footprints.html"))

#%% plot number of cells
fig_ncell = px.bar(ncell_df, x="ss", y="ncell", color="run", barmode="group")
fig_ncell.write_html(os.path.join(FIG_PATH, "ncells.html"))
