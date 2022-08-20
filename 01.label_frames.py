#%% imports and definition
import os
import warnings

import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from routine.behavior import peri_evt_label, read_arduinolog, sync_frames

IN_DPATH = "./data"
IN_PS_PATH = "./intermediate/processed"
IN_SS_CSV = "./metadata/sessions.csv"
PARAM_CUE_PRD = (60, 200)
OUT_PATH = "./intermediate/frame_labels"
os.makedirs(OUT_PATH, exist_ok=True)

#%% walk through sessions and label frames
sessions = pd.read_csv(IN_SS_CSV)
for _, row in tqdm(list(sessions.iterrows())):
    log_file = os.path.join(IN_DPATH, row["data"], "arduinolog.txt")
    try:
        log_df = read_arduinolog(log_file)
    except FileNotFoundError:
        warnings.warn("cannot find arduino log for {}".format(row["data"]))
        continue
    try:
        minian_ds = xr.load_dataset(
            os.path.join(IN_PS_PATH, "{}-{}.nc".format(row["animal"], row["session"]))
        )
    except FileNotFoundError:
        warnings.warn("cannot find processed data for {}".format(row["data"]))
        continue
    nfm = minian_ds.sizes["frame"]
    log_df = sync_frames(log_df, nfm)
    evt_fm, evt_lab = peri_evt_label(
        log_df, ["GO trials", "NOGO trials"], nfm, PARAM_CUE_PRD
    )
    cue_fm = xr.DataArray(
        evt_fm,
        dims=["frame"],
        coords={"frame": minian_ds.coords["frame"]},
        name="cue_fm",
    )
    cue_lab = xr.DataArray(
        evt_lab,
        dims=["frame"],
        coords={"frame": minian_ds.coords["frame"]},
        name="cue_lab",
    )
    lab_ds = xr.merge([cue_fm, cue_lab]).assign_coords(
        animal=row["animal"], session=row["session"]
    )
    lab_ds.to_netcdf(
        os.path.join(OUT_PATH, "{}-{}.nc".format(row["animal"], row["session"]))
    )
