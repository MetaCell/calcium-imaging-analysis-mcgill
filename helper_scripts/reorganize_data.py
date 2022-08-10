#%% imports and definitions
import os
import shutil
import warnings

import pandas as pd
import xarray as xr

IN_PS_PATH = "/home/jovyan/ExpanDrive/Dropbox/06. Metacell analysis/Data/Processed_PV/active_passive_longit combined"
IN_MAP_PATH = "/home/jovyan/ExpanDrive/Dropbox/06. Metacell analysis/Data/Processed_PV/active_passive_longit combined (cellReg)"
IN_RAW_PATH = "/home/jovyan/ExpanDrive/Dropbox/06. Metacell analysis/Data/Raw videos"
IN_DS_FNAME = "minian_dataset.nc"
IN_CP_FILES = ["arduinolog.txt", "timeStamps.csv"]
IN_MP_FILE = "mappings.pkl"
IN_SS_CSV = "../metadata/sessions.csv"
DRY_RUN = False
OUT_SS_PATH = "../metadata/sessions.csv"
OUT_PS_PATH = "../intermediate/processed"
OUT_MP_PATH = "../intermediate/cross_reg"
OUT_RAW_PATH = "/home/jovyan/ExpanDrive/Dropbox/06. Metacell analysis/Data/raw"

os.makedirs(OUT_PS_PATH, exist_ok=True)
os.makedirs(OUT_MP_PATH, exist_ok=True)

#%% load session csv
sessions = pd.read_csv(IN_SS_CSV).set_index("data")

#%% walk through folders and copy data
for root, dirs, files in os.walk(IN_PS_PATH):
    if IN_DS_FNAME in files:
        ds = xr.open_dataset(os.path.join(root, IN_DS_FNAME))
        path_ls = root.split(os.sep)
        ss_type, anm, day = path_ls[-3], path_ls[-1], path_ls[-2]
        date = ds.coords["session"].values.item()
        dpath = os.path.join(anm, date, "My_V4_Miniscope")
        meta = sessions.loc[dpath]
        ss = meta["session"]
        ds = ds.assign_coords(animal=anm, session=ss)
        ps_path = os.path.join(OUT_PS_PATH, "{}-{}.nc".format(anm, ss))
        raw_path = os.path.join(OUT_RAW_PATH, dpath)
        for fname in IN_CP_FILES:
            assert fname in files
        print("saving data to {}".format(ps_path))
        print("copying files to {}".format(raw_path))
        if not DRY_RUN:
            ds.to_netcdf(ps_path)
            os.makedirs(raw_path, exist_ok=True)
            for fname in IN_CP_FILES:
                shutil.copyfile(
                    os.path.join(root, fname), os.path.join(raw_path, fname)
                )

#%% walk through folders and copy mappings
sessions_anm = sessions.reset_index().set_index("animal")
for root, dirs, files in os.walk(IN_MAP_PATH):
    if IN_MP_FILE in files:
        mp = pd.read_pickle(os.path.join(root, IN_MP_FILE))
        anm = root.split(os.sep)[-1]
        ss_map = (
            sessions_anm.loc[anm][["date", "session"]]
            .set_index("date")["session"]
            .to_dict()
        )
        mp["variable", "animal"] = anm
        mp = mp.rename(columns=ss_map).drop(columns=[("group", "group")])
        assert all(c.startswith("Day") for c in mp["session"].columns)
        save_path = os.path.join(OUT_MP_PATH, "{}.pkl".format(anm))
        print("saving to {}".format(save_path))
        if not DRY_RUN:
            mp.to_pickle(save_path)

#%% walk through folders and move files
for root, dirs, files in os.walk(IN_RAW_PATH):
    path_ls = os.path.relpath(root, IN_RAW_PATH).split(os.sep)
    if path_ls[-1] == "My_V4_Miniscope":
        if len(path_ls) == 4:
            anm, date, time, _ = path_ls
        elif len(path_ls) == 3:
            anm, date, _ = path_ls
        else:
            continue
        outpath = os.path.join(anm, date, "My_V4_Miniscope")
        meta_row = sessions.loc[outpath]
        assert isinstance(meta_row, pd.Series)
        outpath = os.path.join(OUT_RAW_PATH, outpath)
        print("copying data from {} to {}".format(root, outpath))
        if not DRY_RUN:
            if os.path.exists(outpath):
                shutil.rmtree(outpath, ignore_errors=True)
            os.makedirs(outpath, exist_ok=True)
            for fname in IN_CP_FILES:
                try:
                    shutil.copyfile(
                        os.path.join(root, fname), os.path.join(outpath, fname)
                    )
                except FileNotFoundError:
                    warnings.warn("cannot find {}".format(os.path.join(root, fname)))
