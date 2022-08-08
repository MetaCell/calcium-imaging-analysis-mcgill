#%% imports and definitions
import os
import shutil

import pandas as pd
import xarray as xr

IN_PS_PATH = "/home/jovyan/ExpanDrive/Dropbox/06. Metacell analysis/Data/Processed_PV/active_passive_longit combined"
IN_MAP_PATH = "/home/jovyan/ExpanDrive/Dropbox/06. Metacell analysis/Data/Processed_PV/active_passive_longit combined (cellReg)"
IN_RAW_PATH = "/home/jovyan/ExpanDrive/Dropbox/06. Metacell analysis/Data/Raw videos"
IN_DS_FNAME = "minian_dataset.nc"
IN_CP_FILES = ["arduinolog.txt", "timeStamps.csv"]
IN_MP_FILE = "mappings.pkl"
DRY_RUN = False
OUT_SS_PATH = "../metadata/sessions.csv"
OUT_PS_PATH = "../intermediate/processed"
OUT_MP_PATH = "../intermediate/cross_reg"
OUT_RAW_PATH = "../data"

os.makedirs(OUT_PS_PATH, exist_ok=True)
os.makedirs(OUT_MP_PATH, exist_ok=True)
ss_ls = []


#%% walk through folders and copy data
for root, dirs, files in os.walk(IN_PS_PATH):
    if IN_DS_FNAME in files:
        ds = xr.open_dataset(os.path.join(root, IN_DS_FNAME))
        path_ls = root.split(os.sep)
        ss_type, anm, day = path_ls[-3], path_ls[-1], path_ls[-2]
        ss = ds.coords["session"].values.item()
        ds = ds.assign_coords(animal=anm, session=ss)
        ps_path = os.path.join(OUT_PS_PATH, "{}-{}.nc".format(anm, ss))
        raw_path = os.path.join(OUT_RAW_PATH, anm, ss, "My_V4_Miniscope")
        for fname in IN_CP_FILES:
            assert fname in files
        row = pd.DataFrame(
            [
                {
                    "animal": anm,
                    "session": ss,
                    "session_type": ss_type,
                    "day": day,
                    "data": os.path.relpath(raw_path, OUT_RAW_PATH),
                }
            ]
        )
        ss_ls.append(row)
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
for root, dirs, files in os.walk(IN_MAP_PATH):
    if IN_MP_FILE in files:
        mp = pd.read_pickle(os.path.join(root, IN_MP_FILE))
        anm = root.split(os.sep)[-1]
        mp["variable", "animal"] = anm
        save_path = os.path.join(OUT_MP_PATH, "{}.pkl".format(anm))
        print("saving to {}".format(save_path))
        if not DRY_RUN:
            mp.to_pickle(save_path)

#%% walk through folders and make symlink
for root, dirs, files in os.walk(IN_RAW_PATH):
    path_ls = os.path.relpath(root, IN_RAW_PATH).split(os.sep)
    if len(path_ls) == 4:
        cohort, anm, ss, time = path_ls
        outpath = os.path.join(OUT_RAW_PATH, anm, ss, time)
        row = pd.DataFrame(
            [
                {
                    "animal": anm,
                    "session": ss,
                    "cohort": cohort,
                    "data": os.path.relpath(outpath, OUT_RAW_PATH),
                }
            ]
        )
        ss_ls.append(row)
        print("linking data from {} to {}".format(root, outpath))
        if not DRY_RUN:
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            os.symlink(root, outpath)

#%% write session csv
sessions = pd.concat(ss_ls, ignore_index=True).sort_values(["animal", "session"])
os.makedirs(os.path.dirname(OUT_SS_PATH), exist_ok=True)
sessions.to_csv(OUT_SS_PATH)
