#%% imports and definitions
import os
import warnings

import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

DRY_RUN = False
IN_SS_CSV = "./metadata/sessions.csv"
IN_PS_PATH = "./intermediate/processed"
IN_MP_PATH = "./intermediate/cross_reg"

#%% load session csv
sessions = pd.read_csv(IN_SS_CSV)

#%% validate processed data
for _, row in tqdm(list(sessions.iterrows())):
    anm, ss = row["animal"], row["session"]
    ds_path = os.path.join(IN_PS_PATH, "{}-{}.nc".format(anm, ss))
    try:
        minian_ds = xr.load_dataset(ds_path)
    except FileNotFoundError:
        warnings.warn("cannot find processed data for {}".format(row["data"]))
        continue
    fixDict = dict()
    anm_ds, ss_ds = (
        minian_ds.coords["animal"].values.item(),
        minian_ds.coords["session"].values.item(),
    )
    if anm_ds != anm:
        fixDict[anm_ds] = anm
        minian_ds = minian_ds.assign_coords(animal=anm)
    if ss_ds != ss:
        fixDict[ss_ds] = ss
        minian_ds = minian_ds.assign_coords(session=ss)
    if fixDict:
        print("fixing {}".format(ds_path))
        print("; ".join(["{} -> {}".format(k, v) for k, v in fixDict.items()]))
        if not DRY_RUN:
            minian_ds.load().to_netcdf(ds_path)

#%% validate mappings
for anm, anm_df in tqdm(sessions.groupby("animal")):
    mp_path = os.path.join(IN_MP_PATH, "{}.pkl".format(anm))
    mp = pd.read_pickle(mp_path)
    ss_dict = anm_df.set_index("date")["session"].to_dict()
    sess = ss_dict.values()
    fixDict = dict()
    ss_cols = mp["session"].columns
    for s_mp in ss_cols[~ss_cols.isin(sess)]:
        fixDict[s_mp] = ss_dict[s_mp]
    if fixDict:
        print("fixing {}".format(mp_path))
        print("; ".join(["{} -> {}".format(k, v) for k, v in fixDict.items()]))
        if not DRY_RUN:
            mp.rename(columns=fixDict).drop(columns=[("group", "group")]).to_pickle(
                mp_path
            )
