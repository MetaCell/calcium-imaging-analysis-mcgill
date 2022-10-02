import itertools as itt
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

IN_PS_PATH = "./intermediate/processed"
IN_LAB_PATH = "./intermediate/frame_labels"
IN_REG_PATH = "./intermediate/cross_reg"


def norm(a):
    amin, amax = np.nanmin(a), np.nanmax(a)
    return (a - amin) / (amax - amin)


def arr_break_idxs(a: np.ndarray):
    return np.where(a[:-1] != a[1:])[0] + 1


def walklevel(path, depth=1):
    if depth < 0:
        for root, dirs, files in os.walk(path):
            yield root, dirs[:], files
        return
    elif depth == 0:
        return
    base_depth = path.rstrip(os.path.sep).count(os.path.sep)
    for root, dirs, files in os.walk(path):
        yield root, dirs[:], files
        cur_depth = root.count(os.path.sep)
        if base_depth + depth <= cur_depth:
            del dirs[:]


def enumerated_product(*args):
    yield from zip(itt.product(*(range(len(x)) for x in args)), itt.product(*args))


def df_notnull(df: pd.DataFrame):
    if df.notnull().all(axis=None):
        return df
    else:
        return pd.DataFrame(columns=df.columns)


def corr_mat(a: np.ndarray, b: np.ndarray, agg_axis=0):
    a = a - a.mean(axis=agg_axis)
    b = b - b.mean(axis=agg_axis)
    return (a * b).sum(axis=agg_axis) / np.sqrt(
        (a**2).sum(axis=agg_axis) * (b**2).sum(axis=agg_axis)
    )


def df_set_metadata(dfs: list, meta: dict):
    if type(dfs) is not list:
        dfs = [dfs]
        return_single = True
    else:
        return_single = False
    for df in dfs:
        for k, v in meta.items():
            df[k] = v
    if return_single:
        return dfs[0]
    else:
        return dfs


def df_map_values(dfs: list, mappings: dict):
    if type(dfs) is not list:
        dfs = [dfs]
    for df in dfs:
        for col, mp in mappings.items():
            df[col] = df[mp[0]].map(mp[1])
    return dfs


def iter_ds(
    ds_path=IN_PS_PATH, lab_path=IN_LAB_PATH, reg_path=IN_REG_PATH, subset_reg=None
):
    for ps_ds in tqdm(os.listdir(ds_path)):
        ps_ds = xr.open_dataset(os.path.join(ds_path, ps_ds))
        anm, ss = (
            ps_ds.coords["animal"].values.item(),
            ps_ds.coords["session"].values.item(),
        )
        try:
            lab_ds = xr.open_dataset(os.path.join(lab_path, "{}-{}.nc".format(anm, ss)))
            ps_ds = ps_ds.assign_coords(lab_ds)
        except FileNotFoundError:
            warnings.warn("no frame label found for {} {}".format(anm, ss))
        if subset_reg is not None:
            if ss not in subset_reg:
                continue
            reg_df = pd.read_pickle(os.path.join(reg_path, "{}.pkl".format(anm)))
            reg_df = reg_df[reg_df["session"][subset_reg].notnull().all(axis="columns")]
            uids = np.sort(reg_df["session", ss]).astype(int)
            ps_ds = ps_ds.sel(unit_id=uids)
        yield (anm, ss), ps_ds


def df_roll(df, col_name="C", sh=None):
    if sh is None:
        sh = np.random.randint(len(df))
    df[col_name] = np.roll(df[col_name], sh)
    return df


def q_score(a, value):
    return np.nanmean(a < value)
