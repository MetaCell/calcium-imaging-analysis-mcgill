import itertools as itt
import os
import pandas as pd

import numpy as np


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
    for df in dfs:
        for k, v in meta.items():
            df[k] = v
    return dfs


def df_map_values(dfs: list, mappings: dict):
    if type(dfs) is not list:
        dfs = [dfs]
    for df in dfs:
        for col, mp in mappings.items():
            df[col] = df[mp[0]].map(mp[1])
    return dfs
