import re

import numpy as np
import pandas as pd

LOG_PAT = r"(?P<timestamp>[\d:.]+) -> (?P<event>[^,]*),?(?P<data>.*)\n"


def read_arduinolog(alog_file):
    with open(alog_file) as alog:
        log_df = pd.DataFrame(
            [re.search(LOG_PAT, l).groupdict() for l in alog.readlines()]
        )
    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
    return log_df


def sync_frames(log_df, nframe, truncate=True):
    on_evt = log_df[log_df["event"].isin(["scope on", "pre-scope on"])]
    assert len(on_evt) == 1
    log_df["timestamp"] = log_df["timestamp"] - on_evt["timestamp"].values[0]
    off_evt = log_df[log_df["event"] == "session end"]
    assert len(off_evt == 1)
    t_session = off_evt["timestamp"].values[0]
    frame_freq = t_session / nframe
    log_df["frame"] = np.around(log_df["timestamp"] / frame_freq).astype(int)
    if truncate:
        return log_df[log_df["frame"] <= nframe].copy()
    else:
        return log_df


def peri_evt_label(log_df, events, nfm, period):
    evt_sub = log_df[log_df["event"].isin(events)]
    evt_fm, evt_lab = evt_sub["frame"], evt_sub["event"]
    out_evt_fm = np.full(nfm, np.nan)
    out_evt_lab = np.full(nfm, "", dtype="<U{}".format(max(len(e) for e in events)))
    for fm, lab in zip(evt_fm, evt_lab):
        f_before = min(period[0], fm)
        f_after = min(period[1], nfm - fm)
        out_evt_fm[(fm - f_before) : (fm + f_after + 1)] = np.arange(
            -f_before, f_after + 1
        )
        out_evt_lab[(fm - f_before) : (fm + f_after + 1)] = lab
    return out_evt_fm, out_evt_lab
