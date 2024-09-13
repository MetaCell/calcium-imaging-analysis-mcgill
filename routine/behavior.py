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
        f_after = min(period[1], nfm - fm - 1)
        out_evt_fm[(fm - f_before) : (fm + f_after + 1)] = np.arange(
            -f_before, f_after + 1
        )
        out_evt_lab[(fm - f_before) : (fm + f_after + 1)] = lab
    return out_evt_fm, out_evt_lab


def label_trials(log_df, trial_bounds=["pre-trials"]):
    return log_df["event"].isin(trial_bounds).astype(int).cumsum()


def label_trial_result(trial_df, trial_lab=["GO trials", "NOGO trials", "premature"]):
    # extract trial type
    ttype_row = trial_df[trial_df["event"].isin(trial_lab)]
    if len(ttype_row) == 0:
        return trial_df
    assert len(ttype_row) == 1, "multiple trial type found: {}".format(
        ttype_row["event"]
    )
    ttype = ttype_row["event"].values.item().split()[0]
    if ttype == "premature":
        return trial_df
    # determine response
    resps = trial_df["event"]
    if ttype == "GO":
        r = resps.loc[ttype_row.index + 1]
        if r.values.item() == "active":  # correct
            resp_add = trial_df.loc[r.index].copy()
            resp_add["event"] = "-".join((ttype, "correct"))
        else:  # incorrect
            resp_add = ttype_row.copy()
            resp_add["event"] = "-".join((ttype, "incorrect"))
    elif ttype == "NOGO":
        correct_resp = resps[resps.isin([ttype + "rewards", "rewards"])]
        if len(correct_resp) == 1:  # correct
            resp_add = trial_df.loc[correct_resp.index].copy()
            resp_add["event"] = "-".join((ttype, "correct"))
        elif len(correct_resp) == 0:  # incorrect
            wrong_resp = resps[resps == "comissions"]
            assert (
                len(wrong_resp) == 1
            ), "NOGO trials without reward should have one 'comissions' response"
            resp_add = trial_df.loc[wrong_resp.index].copy()
            resp_add["event"] = "-".join((ttype, "incorrect"))
        else:
            raise ValueError("Multiple reward value found:\n{}".format(trial_df))
    else:
        raise ValueError("Don't understand trial type: {}".format(ttype))
    trial_df = pd.concat([trial_df, resp_add], ignore_index=True)
    return trial_df
