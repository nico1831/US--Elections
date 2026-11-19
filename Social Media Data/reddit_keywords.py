"""
Shared settings for the Reddit pipeline: file paths, the election keyword list,
the post loader and the tagger. Every Phase 1 script imports from here, so the
keyword list lives in exactly one place.

When you change any pattern, bump KW_VERSION.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent      # the "Social Media Data" folder
RAW  = HERE / "reddit_raw"                  # 204 scraped CSVs (+ _authors.csv later)
OUT  = HERE / "phase1"                      # everything Phase 1 writes goes here

KW_VERSION = "v1 (2026-11-19)"

# General election words, built from the four Trends groups
#   friction: how/where to vote, polling place, sample ballot, voter id
#   access:   absentee, early voting, registration, mail-in, vote by mail, drop box
#   salience: vote/voting/voters, ballot, election, electoral college
GENERAL = (r"\b(?:vote|votes|voted|voting|voters?|ballots?|elections?|electoral college"
           r"|polling (?:place|location|station)s?|absentee|early voting|mail-in"
           r"|vote by mail|register(?:ed)? to vote|voter registration|voter id"
           r"|(?:ballot )?drop box(?:es)?|sample ballot|presidential debate)\b")

# Candidate group: nominees + running mates, per cycle (mirrors candidate_v2_index)
CANDS = {2012: r"\b(?:obama|romney|paul ryan|biden)\b",
         2016: r"\b(?:trump|hillary|clinton|pence|kaine)\b",
         2020: r"\b(?:trump|biden|pence|kamala|harris)\b",
         2024: r"\b(?:trump|harris|kamala|walz|vance)\b"}

# Trial words: kept only if the hand-check shows they are precise enough
TRIAL = r"\b(?:polls|debates?|candidates?)\b"

# Place and shop names removed before matching
EXCL = r"harris county|harris teeter|clinton township|clinton county"

# Which routes count as an election post (flip after the hand-check if needed)
USE_CAND_ONLY = True
USE_TRIAL     = True

# Political words, used only to find "political but untagged" posts for the hand-check
POL = (r"\b(?:governor|senate|senator|congress\w*|mayor|democrats?|republicans?|gop"
       r"|legislat\w+|amendment|politic\w*|campaign\w*|rally)\b")


def load_posts():
    """All 204 state-year files as one table, with a `text` column
    (title + body, or title only when the body was removed)."""
    files = sorted(RAW.glob("20*.csv"))
    assert len(files) == 204, f"expected 204 files in {RAW}, found {len(files)}"
    df = pd.concat([pd.read_csv(f, dtype={"id": str}) for f in files], ignore_index=True)
    df = df.drop_duplicates("id").reset_index(drop=True)
    body = df.selftext.fillna("").astype(str)
    df["removed"] = body.isin(["[removed]", "[deleted]"])
    df["text"] = (df.title.fillna("").astype(str) + " "
                  + body.where(~df.removed, "")).str.strip()
    return df


def tag(df):
    """Adds:
       gen, cand, trial : which keyword groups matched (True/False)
       route            : first group that matched: general > cand_only > trial_only > none
       elec             : counted as an election post under the current settings"""
    df = df.copy()
    df["clean"] = df.text.str.lower().str.replace(EXCL, " ", regex=True)
    df["gen"]   = df.clean.str.contains(GENERAL, regex=True)
    df["trial"] = df.clean.str.contains(TRIAL, regex=True)
    df["cand"]  = [bool(re.search(p, t)) for p, t in zip(df.year.map(CANDS), df.clean)]
    df["route"] = np.select([df.gen, df.cand, df.trial],
                            ["general", "cand_only", "trial_only"], default="none")
    df["elec"]  = df.gen | (df.cand & USE_CAND_ONLY) | (df.trial & USE_TRIAL)
    return df