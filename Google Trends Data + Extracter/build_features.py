#!/usr/bin/env python3
"""
build_features.py — turn the raw Trends exports into the modelling panel.

    python build_features.py --plan trends_pull_urls.csv --raw raw \
        --turnout "US Elections/Official Turnout Data/Presidential Elections" --out out

Pipeline
  1. load    every raw/*.csv, whether hand-downloaded or from fetch_trends.py
  2. validate reject "Compared breakdown" files (rows summing to 100%) — see K2
  3. screen  S1 coverage, S2 cross-state variance, S5 within-category redundancy
  4. index   z-score each surviving term across the 51 units within its year,
             then average z-scores within each category
  5. join    VEP turnout + turnout_prev, write panel.csv

Outputs (in --out): panel.csv, trends_features.csv, screening_report.csv,
term_matrix.csv (long, for the correlation heatmap and Cronbach's alpha).
"""
import argparse, csv, re, sys
from pathlib import Path
import pandas as pd
import numpy as np

STATES = [
 "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
 "Delaware","District of Columbia","Florida","Georgia","Hawaii","Idaho","Illinois",
 "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts",
 "Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
 "New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota",
 "Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina",
 "South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington",
 "West Virginia","Wisconsin","Wyoming"]

MIN_COVERAGE = 45      # S1: non-zero states required, in the WORST year
MIN_SD       = 5.0     # S2: cross-state SD required, in every year
MAX_R        = 0.90    # S5: within-category redundancy ceiling


# ---------------------------------------------------------------- load

def read_geomap(path):
    """Parse a Trends subregion export. Returns (Series indexed by state, note)."""
    raw = Path(path).read_text(encoding="utf-8-sig").splitlines()
    lines = [l for l in raw if l.strip()]
    hdr = next((i for i, l in enumerate(lines) if l.split(",")[0].strip() in ("Region", "Country")), None)
    if hdr is None:
        raise ValueError("no 'Region' header row — is this an Interest-over-time file?")

    rows = list(csv.reader(lines[hdr:]))
    header, body = rows[0], [r for r in rows[1:] if r and r[0].strip()]
    ncols = len(header) - 1
    if ncols > 1:
        raise ValueError(
            f"{ncols} value columns — this is a COMPARED breakdown (rows sum to 100%), "
            "not Interest by subregion. Re-pull this term on its own.")

    def num(x):
        x = x.strip().replace("%", "")
        if x in ("", "N/A"):        return np.nan
        if x.startswith("<"):       return 0.5      # Trends' "<1"
        try:    return float(x)
        except ValueError: return np.nan

    s = pd.Series({r[0].strip(): num(r[1]) for r in body}, dtype=float)
    note = ""
    if s.dropna().sum() > 0 and abs(s.sum() - 100) < 0.5 and len(s) < 10:
        note = "values sum to 100 — suspect compared mode"
    s = s.reindex(STATES)
    return s, note


def load_all(plan_path, raw_dir):
    plan = list(csv.DictReader(open(plan_path, encoding="utf-8-sig")))
    recs, missing, bad = [], [], []
    for row in plan:
        p = Path(raw_dir) / row["save_as"]
        if not p.exists():
            missing.append(row["save_as"]); continue
        try:
            s, note = read_geomap(p)
        except ValueError as e:
            bad.append((row["save_as"], str(e))); continue
        for state, val in s.items():
            recs.append({"year": int(row["year"]), "category": row["category"],
                         "term_id": row["term_id"], "term": row["term"], "state": state, "rsv": val})
        if note:
            bad.append((row["save_as"], note))
    return pd.DataFrame(recs), missing, bad


# ---------------------------------------------------------------- screen

def screen(long, expected_years=4):
    """expected_years: how many election years the panel needs. Relaxed automatically
    to the number actually downloaded, so partial runs still screen usefully."""
    per = (long.groupby(["category", "term_id", "year"])["rsv"]
              .agg(coverage=lambda s: int((s.fillna(0) > 0).sum()),
                   sd="std", max="max")
              .reset_index())

    worst = (per.groupby(["category", "term_id"])
                .agg(min_coverage=("coverage", "min"), min_sd=("sd", "min"),
                     min_max=("max", "min"), n_years=("year", "nunique"))
                .reset_index())

    years_present = long.year.nunique()
    required = min(expected_years, years_present)

    worst["drop_reason"] = ""
    worst.loc[worst.n_years < required, "drop_reason"] = \
        f"K4: has {worst.n_years}/{required} years"
    m = (worst.drop_reason == "") & (worst.min_coverage < MIN_COVERAGE)
    worst.loc[m, "drop_reason"] = f"S1: coverage < {MIN_COVERAGE}"
    m = (worst.drop_reason == "") & (worst.min_sd < MIN_SD)
    worst.loc[m, "drop_reason"] = f"S2: cross-state SD < {MIN_SD}"

    # S5 redundancy, pooled across years, within category
    keep = set(worst.loc[worst.drop_reason == "", "term_id"])
    wide = (long[long.term_id.isin(keep)]
              .pivot_table(index=["year", "state"], columns="term_id", values="rsv"))
    for cat, grp in worst[worst.drop_reason == ""].groupby("category"):
        terms = [t for t in grp.term_id if t in wide.columns]
        if len(terms) < 2:
            continue
        corr = wide[terms].corr()
        cov = worst.set_index("term_id")["min_coverage"]
        for i, a in enumerate(terms):
            for b in terms[i + 1:]:
                if a not in keep or b not in keep:
                    continue
                r = corr.loc[a, b]
                if pd.notna(r) and abs(r) > MAX_R:
                    loser = a if cov[a] < cov[b] else b
                    keep.discard(loser)
                    worst.loc[worst.term_id == loser, "drop_reason"] = \
                        f"S5: r={r:.2f} with '{b if loser == a else a}'"
    worst["included"] = worst.drop_reason == ""
    return per, worst


# ---------------------------------------------------------------- index

def build_indices(long, kept_terms):
    d = long[long.term_id.isin(kept_terms)].copy()
    # z-score each term across the 51 states, within year: cancels the per-file
    # normalisation constant, keeps only cross-sectional geography.
    g = d.groupby(["term_id", "year"])["rsv"]
    d["z"] = (d.rsv - g.transform("mean")) / g.transform("std")
    idx = (d.groupby(["state", "year", "category"])["z"].mean()
             .unstack("category").reset_index())
    idx.columns.name = None
    ren = {"C1_logistics": "logistics_index", "C2_salience": "salience_index",
           "C3_process": "process_index", "C4_candidates": "candidate_index"}
    return idx.rename(columns=ren), d


# ---------------------------------------------------------------- turnout

def load_turnout_db(path, year):
    """Pull one year's column out of 'USA Voter Turnout DB.csv' (used to seed 2008)."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if str(year) not in df.columns:
        return pd.DataFrame(columns=["state", "turnout"])
    d = df[["State", str(year)]].copy()
    d["state"] = d.State.astype(str).str.strip()
    d = d[d.state.isin(STATES)]
    d["turnout"] = pd.to_numeric(
        d[str(year)].astype(str).str.replace("%", "", regex=False).str.strip()
         .replace({"N/A": None, "NA": None, "": None}), errors="coerce")
    return d[["state", "turnout"]]


def load_turnout(folder):
    out = []
    for p in sorted(Path(folder).glob("Turnout_*G_*.csv")):
        year = int(re.search(r"Turnout_(\d{4})G", p.name).group(1))
        df = pd.read_csv(p, thousands=",")
        df.columns = [c.strip().upper() for c in df.columns]
        col = next((c for c in df.columns if "VEP_TURNOUT" in c), None)
        if col is None:
            print(f"  ! {p.name}: no VEP_TURNOUT_RATE column", file=sys.stderr); continue
        sub = df[["STATE", col]].copy()
        sub["state"] = (sub.STATE.astype(str).str.strip()
                .str.replace(r"[\*\u2020\u2021]+$", "", regex=True).str.strip())
        sub = sub[sub.state.isin(STATES)]
        sub["turnout"] = (sub[col].astype(str).str.replace("%", "", regex=False)
                          .str.strip().replace({"": None, "N/A": None}).astype(float))
        sub["year"] = year
        out.append(sub[["state", "year", "turnout"]])
    if not out:
        return pd.DataFrame(columns=["state", "year", "turnout"])
    return pd.concat(out, ignore_index=True).sort_values(["state", "year"])


def add_prev(t, db_path=None):
    """turnout_prev = previous presidential election. 2012 needs 2008, which the
    per-year files don't cover — seed it from USA Voter Turnout DB.csv if given."""
    seed = pd.DataFrame(columns=["state", "year", "turnout"])
    if db_path and Path(db_path).exists():
        s = load_turnout_db(db_path, 2008)
        if not s.empty:
            s["year"] = 2008
            seed = s[["state", "year", "turnout"]]
    full = (pd.concat([seed, t], ignore_index=True)
              .drop_duplicates(["state", "year"]).sort_values(["state", "year"]))
    full["turnout_prev"] = full.groupby("state")["turnout"].shift(1)
    return full[full.year != 2008].reset_index(drop=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="trends_pull_urls.csv")
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--turnout", default="")
    ap.add_argument("--turnout-db", default="",
                    help="path to 'USA Voter Turnout DB.csv' — supplies 2008 so that "
                         "2012 rows get a turnout_prev instead of being unusable")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    long, missing, bad = load_all(args.plan, args.raw)
    print(f"loaded {long.term_id.nunique() if len(long) else 0} terms "
          f"across {long.year.nunique() if len(long) else 0} years")
    if missing:
        print(f"  {len(missing)} files not yet downloaded, e.g. {missing[:3]}")
    for name, why in bad:
        print(f"  REJECTED {name}: {why}", file=sys.stderr)
    if long.empty:
        sys.exit("nothing to build — download some files first")

    years_present = sorted(long.year.unique())
    if len(years_present) < 4:
        print(f"\nPARTIAL RUN — {len(years_present)} of 4 election years present "
              f"({', '.join(map(str, years_present))}).\n"
              f"  K4 (balanced panel) is relaxed to the years you have. S1/S2/S5 still\n"
              f"  apply, but a term passing now can still fail once 2012 is complete.")

    per, report = screen(long, expected_years=4)
    per.to_csv(out / "per_term_year_stats.csv", index=False)
    report.to_csv(out / "screening_report.csv", index=False)
    kept = list(report.loc[report.included, "term_id"])
    print(f"\nscreening: {len(kept)}/{len(report)} terms kept")
    for _, r in report[~report.included].iterrows():
        print(f"  dropped {r.term_id:<24} {r.drop_reason}")
    if not kept:
        sys.exit("all terms screened out — check thresholds or data")
    print("  kept: " + ", ".join(kept))

    idx, zlong = build_indices(long, kept)
    zlong.to_csv(out / "term_matrix.csv", index=False)
    idx.to_csv(out / "trends_features.csv", index=False)

    panel = idx
    if args.turnout:
        t = add_prev(load_turnout(args.turnout), args.turnout_db)
        if not t.empty:
            panel = t.merge(idx, on=["state", "year"], how="left")
    panel.to_csv(out / "panel.csv", index=False)

    print(f"\nwrote {out}/panel.csv  ({len(panel)} rows, {len(panel.columns)} cols)")
    print(panel.head(3).to_string(index=False))
    n_missing = panel.isna().sum()
    if n_missing.any():
        print("\nmissing cells:\n" + n_missing[n_missing > 0].to_string())


if __name__ == "__main__":
    main()
