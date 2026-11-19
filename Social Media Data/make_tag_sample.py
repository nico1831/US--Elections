"""
Phase 1: tag every post with the current keyword list and build a blind
hand-check sample.

Round 1 (checks keyword list v1), 200 posts:
  tagged    25 per year = 15 general + 5 cand_only + 5 trial_only
  untagged  50 random + 50 that look political but were not tagged
Round 2 (after you revise the list), about 100 fresh posts never labeled before:
  tagged    13 per year, from posts the current list counts as election posts
  untagged  50 random

Output (in phase1/):
  route_counts.csv              posts per year and route
  tag_check_r<N>.csv            YOU label: is_election = 1 or 0 (note is optional)
  tag_check_r<N>_coder2.csv     first 60 rows for Schonn: is_election_2 = 1 or 0
  tag_check_r<N>_key.csv        which bucket each row came from (don't open while labeling)

Run:  python make_tag_sample.py --round 1
"""
import argparse, re, sys

import pandas as pd

from reddit_keywords import (OUT, KW_VERSION, GENERAL, TRIAL, CANDS, POL,
                             load_posts, tag)

N_CODER2 = 60
SNIP_PAT = re.compile("|".join([GENERAL, TRIAL, POL] + list(CANDS.values())))


def snippet(text, w=150):
    """About 300 characters around the first election or political word."""
    m = SNIP_PAT.search(text.lower())
    if m is None:
        return text[:2 * w]
    a, b = max(0, m.start() - w), m.end() + w
    return ("..." if a > 0 else "") + text[a:b] + ("..." if b < len(text) else "")


def take(pool, n, seed, kind, stratum, parts):
    """Random sample of n rows from pool; records the bucket and its full size."""
    if len(pool) < n:
        print(f"   note: {stratum} has only {len(pool)} posts, taking all")
    s = pool.sample(min(n, len(pool)), random_state=seed)
    parts.append(s.assign(kind=kind, stratum=stratum, stratum_size=len(pool)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True, choices=[1, 2])
    ap.add_argument("--force", action="store_true", help="overwrite an existing sample")
    args = ap.parse_args()
    rnd, force = args.round, args.force

    OUT.mkdir(exist_ok=True)
    f_lab = OUT / f"tag_check_r{rnd}.csv"
    f_c2  = OUT / f"tag_check_r{rnd}_coder2.csv"
    f_key = OUT / f"tag_check_r{rnd}_key.csv"
    if f_lab.exists() and not force:
        sys.exit(f"{f_lab.name} already exists (it may have your labels). "
                 f"Use --force only if you really want a new sample.")

    print(f"keyword list {KW_VERSION}: loading and tagging ...")
    df = tag(load_posts())

    counts = df.groupby(["year", "route"]).size().unstack(fill_value=0)
    counts["elec_total"] = df.groupby("year").elec.sum()
    counts.to_csv(OUT / "route_counts.csv")
    print(counts.to_string(), "\n")

    # never re-sample a post that was labeled in an earlier round
    seen = set()
    for k in OUT.glob("tag_check_r*_key.csv"):
        if k != f_key:
            seen |= set(pd.read_csv(k, dtype={"id": str}).id)
    df = df[~df.id.isin(seen)]

    parts, seed = [], 100 * rnd
    untag = df[~df.elec]
    if rnd == 1:
        for y in sorted(df.year.unique()):
            d = df[df.year == y]
            for route, n in [("general", 15), ("cand_only", 5), ("trial_only", 5)]:
                take(d[d.route == route], n, seed + y, "tagged", f"{route}_{y}", parts)
        take(untag, 50, seed + 1, "untagged_random", "untagged_random", parts)
        polit = untag[untag.clean.str.contains(POL, regex=True)]
        take(polit, 50, seed + 2, "untagged_political", "untagged_political", parts)
    else:
        for y in sorted(df.year.unique()):
            d = df[(df.year == y) & df.elec]
            take(d, 13, seed + y, "tagged", f"elec_{y}", parts)
        take(untag, 50, seed + 1, "untagged_random", "untagged_random", parts)

    samp = (pd.concat(parts).drop_duplicates("id")
              .sample(frac=1, random_state=seed + 9)      # shuffle so buckets are mixed
              .reset_index(drop=True))
    samp.insert(0, "row", range(1, len(samp) + 1))
    samp["snippet"]   = samp.text.map(snippet)
    samp["full_text"] = samp.text.str.replace(r"\s+", " ", regex=True).str[:1500]

    show = ["row", "id", "year", "state", "snippet", "full_text"]
    (samp[show].assign(is_election="", note="")
         .to_csv(f_lab, index=False, encoding="utf-8-sig"))
    (samp[show].head(N_CODER2).assign(is_election_2="", note_2="")
         .to_csv(f_c2, index=False, encoding="utf-8-sig"))
    (samp[["row", "id", "year", "route", "kind", "stratum", "stratum_size"]]
         .assign(kw_version=KW_VERSION).to_csv(f_key, index=False))

    print(samp.groupby("kind").size().to_string())
    print(f"\nwrote {f_lab.name} ({len(samp)} rows), {f_c2.name} ({N_CODER2} rows), {f_key.name}")


if __name__ == "__main__":
    main()