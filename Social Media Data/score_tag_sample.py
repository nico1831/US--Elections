"""
Phase 1: score a labeled hand-check sample.

Reads (phase1/):  tag_check_r<N>.csv          your labels (is_election)
                  tag_check_r<N>_key.csv      bucket of each row
                  tag_check_r<N>_coder2.csv   Schonn's labels (optional, for kappa)
Writes (phase1/): tag_scores_r<N>.txt         the printed report, for Ch. 3
                  tag_review_r<N>.csv         false positives + missed posts to read

Run:  python score_tag_sample.py --round 1
"""
import argparse, sys

import numpy as np
import pandas as pd

from reddit_keywords import OUT, USE_CAND_ONLY, USE_TRIAL

B = 2000                      # bootstrap draws for the weighted intervals
rng = np.random.default_rng(0)
LOG = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line); LOG.append(line)


def read_labels(path, col):
    """Reads a label file saved by Excel or Sheets; returns row -> 0/1."""
    for enc in ("utf-8-sig", "cp1252"):
        try:
            d = pd.read_csv(path, encoding=enc); break
        except UnicodeDecodeError:
            continue
    v = pd.to_numeric(d[col], errors="coerce")
    bad = d.loc[v.notna() & ~v.isin([0, 1]), "row"].tolist()
    if bad:
        sys.exit(f"{path.name}: {col} must be 1 or 0. Fix rows {bad}")
    return d.assign(**{col: v})[["row", col] + [c for c in ("snippet", "note") if c in d]]


def wilson(k, n, z=1.96):
    if n == 0:
        return "n=0"
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{p:6.1%}  [{c - h:.0%} to {c + h:.0%}]  n={n}"


def weighted(lab, keep):
    """Precision and recall for a set of routes, weighting each bucket by its real size.
    Returns point estimates plus 95% bootstrap intervals (resampling inside buckets)."""
    tagged = lab[lab.kind == "tagged"]
    groups = [(g.is_election.values, g.stratum_size.iloc[0], g.route.iloc[0] in keep)
              for _, g in tagged.groupby("stratum")]
    rnd = lab[lab.kind == "untagged_random"]
    miss_vals, n_untag = rnd.is_election.values, rnd.stratum_size.iloc[0]

    def calc(draw):
        tp = fp_dropped = n_keep = 0.0
        for vals, size, kept in groups:
            p = draw(vals).mean()
            if kept:
                tp += size * p; n_keep += size
            else:
                fp_dropped += size * p        # real election posts that a dropped route loses
        missed = n_untag * draw(miss_vals).mean() + fp_dropped
        return tp / n_keep, tp / (tp + missed)

    point = calc(lambda v: v)
    boots = np.array([calc(lambda v: rng.choice(v, len(v))) for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return point, lo, hi


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True, choices=[1, 2])
    rnd = ap.parse_args().round

    key = pd.read_csv(OUT / f"tag_check_r{rnd}_key.csv", dtype={"id": str})
    lab = read_labels(OUT / f"tag_check_r{rnd}.csv", "is_election").merge(key, on="row")
    todo = lab.loc[lab.is_election.isna(), "row"].tolist()
    if todo:
        sys.exit(f"{len(todo)} rows still unlabeled: {todo[:20]}{' ...' if len(todo) > 20 else ''}")

    say(f"ROUND {rnd}  |  keyword list {key.kw_version.iloc[0]}  |  {len(lab)} labeled posts\n")
    tagged = lab[lab.kind == "tagged"]

    say("PRECISION (share of tagged posts that really are election posts)")
    for r, g in tagged.groupby("route"):
        say(f"  route {r:<11}", wilson(g.is_election.sum(), len(g)))
    for y, g in tagged[tagged.route != "trial_only"].groupby("year"):
        say(f"  year  {y:<11}", wilson(g.is_election.sum(), len(g)))

    say("\nMISS RATE (share of random untagged posts that are election posts)")
    r_ = lab[lab.kind == "untagged_random"]
    say("  untagged random   ", wilson(r_.is_election.sum(), len(r_)))
    p_ = lab[lab.kind == "untagged_political"]
    if len(p_):
        say("  untagged political", wilson(p_.is_election.sum(), len(p_)), " (keyword hunting only)")

    say("\nWEIGHTED by real bucket sizes, 95% bootstrap interval")
    order   = ["general", "cand_only", "trial_only"]
    name_of = lambda keep: " + ".join(r for r in order if r in keep)
    current = {"general"} | ({"cand_only"} if USE_CAND_ONLY else set()) \
                          | ({"trial_only"} if USE_TRIAL else set())
    if rnd == 1:
        options = [set(order), {"general", "cand_only"}, {"general"}]
        if current not in options:
            options.append(current)
    else:
        lab.loc[lab.kind == "tagged", "route"] = "general"   # one bucket per year in round 2
        options = [{"general"}]
    res = {}
    for keep in options:
        (p, rc), lo, hi = weighted(lab, keep)
        name = name_of(keep) if rnd == 1 else "current list"
        res[name] = p
        say(f"  {name:<34} precision {p:6.1%} [{lo[0]:.0%} to {hi[0]:.0%}]"
            f"   recall {rc:6.1%} [{lo[1]:.0%} to {hi[1]:.0%}]")

    c2 = OUT / f"tag_check_r{rnd}_coder2.csv"
    k = np.nan
    if c2.exists():
        two = read_labels(c2, "is_election_2")[["row", "is_election_2"]].merge(
                lab[["row", "is_election"]], on="row").dropna()
        if len(two):
            k = kappa(two.is_election, two.is_election_2)
            say(f"\nAGREEMENT with coder 2: kappa {k:.2f} on {len(two)} posts, "
                f"{(two.is_election == two.is_election_2).mean():.0%} identical")
            diff = two.loc[two.is_election != two.is_election_2, "row"].tolist()
            if diff:
                say(f"  disagreements to discuss: rows {diff}")

    say("\nGATES")
    def gate(ok, text): say(f"  [{'PASS' if ok else 'CHECK'}] {text}")
    main_p = res[name_of(current)] if rnd == 1 else res["current list"]
    gate(main_p >= 0.85, f"weighted precision with current settings {main_p:.1%} (target 85%)")
    if rnd == 1:
        for r, t in [("cand_only", 0.80), ("trial_only", 0.70)]:
            g = tagged[tagged.route == r]
            gate(g.is_election.mean() >= t, f"{r} precision {g.is_election.mean():.0%} (target {t:.0%})")
    gate(r_.is_election.mean() <= 0.03, f"miss rate {r_.is_election.mean():.1%} (target 3% or less)")
    if not np.isnan(k):
        gate(k >= 0.70, f"kappa {k:.2f} (target 0.70)")

    rev = pd.concat([tagged[tagged.is_election == 0].assign(problem="false positive"),
                     lab[(lab.kind != "tagged") & (lab.is_election == 1)].assign(problem="missed")])
    rev[["row", "problem", "year", "route", "kind", "snippet", "note"]].to_csv(
        OUT / f"tag_review_r{rnd}.csv", index=False, encoding="utf-8-sig")
    (OUT / f"tag_scores_r{rnd}.txt").write_text("\n".join(LOG), encoding="utf-8")
    say(f"\nwrote tag_scores_r{rnd}.txt and tag_review_r{rnd}.csv ({len(rev)} posts to read)")


if __name__ == "__main__":
    main()