#!/usr/bin/env python3
"""
fetch_trends.py — pull one Google Trends "Interest by subregion" file per term/year.

Reads trends_pull_urls.csv (68 rows) and writes one CSV per row into raw/,
in the same shape as a manual UI download, so build_features.py cannot tell
the difference between scripted and hand-downloaded files.

    pip install trendspy pandas
    python fetch_trends.py --plan trends_pull_urls.csv --out raw

Resumable: files already present in raw/ are skipped, so re-running after a
rate-limit stop picks up where it left off. If trendspy breaks (it reverse-
engineers a private endpoint and Google changes it periodically), fall back to
the URLs in the plan CSV and download by hand — the loader accepts both.
"""
import argparse, csv, random, sys, time
from pathlib import Path

STATES = None  # populated from the first successful pull


META_COLS = {"geoName", "geoCode", "lat", "lng", "coordinates"}


def fetch_one(tr, term, timeframe):
    """Return [(state_name, value)].

    trendspy's interest_by_region returns a RangeIndex-ed DataFrame whose columns
    are geoName / geoCode / lat / lng plus one column per keyword — the state name
    lives in the geoName COLUMN, not the index. inc_low_vol=True keeps states that
    Google flags as low volume, which otherwise silently vanish from the result.
    """
    df = tr.interest_by_region(term, geo="US", resolution="REGION",
                               timeframe=timeframe, inc_low_vol=True)
    if df is None or len(df) == 0:
        return []
    if "geoName" not in df.columns:
        raise RuntimeError(f"no geoName column; got {list(df.columns)}")
    vals = [c for c in df.columns if c not in META_COLS]
    if not vals:
        raise RuntimeError(f"no value column; got {list(df.columns)}")
    out = []
    for g, v in zip(df["geoName"], df[vals[-1]]):
        try:
            out.append((str(g), int(round(float(v)))))
        except (TypeError, ValueError):
            out.append((str(g), 0))
    return out


def write_ui_shaped(path, term, win_start, win_end, rows):
    """Write the same layout the Trends UI export produces."""
    def us(d):  # 2012-08-08 -> 8/8/12
        y, m, dd = d.split("-")
        return f"{int(m)}/{int(dd)}/{y[2:]}"
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("Category: All categories\n\n")
        w = csv.writer(f)
        w.writerow(["Region", f"{term}: ({us(win_start)} - {us(win_end)})"])
        for region, val in rows:
            w.writerow([region, val])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="trends_pull_urls.csv")
    ap.add_argument("--out", default="raw")
    ap.add_argument("--sleep", type=float, default=12.0,
                    help="base seconds between requests; jittered ±40%%")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new pulls")
    ap.add_argument("--topics", default="",
                    help="topics.csv from resolve_topics.py — substitutes the Topic "
                         "id for the search string on matching term_ids")
    ap.add_argument("--selftest", action="store_true",
                    help="pull one term for 2012 and 2024 and check the timeframe "
                         "argument is actually being honoured")
    args = ap.parse_args()

    try:
        from trendspy import Trends
    except ImportError:
        sys.exit("pip install trendspy")

    if args.selftest:
        tr = Trends()
        a = dict(fetch_one(tr, "polling place", "2012-08-08 2012-11-06"))
        time.sleep(8)
        b = dict(fetch_one(tr, "polling place", "2024-08-07 2024-11-05"))
        common = sorted(set(a) & set(b))
        if len(common) < 10:
            sys.exit("selftest inconclusive — too few overlapping regions")
        xa = [a[k] for k in common]; xb = [b[k] for k in common]
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0] * len(v)
            for pos, i in enumerate(order): r[i] = pos
            return r
        ra, rb = rank(xa), rank(xb)
        n = len(common)
        rho = 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))
        ident = sum(1 for k in common if a[k] == b[k]) / n
        print(f"selftest: 'polling place' 2012 vs 2024 over {n} regions")
        print(f"  Spearman rho = {rho:.3f}   identical values = {ident:.0%}")
        print(f"  top 2012: {sorted(a, key=a.get, reverse=True)[:5]}")
        print(f"  top 2024: {sorted(b, key=b.get, reverse=True)[:5]}")
        if not any(c.isalpha() for c in "".join(common[:5])):
            print("\n  FAIL — region names are not state names. Parsing bug, not a "
                  "timeframe problem; do not trust any file already written.")
        elif ident > 0.90:
            print("\n  FAIL — 90%+ of states have identical values in both windows.\n"
                  "  `timeframe` is being ignored. Download by hand from the plan URLs.")
        elif rho > 0.995:
            print("\n  WARN — rankings almost identical (values differ). Plausible for a\n"
                  "  stable term, but eyeball the two top-5 lists before trusting it.")
        else:
            print("\n  PASS — the windows differ, so `timeframe` is being applied.")
        return

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    plan = list(csv.DictReader(open(args.plan, encoding="utf-8-sig")))

    topics = {}
    if args.topics:
        for r in csv.DictReader(open(args.topics, encoding="utf-8-sig")):
            mid = (r.get("chosen_mid") or "").strip()
            if mid:
                # keyed by the SEARCH STRING: one term_id (e.g. nominee_dem) covers a
                # different person each cycle, so term_id is not a unique key here.
                topics[r["term"].strip().lower()] = (mid, r.get("chosen_title", ""))
        dupes = {}
        for r in csv.DictReader(open(args.topics, encoding="utf-8-sig")):
            dupes.setdefault((r.get("chosen_mid") or "").strip(), []).append(r["term"])
        print(f"topic overrides loaded for {len(topics)} search terms:")
        for term, (mid, title) in topics.items():
            warn = ""
            if len(dupes.get(mid, [])) > 1:
                warn = f"   !! same id as: {', '.join(x for x in dupes[mid] if x.lower() != term)}"
            print(f"  {term:<38} -> {mid:<20} ({title}){warn}")
        if any(len(v) > 1 for k, v in dupes.items() if k):
            print("\n  WARNING: one Topic id is mapped to several different search terms.\n"
                  "  That is almost always wrong — fix chosen_mid in topics.csv first.")
        print()

    tr = Trends()

    done = failed = 0
    for row in plan:
        dest = outdir / row["save_as"]
        if dest.exists():
            continue
        if args.limit and done >= args.limit:
            print(f"--limit {args.limit} reached"); break

        tf = f"{row['window_start']} {row['window_end']}"
        query = row["term"]
        label = f"{row['year']} {row['term']}"
        key = row["term"].strip().lower()
        if key in topics:
            query = topics[key][0]
            label += f"  [topic {query}]"
        try:
            rows = fetch_one(tr, query, tf)
            if not rows and query != row["term"]:
                # a Topic id that returns nothing (common for /g/ ids) — fall back
                print(f"  retry  {label}: topic empty, retrying as plain search term")
                time.sleep(3)
                rows = fetch_one(tr, row["term"], tf)
                if rows:
                    label += "  [FELL BACK to search term]"
            if not rows:
                print(f"  EMPTY  {label}"); failed += 1
            else:
                write_ui_shaped(dest, row["term"], row["window_start"],
                                row["window_end"], rows)
                mx = max(v for _, v in rows)
                flags = []
                if mx < 15:        flags.append("low resolution")
                if len(rows) < 45: flags.append(f"only {len(rows)} regions")
                flag = ("  <-- " + ", ".join(flags)) if flags else ""
                print(f"  ok     {label}  ({len(rows)} regions, max={mx}){flag}")
                done += 1
        except Exception as e:
            print(f"  FAIL   {label}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
        time.sleep(args.sleep * random.uniform(0.6, 1.4))

    print(f"\n{done} fetched, {failed} failed, "
          f"{sum(1 for r in plan if (outdir / r['save_as']).exists())}/{len(plan)} present")
    if failed:
        print("Re-run to retry failures; download stubborn ones by hand from the plan URLs.")


if __name__ == "__main__":
    main()