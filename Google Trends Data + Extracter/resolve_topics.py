#!/usr/bin/env python3
"""
resolve_topics.py — find the Google Trends Topic (entity) id for each term the
plan marks as form=TOPIC, and write topics.csv for you to review.

Why this exists: trendspy's interest_by_region takes a keyword STRING, so a term
marked TOPIC in the plan was fetched as a literal phrase. "united states
presidential election" as an exact string is rare — it returned 27 of 51 states
in 2012 and was screened out. The Topic entity (a /m/... id) aggregates every
phrasing of the concept and has far more coverage.

    python resolve_topics.py --plan trends_pull_urls.csv --out topics.csv

Prints every autocomplete candidate per term, writes the best guess to topics.csv,
then YOU check the `chosen_title`/`chosen_type` columns before fetching. An id is
opaque — a wrong one produces perfectly plausible numbers for the wrong concept,
which is the worst failure mode available here. Edit `chosen_mid` if the guess is
wrong; the alternatives are listed in the same row.
"""
import argparse, csv, sys, time

def score(cand, term):
    """Rank autocomplete candidates for `term`.

    The failure this guards against: Google returns narrow SUB-entities alongside
    the canonical one — "Hillary Clinton email controversy", "Presidency of Joe
    Biden", "Age and health concerns about Donald Trump". They are all legitimately
    typed "Topic", so rewarding the type string picks the wrong entity. What
    actually separates them is title length: a sub-entity's title is the query plus
    extra words. Canonical entities also tend to carry a /m/ id and a descriptive
    type ("46th U.S. President") rather than the bare word "Topic", and /g/ ids
    frequently return nothing from interest_by_region.
    """
    title = str(cand.get("title", "")).strip().lower()
    ctype = str(cand.get("type", "")).strip().lower()
    mid = str(cand.get("mid", ""))
    t = term.strip().lower()

    tw, qw = title.split(), t.split()
    s = 0
    if title == t:
        s += 20                                   # exact entity name
    elif t in title:
        s -= 3 * max(0, len(tw) - len(qw))        # sub-entity: query plus extras
    else:
        s -= 6                                    # not even a superset
    if mid.startswith("/m/"):
        s += 6                                    # canonical, and works in geo pulls
    if ctype and ctype != "search term":
        s += 4
    s += len(set(tw) & set(qw))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="trends_pull_urls.csv")
    ap.add_argument("--out", default="topics.csv")
    ap.add_argument("--sleep", type=float, default=4.0)
    args = ap.parse_args()

    try:
        from trendspy import Trends
    except ImportError:
        sys.exit("pip install trendspy")

    plan = list(csv.DictReader(open(args.plan, encoding="utf-8-sig")))
    wanted, seen = [], set()
    for r in plan:
        if r.get("form", "").upper() != "TOPIC":
            continue
        key = (r["term_id"], r["term"])
        if key not in seen:
            seen.add(key); wanted.append(key)
    if not wanted:
        sys.exit("no rows with form=TOPIC in the plan")

    tr = Trends()
    out = []
    for term_id, term in wanted:
        print(f"\n=== {term_id}  ('{term}')")
        try:
            df = tr.suggestions(term)
        except Exception as e:
            print(f"  lookup failed: {type(e).__name__}: {e}", file=sys.stderr)
            out.append({"term_id": term_id, "term": term, "chosen_mid": "",
                        "chosen_title": "", "chosen_type": "LOOKUP FAILED",
                        "alternatives": ""}); continue

        cands = df.to_dict("records") if hasattr(df, "to_dict") else list(df or [])
        if not cands:
            print("  no suggestions returned")
            out.append({"term_id": term_id, "term": term, "chosen_mid": "",
                        "chosen_title": "", "chosen_type": "NONE",
                        "alternatives": ""}); continue

        cands.sort(key=lambda c: -score(c, term))
        for i, c in enumerate(cands[:6]):
            mark = " <-- picked" if i == 0 else ""
            print(f"  {c.get('mid',''):<18} {str(c.get('title','')):<44} "
                  f"{c.get('type','')}{mark}")
        best = cands[0]
        alts = " | ".join(f"{c.get('mid','')}={c.get('title','')} ({c.get('type','')})"
                          for c in cands[1:5])
        out.append({"term_id": term_id, "term": term,
                    "chosen_mid": best.get("mid", ""),
                    "chosen_title": best.get("title", ""),
                    "chosen_type": best.get("type", ""),
                    "alternatives": alts})
        time.sleep(args.sleep)

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["term_id", "term", "chosen_mid",
                                          "chosen_title", "chosen_type", "alternatives"])
        w.writeheader(); w.writerows(out)

    print(f"\nwrote {args.out} ({len(out)} terms)")
    print("CHECK chosen_title / chosen_type before fetching. Then:")
    print("  1. delete the affected files from raw\\  (they hold the string version)")
    print("  2. python fetch_trends.py --plan trends_pull_urls.csv --out raw --topics topics.csv")


if __name__ == "__main__":
    main()
