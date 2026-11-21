"""
Phase 2: pull replies (comments) on every election post, with hashed authors.

Rules:
  2012, 2016  all replies on each election post
  2020, 2024  first 100 replies by time on each election post (Arctic Shift sorts
              comments by time only, so "first 100" is the cap rule)
  every year  only replies posted before election day (same cutoff as the posts)

Authors are hashed with the SAME salt and the SAME code as p1_pull_authors.py,
so a person's user_hash is identical across posts and replies. Usernames are
never saved.

Output: phase2/replies/<year>_<State>.csv
          id, post_id, parent_id, top_level, created_utc, body, removed, score,
          user_hash, author_status, state, year, subreddit
        phase2/pull_log.csv
        phase2/completeness.csv          (with --summary)

Run (PowerShell, from the Social Media Data folder):
        $env:REDDIT_SALT = "same phrase as Phase 1"
        python p2_pull_comments.py --test 20             # dry run on 20 posts, phase2/test/
        python p2_pull_comments.py --years 2012 2016     # about 6,000 calls
        python p2_pull_comments.py --years 2020 2024     # overnight, about 35,000 calls
        python p2_pull_comments.py --summary             # completeness table

Safe to stop and re-run: finished posts are remembered in .part/.done files.
"""
import argparse, random, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from reddit_keywords import HERE, RAW, KW_VERSION, load_posts, tag
from p1_pull_authors import salt_or_exit, classify      # same hashing as Phase 1

URL     = "https://arctic-shift.photon-reddit.com/api/comments/search"
OUT     = HERE / "phase2"
REPL    = OUT / "replies"
WORKERS = 2                     # parallel state-years; lower to 1 if you see many 429s
CAP     = {2012: None, 2016: None, 2020: 100, 2024: 100}     # None = all replies
CUTOFF  = {2012: "2012-11-06", 2016: "2016-11-08",            # election day, excluded
           2020: "2020-11-03", 2024: "2024-11-05"}
FIELDS  = "id,link_id,parent_id,created_utc,body,score,author"
COLS    = ["id", "post_id", "parent_id", "top_level", "created_utc", "body", "removed",
           "score", "user_hash", "author_status", "state", "year", "subreddit"]

# ---------------------------------------------------------------- shared rate-limit pause
_pause_until, _pause_lock = 0.0, threading.Lock()

def _wait_if_paused():
    while True:
        with _pause_lock:
            left = _pause_until - time.time()
        if left <= 0:
            return
        time.sleep(min(left, 5))

def _pause_all(seconds):
    global _pause_until
    with _pause_lock:
        _pause_until = max(_pause_until, time.time() + seconds)

_local = threading.local()
def _session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = "ADMU-thesis-turnout-research (state-level aggregates only)"
    return _local.s


def get_page(post_id, before, after=None):
    """Up to 100 replies on one post, oldest first, posted before `before`."""
    params = dict(link_id=post_id, before=before, sort="asc", limit=100, fields=FIELDS)
    if after is not None:
        params["after"] = after
    busy = 0
    for attempt in range(12):
        _wait_if_paused()
        try:
            r = _session().get(URL, params=params, timeout=60)
        except requests.RequestException as e:
            wait = min(60, 2 ** attempt)
            print(f"   post {post_id}: network error ({type(e).__name__}), retrying in {wait}s", flush=True)
            time.sleep(wait + random.random()); continue

        if r.status_code == 429:
            wait = int(float(r.headers.get("X-RateLimit-Reset", 30))) + 1
            print(f"   rate limited: all downloads paused {wait}s", flush=True)
            _pause_all(wait); continue

        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code in (408, 422) or r.status_code >= 500 or body.get("error"):
            busy += 1
            wait = min(60, 3 * 2 ** busy)
            print(f"   post {post_id}: archive busy ({r.status_code}), retry {busy} in {wait}s", flush=True)
            time.sleep(wait + random.random()); continue

        if r.status_code >= 400:
            r.raise_for_status()
        return body.get("data") or []
    raise RuntimeError(f"gave up on post {post_id}")


def fetch_post(post_id, year):
    """All replies (2012/2016) or the first CAP replies (2020/2024). Returns (comments, calls)."""
    cap, before = CAP[year], CUTOFF[year]
    got, seen, cur, calls = [], set(), None, 0
    while True:
        page = get_page(post_id, before, cur)
        calls += 1
        new = [c for c in page if c.get("id") not in seen]
        if not new:
            break
        got += new
        seen.update(c["id"] for c in new)
        if cap and len(got) >= cap:
            got = got[:cap]; break
        if len(page) < 100:
            break
        cur = max(c["created_utc"] for c in new)   # next page starts at the newest reply seen
        time.sleep(0.3)
    return got, calls


def to_rows(comments, salt, post_id, state, year, sub, cutoff_ts):
    rows = []
    for c in comments:
        t = c.get("created_utc") or 0
        if t >= cutoff_ts:                          # belt and braces: nothing on/after election day
            continue
        h, status = classify(c.get("author"), salt)
        body   = c.get("body") or ""
        parent = str(c.get("parent_id") or "")
        rows.append(dict(id=str(c["id"]).removeprefix("t1_"), post_id=post_id,
                         parent_id=parent, top_level=parent.startswith("t3_"),
                         created_utc=t, body=body, removed=body in ("[removed]", "[deleted]"),
                         score=c.get("score"), user_hash=h, author_status=status,
                         state=state, year=year, subreddit=sub))
    return rows


def pull_state_year(state, year, posts, salt, out_dir):
    """All replies for one state-year. Remembers finished posts so a re-run continues."""
    f = out_dir / f"{year}_{state.replace(' ', '_')}.csv"
    if f.exists():
        return state, year, None, 0
    part, done_f = f.with_suffix(".part"), f.with_suffix(".done")
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    cutoff_ts = pd.Timestamp(CUTOFF[year], tz="UTC").timestamp()
    calls = 0
    todo = posts[~posts.id.isin(done)]
    for i, p in enumerate(todo.itertuples(), 1):
        if p.num_comments > 0:
            comments, c = fetch_post(p.id, year)
            calls += c
            rows = to_rows(comments, salt, p.id, state, year, p.subreddit, cutoff_ts)
            if rows:
                pd.DataFrame(rows, columns=COLS).to_csv(part, mode="a", header=not part.exists(),
                                                        index=False)
        with open(done_f, "a") as fh:
            fh.write(p.id + "\n")
        if i % 200 == 0:
            print(f"   {year} {state}: {i}/{len(todo)} posts", flush=True)

    df = (pd.read_csv(part, dtype={"id": str, "post_id": str}) if part.exists()
          else pd.DataFrame(columns=COLS)).drop_duplicates("id")
    tmp = f.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(f)                                   # only finished files get the .csv name
    part.unlink(missing_ok=True)
    done_f.unlink(missing_ok=True)
    return state, year, len(df), calls


def election_posts():
    print(f"keyword list {KW_VERSION}: finding election posts ...")
    df = tag(load_posts())
    e = df[df.elec][["id", "year", "state", "subreddit", "created_utc", "num_comments"]].copy()
    e["num_comments"] = e.num_comments.fillna(0).astype(int)
    return e


def run_test(posts, salt, n):
    """Pulls n posts (half 2016, half 2024) into phase2/test/ and prints checks."""
    test_dir = OUT / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    pick = pd.concat([posts[(posts.year == y) & (posts.num_comments > 0)]
                        .sample(n // 2 + n % 2 * (y == 2016), random_state=1) for y in (2016, 2024)])
    rows = []
    for p in pick.itertuples():
        comments, _ = fetch_post(p.id, p.year)
        rows += to_rows(comments, salt, p.id, p.state, p.year, p.subreddit,
                        pd.Timestamp(CUTOFF[p.year], tz="UTC").timestamp())
    r = pd.DataFrame(rows, columns=COLS)
    r.to_csv(test_dir / "test_replies.csv", index=False)

    per = r.groupby("post_id").size().reindex(pick.id, fill_value=0)
    chk = pick.set_index("id").assign(fetched=per)
    print(chk[["year", "state", "num_comments", "fetched"]].to_string())
    print("\nmax replies on a 2024 post (cap 100):", int(chk[chk.year == 2024].fetched.max()))
    late = (r.created_utc >= r.year.map(lambda y: pd.Timestamp(CUTOFF[y], tz="UTC").timestamp())).sum()
    print("replies on/after election day:", int(late), "(should be 0)")
    print("top-level share:", round(r.top_level.mean(), 2), "| removed body share:", round(r.removed.mean(), 2))
    print(r.author_status.value_counts().to_string())

    a = pd.read_csv(RAW / "_authors.csv", dtype={"id": str}).set_index("id").user_hash
    op = chk.index.to_series().map(a)
    hit = sum(((r.post_id == pid) & (r.user_hash == h)).any() for pid, h in op.dropna().items())
    print(f"\nposts where the original poster replied in their own thread with the SAME user_hash: "
          f"{hit} of {op.notna().sum()}  (above 0 means the salt matches Phase 1)")
    print(f"\nwrote {test_dir / 'test_replies.csv'}")


def summary():
    """Completeness table: fetched vs num_comments, dates, duplicates, removed and deleted shares."""
    posts = election_posts()
    files = sorted(REPL.glob("20*.csv"))
    print(f"{len(files)} of 204 reply files found")
    r = pd.concat([pd.read_csv(f, dtype={"id": str, "post_id": str}) for f in files], ignore_index=True)
    a = pd.read_csv(RAW / "_authors.csv", dtype={"id": str})
    post_users = set(a.user_hash.dropna())

    per = r.groupby("post_id").size()
    posts["fetched"] = posts.id.map(per).fillna(0).astype(int)
    rows = []
    for y, p in posts.groupby("year"):
        d = r[r.year == y]
        cut = pd.Timestamp(CUTOFF[y], tz="UTC").timestamp()
        texts = (p.groupby("state").size().add(d.groupby("state").size(), fill_value=0))
        rows.append(dict(
            year=y, election_posts=len(p), posts_with_replies=int((p.fetched > 0).sum()),
            replies=len(d), num_comments_total=int(p.num_comments.sum()),
            fetched_share=round(len(d) / max(1, p.num_comments.sum()), 3),
            big_posts_with_0_fetched=int(((p.num_comments >= 10) & (p.fetched == 0)).sum()),
            on_or_after_election_day=int((d.created_utc >= cut).sum()),
            duplicate_ids=int(d.id.duplicated().sum()),
            removed_share=round(d.removed.mean(), 3),
            deleted_author_share=round((d.author_status == "deleted").mean(), 3),
            reply_users=d.user_hash.nunique(),
            reply_users_who_also_posted=round(d.user_hash.dropna().drop_duplicates().isin(post_users).mean(), 3),
            median_texts_per_state=texts.median(), states_under_10_texts=int((texts < 10).sum())))
    s = pd.DataFrame(rows).set_index("year")
    OUT.mkdir(exist_ok=True)
    s.to_csv(OUT / "completeness.csv")
    print(s.T.to_string())
    print(f"\nwrote {OUT / 'completeness.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2012, 2016, 2020, 2024])
    ap.add_argument("--test", type=int, metavar="N", help="dry run on N posts into phase2/test/")
    ap.add_argument("--summary", action="store_true", help="write phase2/completeness.csv and stop")
    args = ap.parse_args()
    if args.summary:
        return summary()

    salt = salt_or_exit()                       # refuses a salt different from Phase 1
    posts = election_posts()
    if args.test:
        return run_test(posts, salt, args.test)

    REPL.mkdir(parents=True, exist_ok=True)
    sel = posts[posts.year.isin(args.years)]
    jobs = [(s, y, g.sort_values("created_utc")) for (y, s), g in sel.groupby(["year", "state"])]
    print(f"{len(jobs)} state-years, {len(sel):,} election posts, "
          f"{int((sel.num_comments > 0).sum()):,} with replies\n")

    log, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(pull_state_year, s, y, g, salt, REPL): (s, y) for s, y, g in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            s, y = futs[fut]
            try:
                _, _, n, calls = fut.result()
                status = "skipped (exists)" if n is None else f"{n:,} replies, {calls} calls"
                log.append(dict(state=s, year=y, replies=n, calls=calls, error=""))
            except Exception as e:
                status = f"FAILED: {e}"
                log.append(dict(state=s, year=y, replies=None, calls=None, error=str(e)))
            print(f"[{i:3}/{len(jobs)}] {y} {s:<22} {status}   ({time.time() - t0:,.0f}s)", flush=True)

    lf = OUT / "pull_log.csv"
    pd.DataFrame(log).to_csv(lf, mode="a", header=not lf.exists(), index=False)
    failed = [r for r in log if r["error"]]
    print(f"\ndone in {time.time() - t0:,.0f}s. {len(failed)} failed"
          + (" (re-run the same command to retry them)" if failed else ""))


if __name__ == "__main__":
    main()