"""
Pull every post from the 51 state subreddits for the 4 pre-election windows
using the Arctic Shift Reddit archive API.

Output: reddit_raw/<year>_<State>.csv   (one file per state-year, 204 files)
        reddit_raw/_log.csv             (posts and requests per file)

Run:    pip install requests pandas
        python pull_reddit.py

Safe to stop and re-run: finished files are skipped.
"""
import time, random, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

# ---------------------------------------------------------------- settings
OUT      = Path("reddit_raw")
WORKERS  = 2            # parallel downloads; lower to 2 if you see many 429s
BASE     = "https://arctic-shift.photon-reddit.com/api/posts/search"
KEEP     = ["id", "created_utc", "title", "selftext", "score", "num_comments"]  # no usernames

# 90 days before election day; 'before' is exclusive, so election day itself is left out
WINDOWS = {2012: ("2012-08-08", "2012-11-06"),
           2016: ("2016-08-10", "2016-11-08"),
           2020: ("2020-08-05", "2020-11-03"),
           2024: ("2024-08-07", "2024-11-05")}

SUBS = {"Alabama": "alabama", "Alaska": "alaska", "Arizona": "arizona", "Arkansas": "arkansas",
        "California": "California", "Colorado": "Colorado", "Connecticut": "Connecticut",
        "Delaware": "Delaware", "District of Columbia": "washingtondc", "Florida": "florida",
        "Georgia": "Georgia", "Hawaii": "Hawaii", "Idaho": "Idaho", "Illinois": "illinois",
        "Indiana": "Indiana", "Iowa": "Iowa", "Kansas": "kansas", "Kentucky": "Kentucky",
        "Louisiana": "Louisiana", "Maine": "Maine", "Maryland": "maryland",
        "Massachusetts": "massachusetts", "Michigan": "Michigan", "Minnesota": "minnesota",
        "Mississippi": "mississippi", "Missouri": "missouri", "Montana": "Montana",
        "Nebraska": "Nebraska", "Nevada": "Nevada", "New Hampshire": "newhampshire",
        "New Jersey": "newjersey", "New Mexico": "NewMexico", "New York": "newyork",
        "North Carolina": "NorthCarolina", "North Dakota": "northdakota", "Ohio": "Ohio",
        "Oklahoma": "oklahoma", "Oregon": "oregon", "Pennsylvania": "Pennsylvania",
        "Rhode Island": "RhodeIsland", "South Carolina": "southcarolina",
        "South Dakota": "SouthDakota", "Tennessee": "Tennessee", "Texas": "texas", "Utah": "Utah",
        "Vermont": "vermont", "Virginia": "Virginia", "Washington": "Washington",
        "West Virginia": "WestVirginia", "Wisconsin": "wisconsin", "Wyoming": "wyoming"}

# ---------------------------------------------------------------- shared rate-limit pause
# When any thread gets a 429, every thread waits until the server's reset time.
_pause_until = 0.0
_pause_lock  = threading.Lock()

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

_no_auto = set()        # subreddits where limit=auto was refused; they use 100 per page

_local = threading.local()
def _session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = "ADMU-thesis-turnout-research (state-level aggregates only)"
    return _local.s

def get_page(sub, after, before, big=False):
    """Returns up to 100 posts per call (only the columns we need).
    Prints whenever it has to wait, so a pause never looks like a freeze."""
    params = dict(subreddit=sub, after=after, before=before, sort="asc",
                  limit=100, fields=",".join(KEEP))
    busy = 0
    for attempt in range(12):
        _wait_if_paused()
        try:
            r = _session().get(BASE, params=params, timeout=60)
        except requests.RequestException as e:
            wait = min(60, 2 ** attempt)
            print(f"   r/{sub}: network error ({type(e).__name__}), retrying in {wait}s", flush=True)
            time.sleep(wait + random.random()); continue

        if r.status_code == 429:
            wait = int(r.headers.get("X-RateLimit-Reset", 30)) + 1
            print(f"   rate limited: all downloads paused {wait}s", flush=True)
            _pause_all(wait); continue

        try:
            body = r.json()
        except ValueError:
            body = {}

        if r.status_code in (422, 408) or r.status_code >= 500 or body.get("error"):
            busy += 1
            wait = min(60, 3 * 2 ** busy)
            print(f"   r/{sub}: archive busy ({r.status_code}), retry {busy} in {wait}s", flush=True)
            time.sleep(wait + random.random()); continue

        if r.status_code >= 400:
            r.raise_for_status()
        return body.get("data") or []
    raise RuntimeError(f"gave up on r/{sub} after={after}")

# ---------------------------------------------------------------- one state-year
def pull_state_year(state, year):
    sub = SUBS[state]
    after, before = WINDOWS[year]
    f = OUT / f"{year}_{state.replace(' ', '_')}.csv"
    if f.exists():
        return state, year, None, 0          # already done

    rows, seen, cur, calls = [], set(), after, 0
    while True:
        page = get_page(sub, cur, before)
        calls += 1
        new = [p for p in page if p["id"] not in seen]
        if not new:
            break
        rows += new
        seen.update(p["id"] for p in new)
        cur = max(p["created_utc"] for p in new)   # next call starts at the newest post seen
        time.sleep(0.3)

    df = pd.DataFrame(rows).reindex(columns=KEEP)
    df["state"], df["year"], df["subreddit"] = state, year, sub
    tmp = f.with_suffix(".part")
    df.to_csv(tmp, index=False)
    tmp.replace(f)                                  # only finished files get the .csv name
    return state, year, len(df), calls

# ---------------------------------------------------------------- run everything
def main():
    OUT.mkdir(exist_ok=True)
    jobs = [(s, y) for y in WINDOWS for s in SUBS]
    log, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(pull_state_year, s, y): (s, y) for s, y in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            s, y = futs[fut]
            try:
                _, _, n, calls = fut.result()
                status = "skipped (exists)" if n is None else f"{n:,} posts, {calls} calls"
                log.append(dict(state=s, year=y, posts=n, calls=calls, error=""))
            except Exception as e:
                status = f"FAILED: {e}"
                log.append(dict(state=s, year=y, posts=None, calls=None, error=str(e)))
            print(f"[{i:3}/{len(jobs)}] {y} {s:<22} {status}   ({time.time()-t0:,.0f}s)")
    pd.DataFrame(log).to_csv(OUT / "_log.csv", index=False)
    failed = [r for r in log if r["error"]]
    print(f"\ndone in {time.time()-t0:,.0f}s. {len(failed)} failed"
          + (" (re-run the script to retry them)" if failed else ""))

if __name__ == "__main__":
    main()