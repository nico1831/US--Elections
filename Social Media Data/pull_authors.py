"""
Phase 1: look up the author of every scraped post by post id and keep only a
salted hash of the username (a pseudonymous user id). Usernames are never saved.

Output: reddit_raw/_authors.csv          id, user_hash, author_status
        reddit_raw/_salt_fingerprint.txt  lets later scripts confirm the same salt

Run (PowerShell, from the Social Media Data folder):
        $env:REDDIT_SALT = "your-secret-phrase"
        python pull_authors.py

Keep the salt in a password manager. Phase 2 (replies) must use the SAME salt,
or post authors and reply authors will not match up.
Safe to stop and re-run: finished batches are kept in _authors.part.csv.
"""
import hashlib, os, random, sys, time

import pandas as pd
import requests

from reddit_keywords import RAW

URL   = "https://arctic-shift.photon-reddit.com/api/posts/ids"
BATCH = 500                       # API maximum per call
BOTS  = {"AutoModerator", "BotDefense", "RepostSleuthBot", "SaveVideo", "sneakpeekbot"}

FINAL = RAW / "_authors.csv"
PART  = RAW / "_authors.part.csv"
FPRT  = RAW / "_salt_fingerprint.txt"


def salt_or_exit():
    salt = os.environ.get("REDDIT_SALT", "")
    if not salt:
        sys.exit('Set the salt first:  $env:REDDIT_SALT = "your-secret-phrase"')
    fp = hashlib.sha256(("fingerprint:" + salt).encode()).hexdigest()[:12]
    if FPRT.exists() and FPRT.read_text().strip() != fp:
        sys.exit("REDDIT_SALT differs from the one used before (fingerprint mismatch). "
                 "Use the original salt.")
    FPRT.write_text(fp)
    return salt


def classify(author, salt):
    """Returns (user_hash, author_status). Only real users get a hash."""
    if author in (None, "", "[deleted]"):
        return None, "deleted"
    if author in BOTS or author.endswith("Bot") or author.lower().endswith(("_bot", "-bot")):
        return None, "bot"
    return hashlib.sha256((salt + author).encode()).hexdigest()[:16], "ok"


def fetch(session, batch):
    """One API call for up to 500 ids, with retries. Returns the list of posts."""
    busy = 0
    for attempt in range(12):
        try:
            r = session.get(URL, params={"ids": ",".join(batch), "fields": "id,author"},
                            timeout=60)
        except requests.RequestException as e:
            wait = min(60, 2 ** attempt)
            print(f"   network error ({type(e).__name__}), retrying in {wait}s", flush=True)
            time.sleep(wait + random.random()); continue

        if r.status_code == 429:
            wait = int(float(r.headers.get("X-RateLimit-Reset", 30))) + 1
            print(f"   rate limited, waiting {wait}s", flush=True)
            time.sleep(wait); continue

        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code in (408, 422) or r.status_code >= 500 or body.get("error"):
            busy += 1
            wait = min(60, 3 * 2 ** busy)
            print(f"   archive busy ({r.status_code}), retry {busy} in {wait}s", flush=True)
            time.sleep(wait + random.random()); continue

        r.raise_for_status()
        return body.get("data") or []
    raise RuntimeError(f"gave up on batch starting with id {batch[0]}")


def main():
    salt = salt_or_exit()

    files = sorted(RAW.glob("20*.csv"))
    ids = (pd.concat([pd.read_csv(f, usecols=["id"], dtype={"id": str}) for f in files])
             .id.unique().tolist())
    done = set(pd.read_csv(PART, dtype={"id": str}).id) if PART.exists() else set()
    todo = [i for i in ids if i not in done]
    print(f"{len(ids):,} post ids, {len(done):,} already done, {len(todo):,} to look up")

    s = requests.Session()
    s.headers["User-Agent"] = "ADMU-thesis-turnout-research (state-level aggregates only)"
    t0 = time.time()
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        found = {}
        for p in fetch(s, batch):
            pid = str(p.get("id", "")).removeprefix("t3_")
            found[pid] = p.get("author")

        rows = []
        for pid in batch:
            if pid in found:
                h, status = classify(found[pid], salt)
            else:
                h, status = None, "not_found"     # archive has no record of this id
            rows.append(dict(id=pid, user_hash=h, author_status=status))
        pd.DataFrame(rows).to_csv(PART, mode="a", header=not PART.exists(), index=False)

        n = start + len(batch)
        print(f"{n:,}/{len(todo):,}  ({time.time() - t0:,.0f}s)", flush=True)
        time.sleep(0.5)

    a = pd.read_csv(PART, dtype={"id": str}).drop_duplicates("id", keep="last")
    missing = set(ids) - set(a.id)
    assert not missing, f"{len(missing)} ids still missing, re-run the script"
    a.to_csv(FINAL, index=False)
    print(f"\nwrote {FINAL.name}: {len(a):,} rows")
    print(a.author_status.value_counts().to_string())
    print(f"\nYou can delete {PART.name} now.")


if __name__ == "__main__":
    main()