"""Build trends_pull_candidates.csv — expanded candidate-attention terms.
Term list frozen 2026-09-22, before any of these were pulled."""
import csv
from urllib.parse import quote

WINDOWS = {2012: ("2012-08-08", "2012-11-06"),
           2016: ("2016-08-10", "2016-11-08"),
           2020: ("2020-08-05", "2020-11-03"),
           2024: ("2024-08-07", "2024-11-05")}

# term_id -> {year: search string}, form
CYCLE_VARYING = {
    "nominee_dem":      {2012: "barack obama", 2016: "hillary clinton",
                         2020: "joe biden",    2024: "kamala harris"},
    "nominee_rep":      {2012: "mitt romney",  2016: "donald trump",
                         2020: "donald trump", 2024: "donald trump"},
    "running_mate_dem": {2012: "joe biden",    2016: "tim kaine",
                         2020: "kamala harris", 2024: "tim walz"},
    "running_mate_rep": {2012: "paul ryan",    2016: "mike pence",
                         2020: "mike pence",   2024: "jd vance"},
}
FIXED = ["vice presidential debate", "presidential polls", "presidential candidates"]

rows = []
for year, (start, end) in WINDOWS.items():
    items = [(tid, terms[year], "TOPIC") for tid, terms in CYCLE_VARYING.items()]
    items += [(t.replace(" ", "_"), t, "term") for t in FIXED]
    for term_id, term, form in items:
        url = (f"https://trends.google.com/trends/explore?date={start}%20{end}"
               f"&geo=US&hl=en-US&q={quote(term)}")
        rows.append(dict(year=year, category="C4_candidates", term_id=term_id,
                         term=term, form=form, window_start=start, window_end=end,
                         url=url, save_as=f"{year}_C4_candidates_{term_id}.csv",
                         downloaded="", max_value="",
                         notes="added 2026-09-22, pre-registered expansion"))

for i, r in enumerate(rows, 1):
    r["order"] = i

cols = ["order", "year", "category", "term_id", "term", "form", "window_start",
        "window_end", "url", "save_as", "downloaded", "max_value", "notes"]
with open("trends_pull_candidates.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader(); w.writerows(rows)
print(f"wrote trends_pull_candidates.csv ({len(rows)} rows)")