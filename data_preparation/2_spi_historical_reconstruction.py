"""
Thesis Data Pipeline - Step 2
==============================
Build the SPI membership panel at entity level (RIC_BASE) by reverse-
replaying join/leave events from the current constituent list backwards
through time.  Also build a pull universe that preserves the original
case and suffix variants (RIC_FULL) needed for price downloads.

Inputs
------
- current2.csv : current SPI constituents (from LSEG Workspace)
- events2.csv  : historical join / leave events

Outputs
-------
- spi_membership_panel_by_month.csv : (date, RIC_BASE, Name) — one row
      per stock x month-end for every month the stock was in the SPI
- spi_ric_pull_universe.csv         : (RIC_FULL, RIC_BASE, RIC_SUFFIX,
      Issuer, Date, first/last seen) — every RIC variant ever observed
"""

import re
import pandas as pd

# ============================================================
# Helpers
# ============================================================

def normalize_spaces(x: str) -> str:
    """Collapse runs of whitespace to a single space."""
    return re.sub(r"\s+", " ", str(x)).strip()


def clean_expired_tag(s: str) -> str:
    """Strip the '(EXPIRED)' label that LSEG appends to delisted RICs."""
    return re.sub(r"\s*\(EXPIRED\)\s*", "", s, flags=re.IGNORECASE).strip()


def extract_base_ric_case_preserving(s: str) -> str:
    """Extract a Swiss equity RIC token like 'ABBN.S' from a string.
    Allows lowercase letters because some LSEG exports contain them."""
    m = re.search(r"\b([A-Za-z0-9\.]+\.[Ss])\b", s)
    if m:
        return m.group(1)
    toks = s.split()
    return toks[0] if toks else None


def parse_code_to_full_base_suffix(x):
    """Split a raw RIC code into three components:

    RIC_FULL   – case-preserving, cleaned of '(EXPIRED)', keeps suffix
    RIC_BASE   – uppercased base identifier (stable key for joins/leaves)
    RIC_SUFFIX – e.g. '^F23' if present, else None
    """
    if pd.isna(x):
        return (None, None, None)

    raw = normalize_spaces(str(x))
    s = clean_expired_tag(raw)

    m_suf = re.search(r"(\^[A-Za-z0-9]+)$", s)
    suffix = m_suf.group(1) if m_suf else None

    base_part = re.sub(r"\^.*$", "", s).strip()
    base_token = extract_base_ric_case_preserving(base_part)
    if base_token is None:
        return (None, None, None)

    ric_full = base_token + (suffix if suffix else "")
    ric_base = base_token.upper()
    return (ric_full, ric_base, suffix)


# ============================================================
# 1  Load & normalise current constituents
# ============================================================
cur = pd.read_csv("current2.csv")

RIC_COL = "Identifier (RIC)"
NAME_COL = "Company Name"

cur = cur[cur[RIC_COL].notna()].copy()
cur = cur[~cur[RIC_COL].astype(str).str.contains(
    "Totals", case=False, na=False, regex=False
)].copy()

cur["RIC_FULL_RAW"] = cur[RIC_COL].astype(str).apply(normalize_spaces)
cur[["RIC_FULL", "RIC_BASE", "RIC_SUFFIX"]] = cur["RIC_FULL_RAW"].apply(
    lambda x: pd.Series(parse_code_to_full_base_suffix(x))
)
cur[NAME_COL] = cur[NAME_COL].astype(str).apply(normalize_spaces)

current_set_base = set(cur["RIC_BASE"].dropna().unique())
name_map_current = (
    cur.dropna(subset=["RIC_BASE"]).set_index("RIC_BASE")[NAME_COL].to_dict()
)

# ============================================================
# 2  Load & normalise join/leave events
# ============================================================
ev = pd.read_csv("events2.csv")
ev["Date"] = pd.to_datetime(ev["Date"], dayfirst=True, errors="coerce")

ev[["RIC_FULL", "RIC_BASE", "RIC_SUFFIX"]] = ev["Code"].apply(
    lambda x: pd.Series(parse_code_to_full_base_suffix(x))
)

ev["Issuer"] = (
    ev["Issuer"].astype(str).apply(normalize_spaces)
    if "Issuer" in ev.columns
    else None
)
ev["Status"] = ev["Status"].astype(str).str.strip().str.lower()

ev = ev[
    ev["Date"].notna()
    & ev["RIC_BASE"].notna()
    & ev["Status"].isin(["joiner", "leaver"])
].copy()

name_map_events = (
    ev.dropna(subset=["RIC_BASE"]).set_index("RIC_BASE")["Issuer"].to_dict()
)

# ============================================================
# 3  Build pull universe (preserves case & suffixes)
# ============================================================
# Combines every RIC variant seen in either the current constituents
# or the event history, so the price download step can request all of
# them and the best one can be selected later.

pull_universe = pd.concat(
    [
        ev[["RIC_FULL", "RIC_BASE", "RIC_SUFFIX", "Date", "Issuer"]].copy(),
        cur[["RIC_FULL", "RIC_BASE", "RIC_SUFFIX"]].assign(
            Date=pd.NaT, Issuer=cur[NAME_COL].values
        ),
    ],
    ignore_index=True,
)
pull_universe = (
    pull_universe
    .dropna(subset=["RIC_FULL", "RIC_BASE"])
    .drop_duplicates(["RIC_FULL", "RIC_BASE"])
)

# Attach first/last event date per RIC_FULL for audit purposes
date_range = (
    ev.dropna(subset=["RIC_FULL", "Date"])
    .groupby("RIC_FULL")["Date"]
    .agg(["min", "max"])
    .reset_index()
)
pull_universe = pull_universe.merge(date_range, on="RIC_FULL", how="left")

pull_universe = pull_universe.sort_values(["RIC_BASE", "RIC_FULL"]).reset_index(drop=True)
pull_universe.to_csv("spi_ric_pull_universe.csv", index=False)
print(f"Wrote spi_ric_pull_universe.csv — {len(pull_universe)} unique pull RICs")

# ============================================================
# 4  Reconstruct monthly membership by reverse-replaying events
# ============================================================
# Starting from the known current constituent set, we walk backwards
# through time.  At each month a "joiner" is removed (it wasn't in
# the index yet) and a "leaver" is added back (it was still in).
# This correctly handles months with no events — the set simply
# carries over unchanged.

ev["MonthEnd"] = ev["Date"].dt.to_period("M").dt.to_timestamp("M")

first_event_me = ev["MonthEnd"].min()
if pd.isna(first_event_me):
    raise ValueError("No valid events found — cannot reconstruct membership history.")

GUARD_START = pd.Timestamp("1998-01-31")
start_me = max(first_event_me, GUARD_START)
end_me = pd.Timestamp.today().to_period("M").to_timestamp("M")

all_month_ends = pd.date_range(start=start_me, end=end_me, freq="ME")
months_desc = list(all_month_ends[::-1])

members = set(current_set_base)
snapshots = []

for me in months_desc:
    ev_m = ev[ev["MonthEnd"] == me].sort_values("Date", ascending=False)
    for _, row in ev_m.iterrows():
        b = row["RIC_BASE"]
        if row["Status"] == "joiner":
            members.discard(b)
        elif row["Status"] == "leaver":
            members.add(b)
    snapshots.append({"date": me, "RIC_BASEs": members.copy()})

# Append an explicit snapshot for the current month from current2.csv
# (usually duplicates the last month-end; deduped below)
today_me = pd.Timestamp.today().to_period("M").to_timestamp("M")
snapshots.append({"date": today_me, "RIC_BASEs": set(current_set_base)})

# ============================================================
# 5  Expand snapshots to a long panel and attach names
# ============================================================
records = [
    (snap["date"], b)
    for snap in snapshots
    for b in snap["RIC_BASEs"]
]
panel = (
    pd.DataFrame(records, columns=["date", "RIC_BASE"])
    .drop_duplicates()
    .sort_values(["date", "RIC_BASE"])
    .reset_index(drop=True)
)


def resolve_name(ric_base):
    """Prefer the current constituent name; fall back to event history."""
    return name_map_current.get(ric_base, name_map_events.get(ric_base))


panel["Name"] = panel["RIC_BASE"].apply(resolve_name)

# LSEG Workspace tracking begins 27 Jan 1998, making January a partial
# initialisation month.  Start from February to use only complete snapshots.
panel = panel[panel["date"] >= "1998-02-01"].copy()

panel.to_csv("spi_membership_panel_by_month.csv", index=False)
print(f"Wrote spi_membership_panel_by_month.csv — {len(panel)} rows")
print(panel.head(), "\n", panel.tail())

# ============================================================
# 6  Validation: forward replay must recover current constituents
# ============================================================
# Independent consistency check: seed the oldest panel month, then
# replay every event forward in time.  The resulting set must match
# the known current constituents exactly.

oldest_dt = panel["date"].min()
seed = set(panel.loc[panel["date"] == oldest_dt, "RIC_BASE"])

members_fwd = set(seed)
for _, row in ev.sort_values("Date").iterrows():
    b = row["RIC_BASE"]
    if row["Status"] == "joiner":
        members_fwd.add(b)
    elif row["Status"] == "leaver":
        members_fwd.discard(b)

delta = len(members_fwd ^ current_set_base)
print(f"Forward validation matches current set? {members_fwd == current_set_base} "
      f"(|Δ| = {delta})")

# ============================================================
# 7  Diagnostics
# ============================================================

# Check for gaps: month-ends present in the date range but missing from panel
dates = panel["date"].drop_duplicates().sort_values()
expected = pd.date_range(dates.min(), dates.max(), freq="ME")
missing = expected.difference(dates)
print(f"Missing month-ends in panel: {len(missing)}")
if len(missing) > 0:
    print("First missing:", missing[:10].to_list())

# Check for unexpected lowercase in RIC_FULL (informational, not an error)
lc_mask = pull_universe["RIC_FULL"].astype(str).str.contains(r"[a-z]", regex=True, na=False)
print(f"RIC_FULL entries with lowercase letters: {int(lc_mask.sum())}")
if lc_mask.any():
    print("Examples:", pull_universe.loc[lc_mask, "RIC_FULL"].head(15).to_list())