"""
Remove cross-catalogue duplicate sources, merging each confirmed group into one row.

- Groups come from DUPLICATES_report.csv (717 positional cross-catalogue groups).
- For each group: keep the most-complete row, then fill any null field from a twin
  (union of information). The catalogue label (col 0) becomes the combined provenance,
  e.g. "Lao+Mirabest".
- Writes a NEW file; the original is left untouched.

Run:  python deduplicate.py
"""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CATALOGS = REPO / "catalogs"
src = CATALOGS / "master_radio_catalog_clmatched - master_radio_catalog_xmatched_farhana.csv"
dup_csv = CATALOGS / "DUPLICATES_report.csv"
out = CATALOGS / "master_radio_catalog_deduplicated.csv"

df = pd.read_csv(src, low_memory=False)
cat_col = df.columns[0]          # the source-catalogue label column (header is ' x ')
dups = pd.read_csv(dup_csv)

groups = dups.groupby("dup_group")["row_index"].apply(list).to_dict()
all_dup_rows = sorted({i for idxs in groups.values() for i in idxs})

merged_rows = []
conflicts = []
for gid, idxs in groups.items():
    members = df.loc[idxs]
    # base = row with the most populated fields
    completeness = members.notna().sum(axis=1)
    base_idx = completeness.idxmax()
    merged = df.loc[base_idx].copy()

    # union: fill any null in base from a twin that has a value
    for col in df.columns:
        if pd.isna(merged[col]):
            for j in idxs:
                if j != base_idx and pd.notna(df.at[j, col]):
                    merged[col] = df.at[j, col]
                    break

    # combined provenance label
    labels = sorted({str(df.at[j, cat_col]).strip() for j in idxs})
    flat = sorted({p for lab in labels for p in lab.split("+")})
    merged[cat_col] = "+".join(flat)

    # flag conflicting redshifts (kept base value, but record it)
    zvals = members["z"].dropna().unique()
    if len(zvals) > 1 and (zvals.max() - zvals.min()) > 0.05:
        conflicts.append({"dup_group": gid, "kept_z": merged["z"],
                          "all_z": list(zvals), "label": merged[cat_col],
                          "RA": merged["RA"], "Dec": merged["Dec"]})

    merged["_orig_order"] = min(idxs)   # keep first appearance position
    merged_rows.append(merged)

merged_df = pd.DataFrame(merged_rows)

# untouched (non-duplicate) rows
keep = df.drop(index=all_dup_rows).copy()
keep["_orig_order"] = keep.index

final = pd.concat([keep, merged_df], ignore_index=True)
final = final.sort_values("_orig_order").drop(columns="_orig_order").reset_index(drop=True)
final.to_csv(out, index=False)

print(f"Original rows         : {len(df)}")
print(f"Duplicate rows removed : {len(all_dup_rows)} (in {len(groups)} groups)")
print(f"  -> merged into       : {len(merged_rows)} rows")
print(f"Final row count        : {len(final)}  ({len(df)} - {len(all_dup_rows)} + {len(merged_rows)})")
print(f"Conflicting-z groups (base z kept, review): {len(conflicts)}")
for c in conflicts:
    print(f"   group {c['dup_group']} [{c['label']}] kept z={c['kept_z']} from {c['all_z']}")
print(f"\nWritten: {out.name}")
