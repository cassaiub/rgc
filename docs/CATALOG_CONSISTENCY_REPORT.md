# Catalogue Consistency Report
**File:** `master_radio_catalog_clmatched - master_radio_catalog_xmatched_farhana.csv`
**Date:** 2026-06-12 | **Rows:** 7,025 | **Columns:** 62

---

## Headline finding — duplicate physical sources from multiple catalogues

The catalogue stacks four source lists (`Lao` 4,872, `Mirabest` 1,437, `Sasmal` 712, `Lao+Sasmal` 4) **without de-duplicating overlapping sources**. The same physical object frequently appears as two (or three) separate rows because it was detected in more than one parent catalogue.

A positional cross-match (spherical separation between rows of *different* catalogues) finds:

| Match radius | Cross-catalogue duplicate pairs |
|---|---|
| 2″ | **731** |
| 3″ | 764 |
| 5″ | 822 |

Grouping the 5″ matches into connected source-clusters:

- **717 duplicate source groups** covering **1,487 rows**
  - 665 groups of 2 rows, 51 groups of 3 rows, 1 group of 4 rows
- **≈770 rows are redundant** (could be merged away) — roughly **11% of the catalogue**.

**Confidence these are truly the same source, not chance alignments:**
- 626 of the duplicate pairs have a redshift in both rows; **98% agree** to within 1% (or Δz < 0.005).
- The pairs are tight: 731 of the 822 (89%) lie within 2″.

**Breakdown by catalogue pair (5″):**
- Lao ↔ Sasmal: 410
- Lao ↔ Mirabest: 356
- Mirabest ↔ Sasmal: 56

> Note on `FIRST_ID`: matching on `FIRST_ID` text **misses these duplicates**. 606 duplicate pairs carry a `FIRST_ID` in both rows, but only **5** match as strings — each catalogue rounded the J2000 position slightly differently (e.g. `J105545.0+452402` vs `J105545.1+452401`). De-duplication must be done **by sky position, not by ID string.**

The full list is exported to **`DUPLICATES_report.csv`** (columns: `dup_group`, `group_size`, `row_index`, `Catalog`, `FIRST_ID`, `RA`, `Dec`, `z`, `Flux_FIRST_1400MHz`, `Host_name`).

---

## Wrong / conflicting information

**5 duplicate groups give conflicting redshifts for what is the same source** (Δz > 0.05) — at least one value is wrong and must be adjudicated before merging:

| Catalogue A | z_A | Catalogue B | z_B | IDs |
|---|---|---|---|---|
| Sasmal | 0.53 | Lao | 0.679 | J103511.3+340624 / J103511.4+340619 |
| Lao | 0.174 | Mirabest | 0.361 | J101919.8+422737 |
| Sasmal | 2.01 | Lao | 2.234 | J105135.3+005133 / J105135.5+005131 |
| Sasmal | 0.92 | Lao | 0.052 | J085159.0+622757 / J085159.0+622801 |
| Sasmal | 0.39 | Lao | 0.463 | J123507.0+562505 / J123507.2+562508 |

(There are also 5 rows sharing an *identical* `FIRST_ID` string — the same 5 coincidental-rounding matches — where FIRST flux differs by up to 5× between the two rows, e.g. `J105545.0+452402`: 34.9 vs 154.7 mJy. These are the same duplicate sources seen above.)

---

## Missing information that can be filled

**Every one of the 717 duplicate groups has complementary data** — one member carries a value its twin lacks. Merging duplicates would fill gaps in (number of groups where a duplicate can supply a missing value):

| Column | Groups fillable |
|---|---|
| Flux_3000MHz | 709 |
| Host_name | 684 |
| alpha | 552 |
| Size_rad | 388 |
| Flux_NVSS_1400MHz | 359 |
| M_{BH} | 289 |
| Cluster_Name | 195 |
| Flux_150MHz | 189 |
| L_1.4GHz_log | 165 |
| z | 155 |

Independently of duplicates, **125 rows are missing `L_1.4GHz_log` but have both a 1.4 GHz flux and a redshift** — the luminosity can be computed directly.

---

## Other data-hygiene notes

- **Column 1 header is `' x '`** (literally "x"). It actually holds the source-catalogue label (Lao / Mirabest / Sasmal / Lao+Sasmal) and should be renamed `Catalog`.
- **20 rows have no morphology flag set** (all of FR0/FRI/FRII/WAT/NAT/HT/Confused/Double_Double/Diffuse are False) — unclassified.
- Coordinates, redshifts and fluxes are otherwise clean: RA ∈ [0,360), Dec ∈ [−11.3, 64.6], z ∈ [0.00026, 3.43], all fluxes positive, no out-of-range values, no exact full-row duplicates.

---

## Recommended next steps

1. **De-duplicate by position** (suggested 2–3″ radius): collapse each of the 717 groups to one row, taking the union of populated fields. This removes ~770 redundant rows.
2. **Adjudicate the 5 conflicting-z groups** by hand before merging.
3. Add a provenance column recording which parent catalogue(s) each merged source came from (the existing `Lao+Sasmal` label shows this was the intent).
4. Rename column 1 `x` → `Catalog`.
5. Optionally back-fill the 125 computable `L_1.4GHz_log` values.
