# Internal duplication & deduplication — DR1_FIRST_radio_classifications

**Date:** 2026-06-15
**Catalog:** `DR1_FIRST_radio_classifications.csv` (RGZ DR1, FIRST radio classifications)
**Rows:** 99,602
**Tool:** `RGC_2/scripts/crossmatch_catalogs.py` (added `--self` and `--dedup-out` modes)

---

## 1. Goal

Find and remove duplicate entries *within a single catalog* (rather than
cross-matching two different catalogs).

## 2. Method

`crossmatch_catalogs.py` was extended with a self-match mode:

- `--self` — matches the catalog against itself using A's RA/Dec columns.
  Excludes each row matching itself (the sep=0 diagonal); both directions of a
  real pair are kept so the UNIQUE/DUPLICATE split is correct for every source.
- `--dedup-out` — collapses groups of sources within `--radius` into one row
  using transitive grouping (union-find: A~B, B~C ⇒ one group) and keeps the
  highest-vote entry per group (`--vote-col`, default `N_votes`, tie-break
  `N_total`, then row order).

Command used:

```bash
python RGC_2/scripts/crossmatch_catalogs.py \
    --self --ra1 RA --dec1 Dec --radius 5 --dedup-out
```

## 3. Key finding — duplication is real, not chance

Duplicate-flagged rows vs. search radius:

| radius | rows flagged as duplicates |
|-------:|---------------------------:|
|   0″   |  9,870 |
|   1″   | 13,541 |
|   3″   | 13,678 |
|   5″   | 13,782 |
|  10″   | 13,874 |

The count is **nearly flat from 1″ to 10″** (+333 rows over a 100× increase in
search area). Chance alignments with unrelated field sources would scale with
area; instead the duplicates are **genuine repeat entries of the same physical
source**.

### What the duplicates are

- `CatID` is unique for all 99,602 rows → every row is a distinct entry.
- **4,903 sky positions have ≥2 rows at *exactly* the same RA/Dec** (9,870 rows:
  4,840 pairs, 62 triples, 1 quad).
- Duplicate rows share position/`CL`/`N_comp` but have **different
  `RGZID` / `ZooniverseID`** — i.e. the same FIRST source was served to Radio
  Galaxy Zoo volunteers as more than one Zooniverse subject.

Example exact-duplicate group:

```
CatID   RGZID                  ZooniverseID   RA        Dec       N_votes
75764   RGZ_J000130.2+014816   ARG0003q6j     0.37578   1.80451   22   <- kept
75781   RGZ_J000130.1+014816   ARG0003q6r     0.37578   1.80451   18
```

The FIRST beam is ~5″, so 3–5″ is the physically sensible dedup radius, and the
result is insensitive to the exact choice in that range.

## 4. Deduplication result (5″)

```
99,602 rows -> 92,632 unique sources (6,970 removed)
```

Reconciliation: 85,820 isolated sources + 6,812 group representatives = 92,632. ✓
Verified: highest-vote row kept per group; original column/row order preserved;
all kept `CatID`s unique.

## 5. Output files

| file | contents |
|------|----------|
| `DR1_FIRST_radio_classifications_dedup.csv`      | **clean catalog** — one row per source, highest-vote kept (92,632 rows) |
| `DR1_FIRST_radio_classifications_duplicates.csv` | diagnostic: sources with ≥1 neighbour within radius, with `n_match`, `sep_arcsec`, `match_*` |
| `DR1_FIRST_radio_classifications_unique.csv`     | diagnostic: sources with no neighbour within radius |

The three outputs are independent; use `_dedup.csv` as the working catalog and
ignore the other two unless you need the diagnostic split.
