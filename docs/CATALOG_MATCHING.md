# Cross-Match Catalogs: DESI Legacy & AllWISE

This document describes the two external catalogs your radio sources are matched
against, the columns pulled from each, and what each column tells you.

Both scripts take the same input catalog and, for every radio source `(RA, Dec)`,
run a cone search against an external survey and keep the nearest counterpart.

| | `match_desi_legacy.py` | `match_awise_catalog.py` |
|---|---|---|
| **External catalog** | DESI Legacy Imaging Surveys — Tractor + photo-z | AllWISE Source Catalog |
| **Table(s)** | `ls_dr9.tractor` ⋈ `ls_dr9.photo_z` (or `dr10`) | `allwise_p3as_psd` |
| **Access** | NOIRLab Astro Data Lab (`dl.queryClient`, q3c SQL) | IRSA (`astroquery.ipac.irsa`) |
| **Default radius** | 3″ | 6″ |
| **Bands** | optical *g,r,(i),z* + IR *W1,W2* | IR *W1–W4* + 2MASS *J,H,K* |
| **Redshift?** | **Yes** — photo-z + spec-z | **No** (carries input `z` through) |
| **Input catalog** | `rgc/catalogs/master_radio_catalog_deduplicate_removed.csv` | same |
| **Output** | `rgc/catalogs/master_radio_catalog_desi_match.csv` | `rgc/catalogs/master_radio_catalog_awise_match.csv` |

Shared input columns used: `RA`, `Dec`, `widx` (source ID), `z` (redshift, optional).
Both scripts are **resumable** (skip `widx` already in output) and write a row with
`n_match=0` for non-matches so they aren't retried; query errors go to `<out>.errors.csv`.

---

## 1. DESI Legacy Imaging Surveys — `match_desi_legacy.py`

**What it is:** The DESI Legacy Imaging Surveys (DR9/DR10) provide deep optical
imaging (DECaLS, BASS, MzLS) with forced WISE photometry. The **Tractor** catalog
holds the model-based photometry; the **photo_z** catalog (Zhou et al. 2021) holds
photometric redshifts. They are joined on `ls_id`.

**Query:** q3c radial cone search of `ls_drX.tractor`, `LEFT JOIN` photo_z on `ls_id`,
keeping the nearest source within radius.

### Columns pulled from the Tractor catalog (`t.`)

| Column | Meaning |
|---|---|
| `ls_id` | Unique Legacy Survey object identifier (join key to photo_z) |
| `ra`, `dec` | Object position (deg) |
| `type` | Morphological model: `PSF` (point source), `REX` (round exp), `EXP` (exponential/disk), `DEV` (de Vaucouleurs/bulge), `SER` (Sérsic) |
| `maskbits` | Bitmask of imaging issues (near bright star, bad pixels, etc.) — use to reject contaminated sources |
| `flux_g, flux_r, [flux_i,] flux_z` | Optical model flux densities, **nanomaggies** |
| `flux_w1, flux_w2` | WISE forced-photometry flux densities, nanomaggies |
| `dered_flux_g, dered_flux_r, [dered_flux_i,] dered_flux_z` | Galactic-extinction-corrected optical fluxes |

> `i`-band only exists in **DR10**. DR9 bands are `g, r, z`.

### Columns pulled from the photo_z catalog (`p.`)

| Column | Meaning |
|---|---|
| `z_phot_mean` | Photometric redshift, mean of the PDF |
| `z_phot_median` | Photometric redshift, median of the PDF |
| `z_phot_std` | Standard deviation of the photo-z PDF (uncertainty) |
| `z_phot_l68`, `z_phot_u68` | Lower / upper bounds of the 68% confidence interval |
| `z_spec` | Spectroscopic redshift if the object has one (SDSS/DESI), else NaN |

### Derived / written columns in the output CSV

| Output column | How it's computed |
|---|---|
| `widx`, `in_ra`, `in_dec`, `z_cat` | Carried from the input radio catalog (`z_cat` = input `z`) |
| `n_match` | Number of Tractor sources found within the radius |
| `ls_id`, `desi_ra`, `desi_dec` | From the matched Tractor source |
| `sep_arcsec` | Angular separation radio↔DESI counterpart (arcsec) |
| `type`, `maskbits` | From Tractor (see above) |
| `mag_{band}` | AB magnitude = `22.5 − 2.5·log10(flux_nmgy)` for each band (g,r,[i],z,w1,w2) |
| `f_{band}_mjy` | Flux density in mJy = `flux_nmgy × 3.631e-3` |
| `mag_{band}_dered` | Extinction-corrected AB magnitude (optical bands only) |
| `z_phot_mean/median/std`, `z_phot_l68/u68`, `z_spec` | Cleaned photo-z fields (sentinel ≤ −90 → NaN) |

**Unit notes:** 1 nanomaggy = 3.631×10⁻⁶ Jy = 3.631×10⁻³ mJy; AB mag = 22.5 − 2.5·log10(flux/nmgy).
Legacy "no value" sentinel is −99; the script treats any value ≤ −90 as NaN.

**What you get scientifically:** optical + near-IR SED (g→z + W1,W2), morphology
(point-like vs extended), dust-corrected magnitudes, and — most importantly —
**redshifts** (photometric for most, spectroscopic where available). This is the
script that gives you distance/redshift information.

---

## 2. AllWISE Source Catalog — `match_awise_catalog.py`

**What it is:** The AllWISE Source Catalog (IRSA table `allwise_p3as_psd`) is the
all-sky mid-infrared point-source catalog from WISE, combining the cryogenic and
NEOWISE phases. It is **photometric only — no redshift.** It also carries a built-in
positional match to 2MASS near-IR photometry.

**Query:** IRSA cone search of `allwise_p3as_psd`, keeping the nearest source within
radius. Only 24 of the catalog's ~298 columns are requested to keep queries light.

### Columns pulled from AllWISE

| Column | Meaning |
|---|---|
| `designation` | AllWISE source name (J2000 position-based ID) |
| `ra`, `dec` | Source position (deg) |
| `w1mpro … w4mpro` | Profile-fit **Vega** magnitudes in W1 (3.4 µm), W2 (4.6 µm), W3 (12 µm), W4 (22 µm) |
| `w1sigmpro … w4sigmpro` | 1σ uncertainties on each W-band magnitude |
| `w1snr … w4snr` | Signal-to-noise ratio in each band |
| `ph_qual` | Photometric quality flag per band (A/B/C/U/X/Z) |
| `cc_flags` | Contamination & confusion flags (artifacts: diffraction spikes, halos, etc.) |
| `ext_flg` | Extended-source flag (0 = point source; >0 = associated with extended emission) |
| `var_flg` | Variability flag per band |
| `j_m_2mass`, `j_msig_2mass` | 2MASS J-band mag + uncertainty (built-in 2MASS match) |
| `h_m_2mass`, `h_msig_2mass` | 2MASS H-band mag + uncertainty |
| `k_m_2mass`, `k_msig_2mass` | 2MASS K-band mag + uncertainty |

### Derived / written columns in the output CSV

| Output column | How it's computed |
|---|---|
| `widx`, `in_ra`, `in_dec`, `z` | Carried from the input radio catalog (`z` = input redshift) |
| `n_match` | Number of AllWISE sources within the radius |
| `awise_designation`, `awise_ra`, `awise_dec` | From the matched AllWISE source |
| `sep_arcsec` | Angular separation radio↔AllWISE counterpart (arcsec) |
| `w{1-4}mpro`, `w{1-4}sigmpro`, `w{1-4}snr` | Vega mags, uncertainties, SNR |
| `f_w{1-4}_mjy` | Flux density in mJy = `ZP_Jy × 10^(−mag/2.5) × 1000` |
| `w1_w2` | W1−W2 colour (Vega) |
| `agn_wise` | Boolean AGN flag — `True` if `W1−W2 ≥ 0.8` (Stern et al. 2012 criterion) |
| `ph_qual`, `cc_flags`, `ext_flg`, `var_flg` | Quality/contamination flags (above) |
| `j/h/k_m_2mass`, `*_msig_2mass` | 2MASS near-IR photometry |

**Vega zero-point flux densities (Jy)** used for mag→flux: W1 = 309.540, W2 = 171.787,
W3 = 31.674, W4 = 8.363. (flux = ZP × 10^(−mag/2.5).)

**What you get scientifically:** mid-IR SED (W1–W4, 3.4–22 µm) plus near-IR (2MASS
J,H,K), data-quality flags to filter bad photometry, and a ready-made **WISE AGN
diagnostic** (W1−W2 colour and the Stern+2012 flag) to identify dusty/obscured AGN
among your radio sources. No redshift is provided here — it is carried through from
the input catalog's `z` column.

---

## Summary: which script for what?

- **Need redshift / distance, optical morphology, dust-corrected optical+W1/W2 SED?**
  → `match_desi_legacy.py` (DESI Legacy Tractor + photo_z).
- **Need full mid-IR coverage (W1–W4), near-IR (2MASS), or an AGN colour cut?**
  → `match_awise_catalog.py` (AllWISE).

The two are complementary: DESI gives optical+redshift (and W1,W2), AllWISE gives the
longer-wavelength IR bands (W3, W4) and the AGN diagnostic.
