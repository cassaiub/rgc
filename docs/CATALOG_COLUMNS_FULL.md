# Full Column Reference — DESI Legacy & AllWISE

Authoritative column lists pulled **live** from the catalog services
(NOIRLab Astro Data Lab for DESI; IRSA TAP for AllWISE) on 2026-06-15.

Column counts (full catalogs):

| Table | Total columns | Currently fetched |
|---|---|---|
| `ls_dr9.tractor` | **187** | 15 |
| `ls_dr9.photo_z` | **14** | 6 |
| `ls_dr10.tractor` | **214** | 16 |
| `ls_dr10.photo_z` | **24** | 6 |
| `allwise_p3as_psd` | **298** | 24 |

✅ = currently fetched by the script. Everything else is available but unused.

Flux units in Tractor are **nanomaggies** (AB); `mag = 22.5 − 2.5·log10(flux)`.
DR10 adds the **i-band** and a handful of extra columns vs DR9 — those are flagged
"(DR10)". Where a column exists per-band, the band suffixes are listed once.

---

# 1. DESI Legacy — `tractor` table

Grouped by purpose. Band suffixes: DR9 = `g r z w1 w2 w3 w4`; DR10 = `g r i z w1 w2 w3 w4`.

## 1.1 Identifiers & bookkeeping
| Column | Description |
|---|---|
| `ls_id` ✅ | Unique Legacy Survey object ID (join key to photo_z) |
| `objid` | Object number within the brick |
| `brickid` | Brick ID [1, 662174] |
| `brickname` | Brick name encoding sky position (e.g. 1126p222) |
| `release` | Integer for camera/filter set / processing run |
| `brick_primary` | True if object is inside the brick boundary (dedup flag) |
| `ref_cat` | Reference catalog: T2=Tycho-2, G2=Gaia DR2 (GE=EDR3 in DR10), L3=SGA |
| `ref_id` | Reference catalog identifier |
| `ref_epoch` | Reference epoch (e.g. 2015.5 for Gaia) |
| `random_id` | Random ID in [0,100] (for subsampling) |

## 1.2 Position, astrometry & sky indexing
| Column | Description |
|---|---|
| `ra`, `dec` ✅ | J2000 position (deg) |
| `ra_ivar`, `dec_ivar` | Inverse variance on RA / Dec (positional error) |
| `bx`, `by` | X/Y pixel position in the brick image stack |
| `glon`, `glat` | Galactic longitude / latitude |
| `elon`, `elat` | Ecliptic longitude / latitude |
| `htm9` | HTM spatial index (order 9, ~10′) |
| `ring256` | HEALPix index (Nside 256, ring, ~14′) |
| `nest4096` | HEALPix index (Nside 4096, nest, ~52″) |
| `mjd_min`, `mjd_max` | Min/max MJD of observations used in the model |

## 1.3 Proper motion & parallax (from reference catalog, mostly Gaia)
| Column | Description |
|---|---|
| `pmra`, `pmdec` | Proper motion in RA / Dec |
| `pmra_ivar`, `pmdec_ivar` | Inverse variance on proper motions |
| `parallax`, `parallax_ivar` | Parallax and its inverse variance |

## 1.4 Morphology / model
| Column | Description |
|---|---|
| `type` ✅ | Model: PSF (star), REX (round exp), EXP, DEV, SER, DUP |
| `sersic`, `sersic_ivar` | Sérsic index (type=SER) + inverse variance |
| `shape_r`, `shape_r_ivar` | Half-light radius (arcsec) + inverse variance |
| `shape_e1`, `shape_e1_ivar` | Ellipticity component 1 + inverse variance |
| `shape_e2`, `shape_e2_ivar` | Ellipticity component 2 + inverse variance |
| `dchisq_1 … dchisq_5` | Δχ² between successively complex fits (PSF,REX,DEV,EXP,SER) |
| `fitbits` | Bitmask of how the object was fit |

## 1.5 Fluxes (model) — the core photometry
| Column | Description |
|---|---|
| `flux_g`, `flux_r`, `flux_z` ✅ | Optical model flux (nanomaggies); **`flux_i` (DR10)** ✅ |
| `flux_w1`, `flux_w2` ✅ | WISE forced-photometry flux (W1, W2) |
| `flux_w3`, `flux_w4` | WISE forced-photometry flux (W3, W4) — **not fetched** |
| `flux_ivar_g/r/(i)/z` | Inverse variance per optical band — **real flux errors** |
| `flux_ivar_w1…w4` | Inverse variance per WISE band |

## 1.6 Dereddened fluxes & magnitudes (Galactic-extinction corrected)
| Column | Description |
|---|---|
| `dered_flux_g/r/z` ✅ | Dereddened optical flux; **`dered_flux_i` (DR10)** |
| `dered_flux_w1…w4` | Dereddened WISE flux |
| `dered_mag_g/r/(i)/z` | Dereddened magnitudes (precomputed) |
| `dered_mag_w1…w4` | Dereddened WISE magnitudes |

## 1.7 Convenience magnitudes & colors (precomputed by Data Lab)
| Column | Description |
|---|---|
| `mag_g/r/(i)/z`, `mag_w1…w4` | Converted magnitudes (not extinction-corrected) |
| `g_r`, `r_z`, `z_w1` | Optical/IR colors; **`r_i`, `i_z` (DR10)** |
| `w1_w2`, `w2_w3`, `w3_w4` | WISE colors (W1−W2 is the AGN diagnostic) |

## 1.8 Fiber fluxes (for spectroscopic targeting)
| Column | Description |
|---|---|
| `fiberflux_g/r/(i)/z` | Predicted flux in a 1.5″ fiber, 1″ seeing |
| `fibertotflux_g/r/(i)/z` | Same, from all sources at that location |

## 1.9 Signal-to-noise
| Column | Description |
|---|---|
| `snr_g/r/(i)/z` | Optical S/N |
| `snr_w1…w4` | WISE S/N |

## 1.10 Galactic extinction
| Column | Description |
|---|---|
| `ebv` | E(B−V) reddening from SFD98 |
| `mw_transmission_g/r/(i)/z`, `_w1…w4` | Linear transmission [0,1] per band |

## 1.11 Depth, coverage & fit quality
| Column | Description |
|---|---|
| `nobs_g/r/(i)/z`, `_w1…w4` | Number of images contributing at the central pixel |
| `ngood_g/r/i/z` (DR10) | Number of good (unmasked) contributing images |
| `rchisq_g/r/(i)/z`, `_w1…w4` | Profile-weighted reduced χ² of the model fit |
| `fracflux_g/r/(i)/z`, `_w1…w4` | Fraction of flux from other (blended) sources |
| `fracin_g/r/(i)/z` | Fraction of source flux inside the blob (~1 for real) |
| `fracmasked_g/r/(i)/z` | Fraction of masked pixels [0,1] |
| `psfsize_g/r/(i)/z` | Weighted-average PSF FWHM |
| `psfdepth_g/r/(i)/z`, `_w1…w4` | 5σ point-source depth |
| `galdepth_g/r/(i)/z` | 5σ galaxy depth (0.45″ exp) |
| `nea_g/r/(i)/z`, `blob_nea_*` | (Blob-masked) noise-equivalent area |

## 1.12 Masks
| Column | Description |
|---|---|
| `maskbits` ✅ | Touches a flagged pixel (bright star, etc.) — primary quality cut |
| `anymask_g/r/(i)/z` | Central pixel flagged in *any* image |
| `allmask_g/r/(i)/z` | Central pixel flagged in *all* images |
| `wisemask_w1`, `wisemask_w2` | WISE bitmasks |

## 1.13 WISE coadd references
| Column | Description |
|---|---|
| `wise_coadd_id` | unWISE coadd brick name |
| `wise_x`, `wise_y` | Pixel coords in the WISE coadd |

## 1.14 Gaia cross-match (point-source astrometry/photometry)
`gaia_phot_g_mean_mag`, `gaia_phot_bp_mean_mag`, `gaia_phot_rp_mean_mag`,
`gaia_phot_{g,bp,rp}_mean_flux_over_error`, `gaia_phot_{g,bp,rp}_n_obs`,
`gaia_phot_variable_flag`, `gaia_phot_bp_rp_excess_factor`,
`gaia_astrometric_excess_noise(_sig)`, `gaia_astrometric_n_obs_al`,
`gaia_astrometric_n_good_obs_al`, `gaia_astrometric_weight_al`,
`gaia_astrometric_sigma5d_max`, `gaia_astrometric_params_solved`,
`gaia_duplicated_source`, `gaia_a_g_val`, `gaia_e_bp_min_rp_val`.

---

# 2. DESI Legacy — `photo_z` table

DR9 = 14 columns; DR10 = 24 (adds i-band photo-z variants + k-fold indices).

| Column | Description |
|---|---|
| `ls_id` ✅ | Join key to tractor |
| `brickid`, `objid` | Brick / object identifiers |
| `z_phot_mean` ✅ | Photo-z from the mean of the PDF |
| `z_phot_median` ✅ | Photo-z from the median of the PDF |
| `z_phot_std` ✅ | Std dev of the photo-z PDF (uncertainty) |
| `z_phot_l68`, `z_phot_u68` ✅ | 68% confidence interval bounds |
| `z_phot_l95`, `z_phot_u95` | 95% confidence interval bounds — **not fetched** |
| `z_spec` ✅ | Spectroscopic redshift if available (else sentinel) |
| `release` | Camera/filter set |
| `training` | Whether the spec-z was used in photo-z training |
| `survey` | Source survey of the spec-z |
| **DR10 extras** | |
| `z_phot_mean_i`, `z_phot_median_i`, `z_phot_std_i` | Photo-z including i-band |
| `z_phot_l68_i`, `z_phot_u68_i`, `z_phot_l95_i`, `z_phot_u95_i` | i-band intervals |
| `training_i` | i-band training flag |
| `kfold`, `kfold_i` | 10-fold cross-validation subset index |

---

# 3. AllWISE Source Catalog — `allwise_p3as_psd` (298 columns)

Grouped by purpose. Band index b ∈ {1,2,3,4} = W1(3.4µm), W2(4.6µm), W3(12µm), W4(22µm).

## 3.1 Identifiers & position
| Column | Description |
|---|---|
| `designation` ✅ | WISE source name (J2000) |
| `ra`, `dec` ✅ | J2000 position |
| `sigra`, `sigdec`, `sigradec` | Position uncertainties + cross-term |
| `glon`, `glat`, `elon`, `elat` | Galactic / ecliptic coordinates |
| `cntr`, `source_id`, `coadd_id`, `src` | Unique counters / coadd identifiers |
| `wx`, `wy` | Pixel coordinates (all bands) |
| `x`, `y`, `z`, `spt_ind`, `htm20` | Unit-sphere position + spatial indices |

## 3.2 Profile-fit photometry (primary mags)
| Column | Description |
|---|---|
| `w{b}mpro` ✅ | Profile-fit Vega magnitude, band b |
| `w{b}sigmpro` ✅ | Uncertainty on the mag |
| `w{b}snr` ✅ | Signal-to-noise ratio |
| `w{b}rchi2` | Reduced χ² of the profile fit |
| `rchi2` | Total reduced χ² over all bands |
| `nb` | Number of blend components in the fit |
| `na` | Active deblend flag (1 = actively deblended) |

## 3.3 Raw flux & sky (profile fit)
`w{b}flux`, `w{b}sigflux` (raw flux + uncertainty), `w{b}sky`, `w{b}sigsk`
(sky background + uncertainty), `w{b}conf` (sky confusion).

## 3.4 Saturation
`w{b}sat` (fraction of saturated pixels per band), `satnum` (sample at which
saturation onsets, 1 char/band).

## 3.5 Proper motion (AllWISE apparent motion)
`ra_pm`, `dec_pm`, `sigra_pm`, `sigdec_pm`, `sigradec_pm`, `pmra`, `sigpmra`,
`pmdec`, `sigpmdec`, `w{b}rchi2_pm`, `rchi2_pm`, `pmcode`.

## 3.6 Quality & contamination flags
| Column | Description |
|---|---|
| `ph_qual` ✅ | Per-band photometric quality (A=best … U=upper limit) |
| `cc_flags` ✅ | Prioritized contamination/confusion (artifacts) per band |
| `ext_flg` ✅ | Probability source is extended (not single PSF) |
| `var_flg` ✅ | Probability of variability per band |
| `det_bit` | Bit-encoded per-band detection |
| `moon_lev` | Moon-contamination level per band |
| `rel` | Small-separation multiple-detection flag |
| `w{b}cc_map`, `w{b}cc_map_str` | Per-band contamination/confusion bit & char arrays |

## 3.7 Measurement counts & coverage
`w{b}nm` (# frames with SNR≥3), `w{b}m` (# frames), `w{b}cov` (mean coverage depth).

## 3.8 Duplicate resolution
`best_use_cntr` (cntr of the best source in the dup group), `ngrp` (# dup groups).

## 3.9 Standard-aperture photometry
`w{b}mag`, `w{b}sigm`, `w{b}flg` (aperture-corrected standard-aperture mag,
uncertainty, flag), `w{b}mcor` (aperture correction).

## 3.10 Multi-aperture photometry (8 fixed apertures)
For each aperture a ∈ {1..8} and band b: `w{b}mag_{a}`, `w{b}sigm_{a}`,
`w{b}flg_{a}` — instrumental magnitude, uncertainty, flag in that aperture.
(That's 8×4×3 = 96 columns — useful for extended/resolved profiles.)

## 3.11 Variability / repeatability
Per band b: `w{b}magp` (inverse-variance-weighted mean mag), `w{b}sigp1`,
`w{b}sigp2` (population / mean std dev), `w{b}k` (Stetson K index), `w{b}ndf`,
`w{b}mlq` (−ln Q), `w{b}mjdmin`, `w{b}mjdmax`, `w{b}mjdmean`.

## 3.12 Inter-band correlation
`rho12`, `rho23`, `rho34` (band-pair correlation coefficients);
`q12`, `q23`, `q34` (significance of the correlation).

## 3.13 2MASS association (built-in near-IR cross-match)
| Column | Description |
|---|---|
| `tmass_key` | Closest 2MASS All-Sky PSC key |
| `r_2mass`, `pa_2mass`, `n_2mass` | Offset, position angle, # 2MASS sources in radius |
| `j_m_2mass`, `j_msig_2mass` ✅ | J mag + uncertainty |
| `h_m_2mass`, `h_msig_2mass` ✅ | H mag + uncertainty |
| `k_m_2mass`, `k_msig_2mass` ✅ | Ks mag + uncertainty |

## 3.14 2MASS XSC (extended-source / galaxy) parameters
`xscprox` (distance to XSC galaxy center); per band b: `w{b}rsemi` (semi-major
axis), `w{b}ba` (axis ratio), `w{b}pa` (position angle), `w{b}gmag`,
`w{b}gerr`, `w{b}gflg` (elliptical-aperture galaxy mag, uncertainty, flag).

---

# 4. Notes for deciding what else to fetch

**High-value DESI columns not currently pulled:**
- `flux_ivar_*` — real per-band flux errors (you currently report magnitudes with no error bar).
- `flux_w3`, `flux_w4` — DESI already has forced W3/W4; would extend the mid-IR SED without touching AllWISE.
- `shape_r`, `sersic`, `shape_e1/e2` — quantitative morphology beyond the `type` label.
- `snr_*`, `nobs_*`, `rchisq_*`, `fracflux_*`, `anymask_*`, `wisemask_*` — cleaning cuts.
- `mw_transmission_*` / `ebv` — the extinction factors behind the dered fluxes.
- `gaia_phot_g_mean_mag`, `parallax`, `pmra/pmdec` — star/QSO discrimination.
- photo_z: `z_phot_l95/u95` (95% interval); DR10 i-band photo-z variants.

**High-value AllWISE columns not currently pulled:**
- `w{b}flux`/`w{b}sigflux` — raw fluxes if you prefer flux-space over Vega mags.
- `w{b}rchi2`, `nb`, `na`, `rel` — fit quality / deblending / reliability.
- `w{b}sat`, `w{b}cc_map` — saturation and detailed contamination beyond `cc_flags`.
- `pmra`/`pmdec` (+sig) — AllWISE motion (helps reject stars).
- `w{b}m`, `w{b}nm`, `w{b}cov` — frame coverage / detection robustness.
- XSC galaxy columns (`w{b}gmag`, `w{b}rsemi`, …) for resolved sources.
