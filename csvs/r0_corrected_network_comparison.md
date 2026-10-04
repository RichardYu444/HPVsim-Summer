# R0-corrected networks: pairwise comparison

Three runs of run_r0_corrected_1900.py (mean degree 1.4, R0 1-1.5 per genotype, hpv16/hpv18/hi5,
100k agents, 1900-2050, 50 runs each). A = first network named in the row, B = second.

## Overall HPV prevalence time series (median of 50 runs per year, 151 years)

| Comparison | Pearson r | DTW, z-normalised (shape) | DTW, raw (pp; shape + level) |
|---|---|---|---|
| Default vs Gamma-2 | 0.088 | 9.08 | 18.92 |
| Default vs Power law | 0.070 | 11.05 | 17.11 |
| Gamma-2 vs Power law | 0.794 | 5.83 | 5.74 |

Noise floor -- the same measures between two disjoint 25-run halves of one network (mean of 20 random splits):

| Network | Pearson r | DTW, z-normalised | DTW, raw (pp) |
|---|---|---|---|
| Default | 1.000 | 0.20 | 0.56 |
| Gamma-2 | 0.965 | 1.55 | 2.34 |
| Power law | 0.995 | 0.53 | 1.55 |

## Cumulative cancers (sum of yearly new cancers, UK-scaled), Welch t-test on 50 vs 50 runs

### overall (1900-2050)

| Comparison | mean A | mean B | A - B | % diff | t | df | p | Cohen d | Mann-Whitney p |
|---|---|---|---|---|---|---|---|---|---|
| Default vs Gamma-2 | 537,228 | 337,675 | 199,553 | +59.1% | 14.43 | 51.0 | 1.24e-19 | 2.89 | 1.84e-17 |
| Default vs Power law | 537,228 | 325,286 | 211,942 | +65.2% | 17.40 | 51.6 | 3.14e-23 | 3.48 | 7.07e-18 |
| Gamma-2 vs Power law | 337,675 | 325,286 | 12,389 | +3.8% | 0.68 | 96.4 | 0.498 | 0.14 | 0.287 |

### pre-screening (1900-1979)

| Comparison | mean A | mean B | A - B | % diff | t | df | p | Cohen d | Mann-Whitney p |
|---|---|---|---|---|---|---|---|---|---|
| Default vs Gamma-2 | 359,369 | 172,260 | 187,109 | +108.6% | 25.94 | 53.5 | 5.67e-32 | 5.19 | 7.07e-18 |
| Default vs Power law | 359,369 | 171,361 | 188,007 | +109.7% | 34.94 | 57.3 | 2.39e-40 | 6.99 | 7.07e-18 |
| Gamma-2 vs Power law | 172,260 | 171,361 | 899 | +0.5% | 0.10 | 89.8 | 0.918 | 0.02 | 0.642 |

### post-screening (1980-2050)

| Comparison | mean A | mean B | A - B | % diff | t | df | p | Cohen d | Mann-Whitney p |
|---|---|---|---|---|---|---|---|---|---|
| Default vs Gamma-2 | 177,859 | 165,415 | 12,445 | +7.5% | 1.84 | 50.6 | 0.072 | 0.37 | 0.712 |
| Default vs Power law | 177,859 | 153,924 | 23,935 | +15.5% | 3.43 | 50.5 | 0.001 | 0.69 | 0.003 |
| Gamma-2 vs Power law | 165,415 | 153,924 | 11,490 | +7.5% | 1.19 | 97.9 | 0.237 | 0.24 | 0.134 |

## Per-network cumulative cancers (exact numbers)

| Network | Period | Mean | SD | 95% CI of mean | Median | Min | Max |
|---|---|---|---|---|---|---|---|
| Default | overall (1900-2050) | 537,228 | 13,799 | 533,306 - 541,150 | 536,933 | 509,297 | 569,813 |
| Default | pre-screening (1900-1979) | 359,369 | 10,690 | 356,330 - 362,407 | 358,779 | 340,280 | 381,373 |
| Default | post-screening (1980-2050) | 177,859 | 6,129 | 176,118 - 179,601 | 178,503 | 165,721 | 193,283 |
| Gamma-2 | overall (1900-2050) | 337,675 | 96,831 | 310,155 - 365,194 | 357,906 | 132,517 | 527,971 |
| Gamma-2 | pre-screening (1900-1979) | 172,260 | 49,876 | 158,085 - 186,435 | 180,076 | 74,697 | 274,671 |
| Gamma-2 | post-screening (1980-2050) | 165,415 | 47,451 | 151,929 - 178,900 | 178,104 | 57,820 | 253,300 |
| Power law | overall (1900-2050) | 325,286 | 85,038 | 301,118 - 349,453 | 336,935 | 132,467 | 491,571 |
| Power law | pre-screening (1900-1979) | 171,361 | 36,513 | 160,984 - 181,738 | 177,430 | 84,284 | 246,510 |
| Power law | post-screening (1980-2050) | 153,924 | 49,006 | 139,997 - 167,852 | 156,059 | 48,183 | 250,304 |

## Same tests on the cumulative incidence RATE (sum of annual cancer_incidence per 100,000 women)

| Comparison | Period | mean A | mean B | % diff | t | p | Cohen d |
|---|---|---|---|---|---|---|---|
| Default vs Gamma-2 | overall | 1,744.3 | 1,060.8 | +64.4% | 15.63 | 4.15e-21 | 3.13 |
| Default vs Gamma-2 | pre-screening | 1,225.8 | 582.5 | +110.4% | 26.45 | 2.15e-32 | 5.29 |
| Default vs Gamma-2 | post-screening | 518.4 | 478.3 | +8.4% | 2.02 | 0.049 | 0.40 |
| Default vs Power law | overall | 1,744.3 | 1,035.1 | +68.5% | 18.56 | 1.56e-24 | 3.71 |
| Default vs Power law | pre-screening | 1,225.8 | 591.6 | +107.2% | 34.43 | 7.02e-40 | 6.89 |
| Default vs Power law | post-screening | 518.4 | 443.6 | +16.9% | 3.68 | 5.69e-04 | 0.74 |
| Gamma-2 vs Power law | overall | 1,060.8 | 1,035.1 | +2.5% | 0.45 | 0.655 | 0.09 |
| Gamma-2 vs Power law | pre-screening | 582.5 | 591.6 | -1.5% | -0.30 | 0.761 | -0.06 |
| Gamma-2 vs Power law | post-screening | 478.3 | 443.6 | +7.8% | 1.23 | 0.221 | 0.25 |

Notes: Pearson r carries no p-value because consecutive years are autocorrelated. Pre-screening includes the 1900-1920 ramp-up from the seeded start (no cancers in 1900-01), the same for every network. Nine cancer tests per measure (3 pairs x 3 periods): a Bonferroni threshold is 0.05/9 = 0.0056.
