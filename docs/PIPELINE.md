# Pipeline overview and data lineage

This traces what each notebook in `notebooks/` reads and writes, and which paper it feeds.
Reconstructed from static analysis of the notebooks (grep of read/write calls) plus manual
verification against `results/` and `data/` — a few upstream notebooks assign paths via
variables rather than literals, so their I/O couldn't be captured automatically; those are
marked "not captured" below rather than guessed.

11 notebooks originally had hardcoded absolute paths (`/home/le/data/...`, `/home/le/results/...`)
left over from before this repo existed as its own folder — these were mechanically rewritten to
the relative `data/...`/`results/...` paths the rest of the pipeline already used, so the pipeline
actually runs from a checkout of this repo rather than only on the original machine.

`data/01_preprocessed/` also had a set of large superseded intermediate files (early-iteration
article-cleaning outputs like `cleaned_dataset*.csv/.parquet`, `articles_cleaned_final.csv`,
`articles_cleaned_iqr_filtered.*`, `all_news_2008_before_matching.parquet` — none referenced by
any notebook kept in this repo) that were removed, along with a handful of small superseded
root-level files (old `filtering_log*.json` variants, a duplicate `.parquet` of a review sample
whose `.csv` twin is the one actually read, an unreferenced embeddings pickle, one tiny removed-
articles log). ~2.6GB reclaimed.

`results/` had the same problem, worse: 126 of 193 files (~198.6MB of ~230MB) turned out to be
outputs of the pre-consolidation `60`/`61`/`61b` notebooks and the `62`–`64` crisis-validation /
`22_cross_validation_v3` / `23_baseline_comparisons` notebooks already dropped from this repo —
their code is gone but their old outputs (every `_strict` variant, older `_v2`/`_v3` price caches
and event-study runs, a whole `emotion_filter_comparison/` and `financial_sector_validation/`
subfolder, two 68MB classification parquets) were sitting unused. Confirmed by checking every
file's basename against the literal source of all 18 kept notebooks + `src/*.py` before deleting
anything. What's left in `results/` (~40MB) is exactly what `21`–`23` and `30`–`32` read or write.


This split is inferred from notebook content and naming, not dictated — correct it if it's wrong.

## Stage 1 — Taxonomy, dictionary, entity linking

| Notebook | Reads | Writes |
|---|---|---|
| `00_import_msci.ipynb` | — | setup/import utility, no formal output |
| `00_validate_taxonomy.ipynb` | not captured | `data/00_raw/taxonomy_overlap_recommendations.csv` |
| `01_org_match.ipynb` | `data/all_articles.csv` | Organization matching. **Note:** the notebook's own print statements reference `data/matched_output_v6.parquet` and `data/discarded_output_v6.parquet` as outputs, but neither file exists anywhere on disk — either the notebook was never run to completion, the real output path differs from what's printed, or the file was later deleted. Flagging for you to check rather than guessing. (Correction: an earlier draft of this doc named these files `01_org_matched_v6.parquet`/`discarded_output_v6_clean.csv` — those were mistranscribed; the names above are what the notebook source actually says.) |
| `11_get_candidate_synonyms.ipynb` | not captured | not captured |
| `11b_validate_candidate_phrases.ipynb` | not captured | not captured |
| `12_calculate_similarities_sbert.ipynb` | `data/00_raw/taxonomy_sheet_v2.csv`, `results/phase i/11_candidate_phrases_v4.csv` | not captured |
| `13_preprocess_dictionary.ipynb` | not captured | not captured (feeds `results/phase i/13_final_dictionary_v4.csv`, consumed by `23`) |
| `14_filter_sections.ipynb` | `data/00_raw/all_articles.csv` | `data/01_preprocessed/articles_cleaned_sections_from_2008.csv` |
| `15_link_entity_to_data_cleaned.ipynb` | `data/01_preprocessed/articles_cleaned_final_v2.parquet`, `data/full_dataset_review_sample_reviewed.csv` | `data/01_preprocessed/articles_entity_linked_cleaned.{csv,parquet}` |
| `16_check_quality_issue_from_data_cleaned.ipynb` | `data/01_preprocessed/articles_cleaned_sections_from_2008.csv`, `data/articles_linked_improved_full.parquet` | `data/01_preprocessed/articles_entity_linked_cleaned.{csv,parquet}` (final, post-QA version) |

Manual review steps behind `15`/`16` are documented in [MANUAL_REVIEW.md](MANUAL_REVIEW.md)
(entity-linking correctness review, coverage review).

## Stage 2 — Labeling and classification 

| Notebook | Reads | Writes |
|---|---|---|
| `21_provisional_multi_labelling.ipynb` | not captured; also reads `data/samples_reviewed/rule_classification_validation_sample_reviewed.csv`, which **does not exist on disk** — flagging rather than guessing, same as the `01_org_match` note above | provisional multi-label samples under `data/samples/` |
| `22a_create_test_validation_sets.ipynb` | not captured | test/validation sets under `data/samples/` |
| `22_cross_validation_analysis.ipynb` | not captured | threshold sweeps, confusion matrices, `comprehensive_comparison.csv` etc. under `results/cross_validation_analysis/` — **canonical** CV notebook, referenced by `23` for its chosen confidence threshold. (A `22_cross_validation_v3.ipynb` existed as an older/alternate run not referenced by `23` and has been removed.) |
| `23_full_dataset_classification.ipynb` | `data/01_preprocessed/sentences_entity_linked_cleaned.parquet`, `results/phase i/13_final_dictionary_v4.csv`, `data/samples_reviewed/labelled_set_final.csv` | `results/full_dataset_classification/full_dataset_labeled.parquet` |

`23` is the main classifier: BGE-M3 embeddings + keyword semantic similarity, confidence
threshold τ = 0.40 (articles below threshold → "No Event"), applied after ambiguity/tier-hierarchy
resolution. `data/samples_reviewed/labelled_set_final.csv` is the ground-truth set used for its
reported classification metrics — see [MANUAL_REVIEW.md](MANUAL_REVIEW.md) for how it was built.

`demo_article_classification.ipynb` is a standalone, illustrative walkthrough of the Stage 1/2
classification logic on two example articles (CrowdStrike, AstraZeneca). It does not feed the
pipeline and is not required to reproduce results — kept for auditing the method.

## Stage 3 — Emotion peaks, event study, cross-tabulation

| Notebook | Reads | Writes |
|---|---|---|
| `30_organization_emotion_profiles.ipynb` | `data/01_preprocessed/entity_matches_cleaned.parquet`, `data/01_preprocessed/sentences_entity_linked_cleaned.parquet`, `results/full_dataset_classification/full_dataset_labeled.parquet` | `results/phase_iii/organization_daily_emotions_v2.parquet`, `results/phase_iii/all_peaks_loose_threshold.{parquet,csv}`, `results/phase_iii/validation_sample_peaks.csv` (manual-review sample; only regenerated if it doesn't already exist, so a rerun won't overwrite your annotations) |
| `31_event_study.ipynb` | `data/00_raw/msci_world.csv`, `results/phase_iii/all_peaks_loose_threshold.parquet`, `results/phase_iv/price_cache_v4_acwi.parquet` | `results/phase_iv/event_study_results_v3.parquet` + CSV/tex summaries (robustness, sector, region, placebo test) |
| `32_cross_tabulation.ipynb` | `results/phase_iii/all_peaks_loose_threshold.parquet`, `results/phase_iv/event_study_results_v3.parquet` | `results/phase_iv/crosstab_results.csv`, SCCT pair-margin tables, `paper_crosstab_table.csv` |

- `30` detects sustained negative-emotion elevation ("peaks") per organization from rolling
  z-score baselines over four emotions (anger, sadness, fear, disgust): z-score threshold = 1.5,
  minimum duration = 2 consecutive days (the most permissive setting from the sensitivity sweep,
  chosen to maximize recall ahead of manual validation — see MANUAL_REVIEW.md).
- `31` computes cumulative abnormal returns (CAR) via an OLS market-model event study against the
  MSCI World index: estimation window [-210, -11] trading days, event window [-1, +5], significance
  via the BMP test and the Wilcoxon signed-rank test. Firm-level prices are fetched via `yfinance`
  and cached at `results/phase_iv/price_cache_v4_acwi.parquet`.
- `32` cross-tabulates peaks against SCCT clusters/tiers for the appendix tables. It
  contains its own internal crisis-type classification (`classify_crisis_type()`), which is why
  a separate crisis-typing step isn't needed between `30` and `31` — see note below.

### Corrected pipeline flow (previously mis-documented)

The old `code_availability_submission/README.md` claimed a four-step sequential flow
(`01→02→03→04`) where step `03` (crisis-type classification) fed step `04` (event study). That
was verified to be **inaccurate**: the event-study notebook actually reads directly from the
emotion-peak notebook's output, never from a crisis-typing intermediate step, and the equivalent
crisis-typing logic lives inside `32_cross_tabulation.ipynb` instead. The standalone crisis-typing
notebook has been dropped from this repo as redundant/unintegrated. The corrected flow is:

```
23_full_dataset_classification  →  full_dataset_labeled.parquet
        ↓
30_organization_emotion_profiles  →  all_peaks_loose_threshold.parquet
        ↓
31_event_study  →  event_study_results_v3.parquet
        ↓
32_cross_tabulation  (reads both 30's and 31's outputs; classifies crisis type internally)
```

## Environment

Python 3.10+. See `requirements.txt`. `.env.local` (gitignored) holds API keys used by
`11_get_candidate_synonyms.ipynb` (the only remaining notebook that loads it via `dotenv`).
Keyword/sentence embedding caches (`embeddings/keyword_embeddings_v4.pt`,
`embeddings/sentence_embeddings_v4.pt`) are created on first run if absent and reused afterward.
