# Reputational Risk Early Warning System

Design Science Research pipeline for detecting and measuring corporate reputational
risk events from news text. 

## Structure

```
notebooks/   17 pipeline notebooks (numbered, execution order below) + 1 illustrative demo
src/         reputational_risk_pipeline package (classification, dictionary, matching, config)
docs/
  PIPELINE.md        notebook-by-notebook data lineage, Paper 1 / Paper 2 mapping
  MANUAL_REVIEW.md   human annotation/review/audit steps behind the automated pipeline
data/        raw + preprocessed article data, review samples (gitignored, ~0.8GB)
embeddings/  cached sentence/keyword embeddings (gitignored, ~1.6GB)
results/     all pipeline outputs, figures, paper tables (gitignored, ~40MB)
```

`data/`, `embeddings/`, `results/` are excluded from git (see `.gitignore`) but must exist as
siblings of `notebooks/` on disk — notebooks use relative paths. They're distributed separately;
see [Data availability](#data-availability) below.

## Pipeline order

See [docs/PIPELINE.md](docs/PIPELINE.md) for the full lineage table. Short version:

```
00_import_msci, 00_validate_taxonomy
01_org_match
11_get_candidate_synonyms → 11b_validate_candidate_phrases → 12_calculate_similarities_sbert → 13_preprocess_dictionary
14_filter_sections → 15_link_entity_to_data_cleaned → 16_check_quality_issue_from_data_cleaned
21_provisional_multi_labelling → 22a_create_test_validation_sets → 22_cross_validation_analysis → 23_full_dataset_classification
30_organization_emotion_profiles → 31_event_study → 32_cross_tabulation
```


Manual annotation/review/audit steps behind several of these stages (entity-linking QA,
dual-annotator labeling + disagreement audit, emotion-peak crisis validation) are documented in
[docs/MANUAL_REVIEW.md](docs/MANUAL_REVIEW.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs the src/ package (reputational_risk_pipeline)
cp .env.local.example .env.local   # fill in API keys (used by 11_get_candidate_synonyms.ipynb)
```

## Data availability

`data/`, `embeddings/`, and `results/` are not in this repo due to size (~2.4GB). They were handed
off separately as an archive — contact the repo owner for access.
