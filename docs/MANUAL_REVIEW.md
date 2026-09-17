# Manual review, annotation, and audit steps

The automated pipeline (`notebooks/`) has several human-in-the-loop review passes that don't show
up as notebook code — they live as CSV/report files under `data/samples/`,
`data/samples_reviewed/`, and `results/*/`. This reconstructs what each pass did from the files
themselves (columns, row counts, and the `.txt`/`.json` reports that were generated alongside
them), in the same spirit as the `exp-*` notes. **This is my reconstruction, not a transcript of
what you actually did — please correct anything that's wrong or incomplete.**

## 1. Entity-linking / organization-matching review (feeds `14`–`16`)

| File | Rows | Purpose |
|---|---|---|
| `data/full_dataset_review_sample.csv` → `..._reviewed.csv` | 201 → 200 | Manually checked whether each sampled article's matched entity/company/ticker is correct (`is_keyword_validated` etc.). |
| `data/samples_reviewed/org_matched_sample.csv` | 167 | QA pass on organization matching, with an explicit `org_correctly_matched` column and free-text `reviewer_note`. |
| `data/coverage_review_sample.csv` → `..._REVIEWED.csv` | 72 → 71 | Coverage/false-negative check — for each article, whether it actually contains an MSCI-listed company (`contains_msci_company`) that the automated matcher may have missed. |
| `data/samples_reviewed/validation_sample_hybrid_reviewed.csv` | 100 | Validates the hybrid fuzzy + bi-encoder/cross-encoder entity-matching method, with an `is_correct_match` label per case. |

## 2. Quality filtering review (feeds `16` / repetition & length filters)

| File | Rows | Purpose |
|---|---|---|
| `data/quality_filter_validation_set.csv` | 28 | Reviewed removal reasons for filtered articles (`removal_reason`, `num_repetitions`). |
| `data/repetition_validation.csv` / `..._corrected.csv` | 24 | Validated repetitive-article detection. |
| `data/downstream_fp_test_articles.csv` | 19 | Held-out false-positive test cases for the entity-type filter. |
| `data/long_articles_for_review.csv` | 1,361 | Queue of unusually long articles flagged for manual length/repetition review. |

## 3. Provisional multi-labeling review (feeds `21`)

`data/samples/21_provisionally_labeled_samples.csv` (401 rows, automated multi-category flags)
was manually reviewed into `data/samples_reviewed/21_provisionally_labeled_samples_reviewed.csv`
(400 rows) — one row dropped during review. Category names here (`Communication and media`,
`Financial performance`, ...) are an earlier taxonomy naming, later consolidated into the
7-category taxonomy used everywhere downstream (Governance, Personnel, Products, IT/Data,
Processes, Communication, Legal).

## 4. Test/validation set construction and dual annotation (feeds `22`/`22a`)

`22a` produced provisional test/validation sets for labeling under `data/samples/`. Two
annotators — initials **DH** and you (**LN**) — independently labeled articles into the
7-category taxonomy. `data/samples_reviewed/labelled_set_final.csv` (341 rows: `article_id, text,
label_DH, label_LN, actual_label, compare`) is the reconciled ground-truth set actually consumed
by `23_full_dataset_classification.ipynb` for its reported classification metrics.

**Not reported here on purpose:** `data/samples_reviewed/` also holds several intermediate files
(`test_set*.csv`, `validation_set*.csv`, `audit_results/`) from earlier rounds of this process,
including a disagreement-audit pass. Those numbers were never intended for publication and are
deliberately left out of this writeup — only `labelled_set_final.csv`, the file the pipeline
actually reads, is documented above. If a different inter-annotator agreement figure is reported
in the paper, it isn't derived from anything in this repo.

## 5. Emotion-peak / crisis validation (feeds `30`–`32`)

**Not done yet.** `results/phase_iii/validation_sample_peaks.csv` (152 peaks) is written directly
by `30_organization_emotion_profiles.ipynb`, which checks whether this file already exists before
regenerating it — so once a real manual annotation exists, a full pipeline rerun won't overwrite
it.

Earlier drafts of this doc described a 50-peak sample as manually reviewed across three files
(`emotion_validation_sample_labeled.csv`, `emotion_validation_with_taxonomy_reasoning.csv`,
`emotion_validation_sample_crisis_validated.csv`, plus follow-ups `no_cases_categorized.csv` /
`possibly_cases_analyzed.csv`). That was wrong — those were a test run using an LLM to generate
the labels/validation, not a real review, and have been deleted.
