"""
classification.py

Two-stage pipeline:
1) Sentence-level scoring
   - BAAI/bge-m3 embeddings (normalized, GPU if available)
   - For each sentence & category: mean of top-8 keyword similarities (dot product).

2) Article-level aggregation
   - For each article & category: mean of top-10 sentence scores.
   - Threshold + delta rules.
   - Tiered priorities over short category names.
   - Rule-based skips (Mets, Washington, Treasury…) → final_category = 'No Event'.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import spacy


# ---------------------------------------------------------------------
# 0. Rule-based article skip (from 22_cross_validation_v2.ipynb)
# ---------------------------------------------------------------------


def should_skip_article(text: str) -> bool:
    """
    Returns True if article should be skipped (treated as 'No Event')
    based on the rule-based blacklist in 22_cross_validation_v2.ipynb.

    Rules:
      1) 'mets' but not 'metso'
      2) 'washington' but not 'soul patts'
      3) contains 'treasury auctions set'
      4) 'treasury' without 'blackrock'
    """
    if not isinstance(text, str):
        return False

    t = text.lower()

    # 1) "Mets" but not "Metso"
    if "mets" in t and "metso" not in t:
        return True

    # 2) "Washington" but not mentioning "Soul Patts"
    if "washington" in t and "soul patts" not in t:
        return True

    # 3) contains "Treasury Auctions Set"
    if "treasury auctions set" in t:
        return True

    # 4) "Treasury" without mentioning "BlackRock"
    if "treasury" in t and "blackrock" not in t:
        return True

    return False


# ---------------------------------------------------------------------
# 1. Category rename map (long → short) and loading dictionary
# ---------------------------------------------------------------------

CATEGORY_RENAME_MAP = {
    "Communication and media": "Communication",
    "Financial performance": "Finance",
    "Information technology and data management": "IT/Data",
    "Legality and regulation": "Legal",
    "Processes and supply chains": "Processes",
    "Products and services": "Products",
    "Strategy and governance": "Governance",
    "Personnel": "Personnel",
    "No Event": "No Event",
}


def load_keyword_dictionary(path: str) -> Dict[str, List[str]]:
    """
    Load keyword dictionary CSV with at least:
    - 'category'
    - 'keyword'

    Apply CATEGORY_RENAME_MAP so categories are directly the short names.

    Returns
    -------
    dict: {short_category: [keyword1, keyword2, ...]}
    """
    df = pd.read_csv(path)

    if "category" not in df.columns or "keyword" not in df.columns:
        raise ValueError("Keyword dictionary must have 'category' and 'keyword' columns.")

    # Apply rename map (handles both long & already-short names)
    df["category_renamed"] = df["category"].apply(
        lambda c: CATEGORY_RENAME_MAP.get(c, c)
    )

    dictionaries: Dict[str, List[str]] = (
        df.groupby("category_renamed")["keyword"].apply(list).to_dict()
    )

    return dictionaries


# ---------------------------------------------------------------------
# 2. Model loading (BAAI/bge-m3, GPU by default)
# ---------------------------------------------------------------------


def load_sbert_model(
    model_name: str = "BAAI/bge-m3",
    device: str = "cuda",
) -> SentenceTransformer:
    """
    Load the sentence-transformer model.

    Default: BAAI/bge-m3 on GPU (if available), otherwise CPU.
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("[classification] CUDA not available, falling back to CPU.")
        device = "cpu"

    model = SentenceTransformer(model_name, device=device)
    return model


# ---------------------------------------------------------------------
# 3. Sentence splitting utilities (spaCy sentencizer)
# ---------------------------------------------------------------------

_SPACY_NLP = None


def get_spacy_sentencizer(model_name: str = "en_core_web_sm"):
    """
    Lazily load a spaCy model with a sentencizer.
    """
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            nlp = spacy.load(model_name)
        except OSError:
            # Fallback: try to download or use a blank model with sentencizer
            print(
                f"[classification] spaCy model '{model_name}' not found. "
                f"Using 'en_core_web_sm' or a blank English model with sentencizer."
            )
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        _SPACY_NLP = nlp
    return _SPACY_NLP


def split_articles_into_sentences(
    df_articles: pd.DataFrame,
    text_col: str = "text",
    article_id_col: str = "article_id",
) -> pd.DataFrame:
    """
    Split each article text into sentences using spaCy and return a
    sentence-level DataFrame with columns:

        article_id, sentence_id, sentence_text
    """
    nlp = get_spacy_sentencizer()

    article_ids: List = []
    sentence_ids: List[int] = []
    sentence_texts: List[str] = []

    for _, row in tqdm(
        df_articles.iterrows(),
        total=len(df_articles),
        desc="Splitting into sentences",
    ):
        article_id = row[article_id_col]
        text = row.get(text_col, "")
        if not isinstance(text, str):
            text = str(text)

        doc = nlp(text)
        s_idx = 0
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            article_ids.append(article_id)
            sentence_ids.append(s_idx)
            sentence_texts.append(sent_text)
            s_idx += 1

    sent_df = pd.DataFrame(
        {
            article_id_col: article_ids,
            "sentence_id": sentence_ids,
            "sentence_text": sentence_texts,
        }
    )
    return sent_df


# ---------------------------------------------------------------------
# 4. Embedding utilities (normalized, suitable for BGE)
# ---------------------------------------------------------------------

TOP_K_KEYWORDS = 8
TOP_K_SENTENCES = 10


def encode_keywords(
    keyword_dict: Dict[str, List[str]],
    model: SentenceTransformer,
) -> Dict[str, torch.Tensor]:
    """
    Encode keyword lists per category.

    Returns
    -------
    dict: {category: tensor[num_keywords, emb_dim]}
    """
    encoded: Dict[str, torch.Tensor] = {}
    for category, keywords in keyword_dict.items():
        emb = model.encode(
            keywords,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # important for BGE models
        )
        encoded[category] = emb  # stays on same device as model (GPU if cuda)
    return encoded


def encode_sentences(
    df_sentences: pd.DataFrame,
    model: SentenceTransformer,
    sentence_col: str = "sentence_text",
) -> torch.Tensor:
    """
    Encode sentence texts.
    """
    texts = df_sentences[sentence_col].astype(str).tolist()
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # important for BGE models
    )
    return embeddings  # tensor on GPU if model is on GPU


# ---------------------------------------------------------------------
# 5. Sentence-level scoring and article-level aggregation
# ---------------------------------------------------------------------


def compute_sentence_category_scores(
    sentence_embeddings: torch.Tensor,
    df_sentences: pd.DataFrame,
    keyword_embeddings: Dict[str, torch.Tensor],
    article_id_col: str = "article_id",
) -> pd.DataFrame:
    """
    For each sentence, compute similarity score with each category's keywords.

    Sentence-level score per category:
      - Compute dot products with all keywords (normalized embeddings).
      - Take the top TOP_K_KEYWORDS most similar keywords.
      - Score = mean of those top-k scores.

    Returns
    -------
    DataFrame with columns:
      [article_id_col, sentence_id] + one column per category.
    """
    num_sentences = sentence_embeddings.shape[0]
    categories = list(keyword_embeddings.keys())

    records: List[Dict] = []

    for i in tqdm(range(num_sentences), desc="Scoring sentences"):
        emb = sentence_embeddings[i]  # [dim], on device
        row = {
            article_id_col: df_sentences.iloc[i][article_id_col],
            "sentence_id": df_sentences.iloc[i]["sentence_id"],
        }

        for category in categories:
            kw_emb = keyword_embeddings[category]  # [n_kw, dim]
            # sims: [n_kw], dot product because embeddings are normalized
            sims = torch.matmul(kw_emb, emb)
            k = min(TOP_K_KEYWORDS, sims.shape[0])
            if k == 0:
                score = float("nan")
            else:
                topk_vals = torch.topk(sims, k=k).values
                score = float(topk_vals.mean().item())
            row[category] = score

        records.append(row)

    sent_scores_df = pd.DataFrame(records)
    return sent_scores_df


def aggregate_sentence_scores_to_article(
    sent_scores_df: pd.DataFrame,
    article_id_col: str = "article_id",
) -> pd.DataFrame:
    """
    Aggregate sentence-level scores to article-level scores.

    Article-level score for each category:
      - For all sentences of the article, take the TOP_K_SENTENCES
        highest scored sentences (for that category).
      - Final article score = mean of those top-k sentence scores.

    Returns
    -------
    DataFrame with columns:
      [article_id_col] + one column per category.
    """
    category_cols = [
        c for c in sent_scores_df.columns
        if c not in [article_id_col, "sentence_id"]
    ]

    article_records: List[Dict] = []
    grouped = sent_scores_df.groupby(article_id_col)

    for article_id, group in tqdm(grouped, desc="Aggregating to article level"):
        rec = {article_id_col: article_id}
        for cat in category_cols:
            scores = group[cat].dropna().sort_values(ascending=False)
            if len(scores) == 0:
                rec[cat] = float("nan")
            else:
                k = min(TOP_K_SENTENCES, len(scores))
                rec[cat] = float(scores.iloc[:k].mean())
        article_records.append(rec)

    article_scores_df = pd.DataFrame(article_records)
    return article_scores_df


# ---------------------------------------------------------------------
# 6. Article-level classification rules (Stage 1)
# ---------------------------------------------------------------------


def apply_classification_rules(
    scores_df: pd.DataFrame,
    threshold: float,
    delta: float,
) -> pd.DataFrame:
    """
    Apply initial multi-labeling and ambiguity rules on article-level scores:

    - If no category >= threshold → 'Low Confidence'
    - If exactly one category >= threshold → 'Single Label'
    - If multiple categories >= threshold:
        - If (top1 - top2) < delta → 'Complex Event (High Confidence, Low Delta)'
          and select all categories within top1 - delta
        - Else → 'Complex Event (High Confidence, High Delta)' with top 2 categories
    """
    print("Applying initial article-level classification rules...")

    provisional_labels: List[str] = []
    provisional_categories_list: List[str] = []

    for _, row in tqdm(scores_df.iterrows(), total=len(scores_df), desc="Classifying Articles"):
        # drop NaNs
        numeric_row = row.dropna()
        # remove non-category columns if present
        numeric_row = numeric_row[
            [c for c in numeric_row.index if c not in ["article_id"]]
        ]

        high_scores = numeric_row[numeric_row >= threshold].sort_values(ascending=False)

        if len(high_scores) == 0:
            provisional_labels.append("Low Confidence")
            provisional_categories_list.append(None)

        elif len(high_scores) == 1:
            provisional_labels.append("Single Label")
            provisional_categories_list.append(high_scores.index[0])

        else:
            top_score = high_scores.iloc[0]
            second_score = high_scores.iloc[1]
            if (top_score - second_score) < delta:
                provisional_labels.append("Complex Event (High Confidence, Low Delta)")
                close_categories = high_scores[high_scores >= top_score - delta]
                provisional_categories_list.append(", ".join(close_categories.index))
            else:
                provisional_labels.append("Complex Event (High Confidence, High Delta)")
                provisional_categories_list.append(", ".join(high_scores.index[:2]))

    result = scores_df.copy()
    result["provisional_label"] = provisional_labels
    result["provisional_categories"] = provisional_categories_list

    return result


# ---------------------------------------------------------------------
# 7. Final tiered category selection (Stage 2) - using SHORT names
# ---------------------------------------------------------------------

# New short-name tiers
TIER_1 = ["Governance", "Personnel"]
TIER_2 = ["Products", "IT/Data", "Processes"]
TIER_3 = ["Legal", "Finance", "Communication"]
ORDERED_CATEGORIES = TIER_1 + TIER_2 + TIER_3


def _parse_categories(categories_str) -> List[str]:
    if categories_str is None or (isinstance(categories_str, float) and np.isnan(categories_str)):
        return []
    return [c.strip() for c in str(categories_str).split(",") if c.strip()]


def get_final_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the final category based on a tiered priority system,
    using the short category names:

    1. If provisional_label is Low Confidence → final_category = None.
    2. If Single Label → final_category = that category.
    3. If Complex Event:
        - Prefer Tier 1 categories if present.
        - Else prefer Tier 2.
        - Else Tier 3.
    """
    print("Applying tiered priority system for final category selection...")

    def find_final_category(row):
        label = row["provisional_label"]
        categories_str = row["provisional_categories"]

        if label == "Low Confidence" or categories_str is None:
            return None

        # Single label: just return it
        if label == "Single Label":
            return str(categories_str).strip()

        # Complex event: parse categories
        potential_categories = _parse_categories(categories_str)

        if not potential_categories:
            return None

        # First, check Tier 1
        for cat in TIER_1:
            if cat in potential_categories:
                return cat

        # Then Tier 2
        for cat in TIER_2:
            if cat in potential_categories:
                return cat

        # Finally Tier 3
        for cat in TIER_3:
            if cat in potential_categories:
                return cat

        # Fallback: first category if none matched (should not happen)
        return potential_categories[0]

    df = df.copy()
    df["final_category"] = df.apply(find_final_category, axis=1)
    return df


# ---------------------------------------------------------------------
# 8. High-level wrapper: classify_articles (sentence → article)
# ---------------------------------------------------------------------


def classify_articles(
    df_articles: pd.DataFrame,
    keyword_dict_path: str,
    model_name: str = "BAAI/bge-m3",
    device: str = "cuda",
    text_col: str = "text",
    article_id_col: str = "article_id",
    threshold: float = 0.4,
    delta: float = 0.04,
) -> pd.DataFrame:
    """
    Full two-stage classification pipeline:

    Stage 1 (Sentence level)
      - split articles into sentences
      - encode sentences with BGE
      - compute sentence-level category scores
        (mean over top-8 most similar keywords)

    Stage 2 (Article level)
      - aggregate per article and category
        (mean over top-10 highest-scoring sentences)
      - apply threshold + delta rules
      - apply tiered priority rules
      - override 'skip_article' via rule-based blacklist to 'No Event'
    """
    # Copy so we don't mutate caller's dataframe
    df_articles = df_articles.copy()

    # Step 0: apply rule-based skip rules (blacklist)
    print("Applying rule-based skip (No Event) checks...")
    df_articles["skip_article"] = df_articles[text_col].apply(should_skip_article)

    # Step 1: load resources
    print("Loading keyword dictionary...")
    keyword_dict = load_keyword_dictionary(keyword_dict_path)
    print(f"Loaded {len(keyword_dict)} categories (short names): {list(keyword_dict.keys())}")

    print("Loading sentence-transformer model...")
    model = load_sbert_model(model_name, device=device)

    # Step 2: sentence splitting and encoding
    print("Splitting articles into sentences...")
    sent_df = split_articles_into_sentences(
        df_articles, text_col=text_col, article_id_col=article_id_col
    )
    if sent_df.empty:
        raise ValueError("No sentences extracted. Check that text_col contains text data.")

    print("Encoding sentences...")
    sentence_emb = encode_sentences(sent_df, model, sentence_col="sentence_text")

    print("Encoding keywords...")
    keyword_emb = encode_keywords(keyword_dict, model)

    # Step 3: sentence-level scores
    print("Computing sentence–category scores...")
    sent_scores_df = compute_sentence_category_scores(
        sentence_embeddings=sentence_emb,
        df_sentences=sent_df,
        keyword_embeddings=keyword_emb,
        article_id_col=article_id_col,
    )

    # Step 4: aggregate to article level
    print("Aggregating sentence scores to article-level scores...")
    article_scores_df = aggregate_sentence_scores_to_article(
        sent_scores_df, article_id_col=article_id_col
    )

    # Step 5: apply article-level classification rules
    print("Applying article-level classification rules...")
    classified_scores = apply_classification_rules(
        article_scores_df, threshold=threshold, delta=delta
    )

    # Step 6: final category selection
    print("Applying final tiered rules...")
    classified_scores = get_final_category(classified_scores)

    # Step 7: attach article_id and merge back
    classified_scores[article_id_col] = article_scores_df[article_id_col].values
    out = df_articles.merge(classified_scores, on=article_id_col, how="left")

    # Step 8: override final category for skipped articles to 'No Event'
    mask_skip = out["skip_article"].astype(bool)
    if mask_skip.any():
        out.loc[mask_skip, "final_category"] = "No Event"

    return out
