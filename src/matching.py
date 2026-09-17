"""
matching.py

Organisation extraction and matching pipeline.

Main steps:
1. Load / clean MSCI (or other) reference universe.
2. Extract ORG entities from news using spaCy.
3. Normalise/clean names.
4. Match against MSCI using exact + fuzzy matching (RapidFuzz).
5. Process large datasets in chunks and save matched vs discarded rows.

Intended to generalise the logic from `01_org_match.ipynb`.
"""

import os
import re
import logging
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
from rapidfuzz import process, fuzz

import spacy


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------
# 1. Name cleaning and global constants
# ---------------------------------------------------------------------

# Legal suffixes / generic words to strip from company names
LEGAL_SUFFIXES = {
    "inc", "inc.", "corp", "corp.", "corporation",
    "co", "co.", "company", "ltd", "ltd.", "llc", "plc",
    "ag", "sa", "nv", "gmbh", "bv", "srl", "ab", "oyj",
    "group", "holdings", "holding", "kk", "pte", "kg",
}

PUNCTUATION_PATTERN = re.compile(r"[^a-z0-9\s]+")

# Global blacklist for *ORG mentions* (things like “state”, “museum”)
GLOBAL_ORG_BLACKLIST = {
    "mets", "hip", "society", "state", "house", "gallery",
    "city", "bank", "academy", "university", "hospital",
    "center", "centre", "museum", "funeral",
}


def clean_name(name: str) -> str:
    """
    Normalise company / organisation names for matching.

    - lowercases
    - removes punctuation
    - strips legal / generic suffixes
    - collapses whitespace
    """
    if not isinstance(name, str):
        return ""

    name = name.lower().strip()
    name = PUNCTUATION_PATTERN.sub(" ", name)
    tokens = [t for t in name.split() if t and t not in LEGAL_SUFFIXES]
    return " ".join(tokens)


# ---------------------------------------------------------------------
# 2. MSCI (or other) universe loading and cleaning
# ---------------------------------------------------------------------

def clean_msci_world(filepath: str) -> pd.DataFrame:
    """
    Clean the iShares MSCI World CSV file.

    File characteristics (based on your screenshot):
    - Delimiter: ;
    - Several metadata rows at the top (dates, fund description, etc.)
    - Real header line starts with: "Ticker;Name;Sector;Asset Class;..."
    - Decimal comma for numeric values (e.g. 211.307.053,72).

    We:
    - Detect the header line starting with "Ticker;"
    - Read from there using sep=";" and decimal=","
    - Keep the columns: Ticker, Name, Sector, Location (others are ignored)
    - Add a cleaned name column: name_clean
    """
    # 1) Find the header row index
    header_prefix = "Ticker;"
    header_idx = None
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.startswith(header_prefix):
                header_idx = i
                break

    if header_idx is None:
        raise ValueError(
            f"Could not find a header line starting with '{header_prefix}' "
            f"in {filepath}."
        )

    # 2) Read the CSV from the header row onward
    #    - skiprows=header_idx so that the header line becomes first row
    #    - sep=';' matches your file
    #    - decimal=',' to parse numbers (even though we mainly use names)
    df = pd.read_csv(
        filepath,
        sep=";",
        skiprows=header_idx,
        header=0,
        decimal=",",
        engine="python",
    )

    # 3) Standardise column names we care about
    #    (they appear as exactly Ticker, Name, Sector, Location in your file)
    rename_map = {}
    for col in df.columns:
        if col.strip().lower() == "ticker":
            rename_map[col] = "ticker"
        elif col.strip().lower() == "name":
            rename_map[col] = "name"
        elif col.strip().lower() == "sector":
            rename_map[col] = "sector"
        elif col.strip().lower() == "location":
            rename_map[col] = "location"

    df = df.rename(columns=rename_map)

    if "name" not in df.columns:
        raise ValueError("Could not find a 'Name' column in MSCI CSV.")

    if "ticker" not in df.columns:
        # not fatal, but useful to know
        logger.warning("No 'Ticker' column found in MSCI CSV.")

    if "location" not in df.columns:
        logger.warning("No 'Location' column found in MSCI CSV.")

    # 4) Drop rows where Name is missing and compute cleaned name
    df["name"] = df["name"].astype(str)
    df["name_clean"] = df["name"].apply(clean_name)

    df = df.dropna(subset=["name_clean"])
    df = df[df["name_clean"] != ""]

    return df


def load_msci_universe(
    filepath: str,
    industry_filter: Optional[List[str]] = None,
    industry_col: str = "sector",
) -> pd.DataFrame:
    """
    Load and (optionally) filter MSCI universe, then add cleaned names.

    Parameters
    ----------
    filepath : str
        Path to MSCI CSV / parquet file.
    industry_filter : list[str] or None
        If provided, keep only these industries (e.g. ["Financials"]).
    industry_col : str
        Column name containing industry / sector information.
        For your CSV this is 'sector'.

    Returns
    -------
    pd.DataFrame
        Must contain at least ['name', 'ticker', 'location', 'name_clean'].
    """
    # If you ever save a cleaned parquet version, we can support that too
    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
        if "name_clean" not in df.columns:
            df["name_clean"] = df["name"].astype(str).apply(clean_name)
    else:
        df = clean_msci_world(filepath)

    # Optional sector/industry filter
    if industry_filter is not None:
        col = industry_col
        if col not in df.columns:
            logger.warning(
                "Requested industry/sector filter on column '%s', "
                "but it does not exist in the MSCI file. Available columns: %s",
                col, list(df.columns),
            )
        else:
            df = df[df[col].isin(industry_filter)].copy()
            logger.info(
                "Filtered MSCI universe to %d rows for %s in column '%s'.",
                len(df), industry_filter, col,
            )

    df = df.dropna(subset=["name_clean"])
    df = df[df["name_clean"] != ""]

    logger.info(
        "Loaded MSCI universe with %d rows and %d unique cleaned names.",
        len(df), df["name_clean"].nunique()
    )
    return df



def build_lookup_structures(msci_df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Build helper structures:

    - name_lookup: clean_name -> {name, ticker, location}
    - name_list: list of all clean_name values (for RapidFuzz)

    Returns
    -------
    (name_lookup, name_list)
    """
    meta_cols = [c for c in ["name", "ticker", "location"] if c in msci_df.columns]
    name_lookup: Dict[str, Dict[str, Any]] = dict(
        zip(
            msci_df["name_clean"],
            msci_df[meta_cols].to_dict("records")
        )
    )
    name_list = list(name_lookup.keys())
    return name_lookup, name_list


# ---------------------------------------------------------------------
# 3. Blacklist loading and ORG extraction
# ---------------------------------------------------------------------

def load_manual_blacklist(path: Optional[str]) -> List[str]:
    """
    Load a manual blacklist of organisation names.

    Accepts CSV (column 'name') or plain text (one name per line).
    """
    if path is None or not os.path.exists(path):
        logger.info("No manual blacklist found at %s.", path)
        return []

    if path.endswith(".csv"):
        df = pd.read_csv(path)
        col = "name" if "name" in df.columns else df.columns[0]
        names = df[col].astype(str).str.lower().tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip().lower() for line in f if line.strip()]

    logger.info("Loaded %d manual blacklist entries.", len(names))
    return names


def load_spacy_model(model_name: str = "en_core_web_trf"):
    """
    Load spaCy model. Fall back to 'en_core_web_sm' if the transformer
    model is not available.
    """
    try:
        nlp = spacy.load(model_name)
    except OSError:
        logger.warning(
            "spaCy model '%s' not found. Falling back to 'en_core_web_sm'. "
            "Run 'python -m spacy download %s' if you want the large model.",
            model_name, model_name
        )
        nlp = spacy.load("en_core_web_sm")
    return nlp


def extract_orgs_batch(
    texts: List[str],
    nlp = None,
    batch_size: int = 64,
    global_blacklist: Optional[set] = None,
    manual_blacklist: Optional[List[str]] = None,
) -> List[List[str]]:
    """
    Extract ORG entities from a list of texts.

    Filters:
    - entity length > 2 characters
    - lowercase form not in global or manual blacklist
    """
    if nlp is None:
        nlp = load_spacy_model()

    if global_blacklist is None:
        global_blacklist = GLOBAL_ORG_BLACKLIST

    manual_set = set(manual_blacklist or [])
    results: List[List[str]] = []

    for doc in nlp.pipe(texts, batch_size=batch_size):
        orgs = {ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"}
        filtered: List[str] = []
        for org in orgs:
            org_lower = org.lower()
            if len(org) <= 2:
                continue
            if org_lower in global_blacklist:
                continue
            if org_lower in manual_set:
                continue
            filtered.append(org)
        results.append(filtered)

    return results


# ---------------------------------------------------------------------
# 4. Matching logic
# ---------------------------------------------------------------------

def find_match(
    org: str,
    name_lookup: Dict[str, Dict[str, Any]],
    name_list: List[str],
    score_cutoff: int = 90,
    partial_cutoff: int = 80,
) -> Dict[str, Any]:
    """
    Match a single ORG mention to the reference universe.

    Strategy:
    1. normalise with clean_name → exact dictionary lookup
    2. if not found, use RapidFuzz.extractOne over name_list
    3. return metadata + match type + status

    Returns dictionary with keys:
    - status: 'MATCH' or 'NO_MATCH'
    - matched_org: cleaned_org_name
    - name, ticker, location (if match)
    - match_type: 'exact', 'fuzzy_strict', 'fuzzy_loose', 'none'
    """
    original_org = org
    cleaned = clean_name(org)

    if not cleaned:
        return {
            "status": "NO_MATCH",
            "source_org": original_org,
            "target_org": None,
            "match_type": "none",
        }

    # Exact match on cleaned name
    if cleaned in name_lookup:
        meta = name_lookup[cleaned].copy()
        meta.update(
            {
                "status": "MATCH",
                "matched_org": cleaned,
                "match_type": "exact",
            }
        )
        return meta

    # Fuzzy match using RapidFuzz
    if not name_list:
        return {
            "status": "NO_MATCH",
            "source_org": original_org,
            "target_org": None,
            "match_type": "none",
        }

    best_match = process.extractOne(
        cleaned,
        name_list,
        scorer=fuzz.WRatio,
        score_cutoff=partial_cutoff,
    )

    if best_match is None:
        return {
            "status": "NO_MATCH",
            "source_org": original_org,
            "target_org": None,
            "match_type": "none",
        }

    best_name, score, _ = best_match
    match_type = "fuzzy_loose"
    if score >= score_cutoff:
        match_type = "fuzzy_strict"

    meta = name_lookup[best_name].copy()
    meta.update(
        {
            "status": "MATCH",
            "matched_org": best_name,
            "match_type": match_type,
            "similarity_score": score,
        }
    )
    return meta


# ---------------------------------------------------------------------
# 5. High-level chunked pipeline
# ---------------------------------------------------------------------

def process_dataset_in_chunks(
    input_df: pd.DataFrame,
    msci_df: pd.DataFrame,
    save_path: str,
    discarded_path: str,
    title_col: str = "title",
    summary_col: str = "summary",
    id_col: str = "article_id",
    keywords_col: str = "keywords",
    source_col: str = "source",
    date_col: str = "published_date",
    blacklist_path: Optional[str] = None,
    nlp_model_name: str = "en_core_web_trf",
    chunk_size: int = 50_000,
    output_format: str = "parquet",
    resume: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main pipeline that:
    - extracts ORGs from title + summary
    - matches to MSCI universe
    - saves matched + discarded rows

    Parameters are adapted to your ECIS pipeline (01_org_match.ipynb).
    """
    logger.info("Preparing lookup structures for MSCI universe...")
    name_lookup, name_list = build_lookup_structures(msci_df)

    manual_blacklist = load_manual_blacklist(blacklist_path)
    nlp = load_spacy_model(nlp_model_name)

    matched_records: List[Dict[str, Any]] = []
    discarded_records: List[Dict[str, Any]] = []

    total_rows = len(input_df)
    logger.info("Starting org-matching over %d rows.", total_rows)

    # Optional resume – if existing output files exist, you could load them and
    # skip processed rows. For simplicity we just ignore 'resume' here; extend if needed.
    if resume:
        logger.warning("`resume=True` requested, but resume logic is not implemented. Starting from scratch.")

    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk = input_df.iloc[start:end].copy()

        # Concatenate title + summary for NER
        texts = (
            chunk[title_col].fillna("").astype(str) + ". " +
            chunk[summary_col].fillna("").astype(str)
        ).tolist()

        orgs_per_article = extract_orgs_batch(
            texts,
            nlp=nlp,
            global_blacklist=GLOBAL_ORG_BLACKLIST,
            manual_blacklist=manual_blacklist,
        )

        for (index, row), orgs in zip(chunk.reset_index().iterrows(), orgs_per_article):
            article_id = row[id_col]
            base_record = {
                "article_id": article_id,
                "title": row[title_col],
                "summary": row[summary_col],
                "source": row.get(source_col),
                "published_date": row.get(date_col),
                "keywords": row.get(keywords_col),
            }

            successful_matches_for_row: List[Dict[str, Any]] = []

            for org in orgs:
                match_result = find_match(org, name_lookup, name_list)

                if match_result["status"] == "MATCH":
                    rec = base_record.copy()
                    rec.update(
                        {
                            "extracted_org": org,
                            "matched_org": match_result["matched_org"],
                            "name": match_result.get("name"),
                            "ticker": match_result.get("ticker"),
                            "location": match_result.get("location"),
                            "match_type": match_result.get("match_type"),
                            "similarity_score": match_result.get("similarity_score"),
                        }
                    )
                    successful_matches_for_row.append(rec)

            if successful_matches_for_row:
                matched_records.extend(successful_matches_for_row)
            else:
                # Nothing matched for this row – store as discarded
                disc = base_record.copy()
                discarded_records.append(disc)

        logger.info("Processed rows %d–%d / %d", start, end, total_rows)

    matched_df = pd.DataFrame(matched_records)
    discarded_df = pd.DataFrame(discarded_records)

    # Save outputs
    if output_format == "csv":
        matched_df.to_csv(save_path, index=False)
        discarded_df.to_csv(discarded_path, index=False)
    else:
        matched_df.to_parquet(save_path, index=False)
        discarded_df.to_parquet(discarded_path, index=False)

    logger.info("Saved %d matched rows to %s", len(matched_df), save_path)
    logger.info("Saved %d discarded rows to %s", len(discarded_df), discarded_path)

    return matched_df, discarded_df
