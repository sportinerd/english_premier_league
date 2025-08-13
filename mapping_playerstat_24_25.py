
from __future__ import annotations
import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from fuzzywuzzy import fuzz
from pymongo import MongoClient
from bson import ObjectId
import os
from dotenv import load_dotenv
load_dotenv()
# --- MongoDB Connection Details ---
INPUT_DB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

# Source DB / Collections
SRC_DB_NAME = "English_premier_league"
CSV_COLL_NAME = "player_stat_24_25"
JSON_COLL_NAME = "mapped_fantasy_player"

# Output (mapped) target collection in same DB
OUT_MAPPED_COLL_NAME = "player_stat_24_25_mapped"

# Unmapped go to a separate DB (per your instruction: "separate db")
UNMAPPED_DB_NAME = "English_premier_league"
UNMAPPED_COLL_NAME = "player_stat_24_25_unmapped"

# --- Utility: text normalization ---
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")  # remove punctuation except word chars/space

def strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u00a0", " ")
    s = strip_accents(s)
    s = s.replace("-", " ")
    s = s.replace(".", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    return s

# --- Position normalization ---
CSV_POS_TO_GROUP = {"DEF": "DEF", "MID": "MID", "FWD": "FWD", "GK": "GK"}
JSON_FULL_TO_GROUP = {
    "defender": "DEF",
    "midfielder": "MID",
    "attacker": "FWD",
    "forward": "FWD",
    "goalkeeper": "GK",
}

def normalize_csv_pos(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    p = p.strip().upper()
    return CSV_POS_TO_GROUP.get(p, None)

def normalize_json_pos(position: Optional[str], position_group: Optional[str]) -> Optional[str]:
    if position_group and position_group.strip():
        return position_group.strip().upper()
    if position and position.strip():
        return JSON_FULL_TO_GROUP.get(position.strip().lower(), None)
    return None

# --- Team normalization ---
TEAM_SYNONYMS = {
    "manchester united": "manchester united",
    "man united": "manchester united",
    "man utd": "manchester united",
    "mufc": "manchester united",
    "manchester city": "manchester city",
    "man city": "manchester city",
    "mcfc": "manchester city",
    "city": "manchester city",  # risky but commonly used
    "tottenham hotspur": "tottenham hotspur",
    "spurs": "tottenham hotspur",
    "thfc": "tottenham hotspur",
    "arsenal": "arsenal",
    "chelsea": "chelsea",
    "liverpool": "liverpool",
    "west ham": "west ham united",
    "wolverhampton wanderers": "wolverhampton wanderers",
    "wolves": "wolverhampton wanderers",
    "brighton": "brighton & hove albion",
    "newcastle": "newcastle united",
    "aston villa": "aston villa",
    "everton": "everton",
    "crystal palace": "crystal palace",
    "brentford": "brentford",
    "fulham": "fulham",
    "bournemouth": "afc bournemouth",
    "nottingham forest": "nottingham forest",
    "leicester": "leicester city",
}

def normalize_team_name(team: Optional[str]) -> str:
    if not team:
        return ""
    base = norm_text(team)
    base = base.replace(" fc", "").replace(" afc", "").strip()
    return TEAM_SYNONYMS.get(base, base)

# --- Name handling functions ---
def name_candidates_from_csv(name: str) -> List[str]:
    """Generate name candidates from CSV name field."""
    if not name:
        return []
    
    candidates = []
    name = name.strip()
    
    # Add original normalized name
    norm_name = norm_text(name)
    if norm_name:
        candidates.append(norm_name)
    
    # Split by common separators and create variants
    parts = re.split(r'[,\(\)]', name)
    for part in parts:
        part = part.strip()
        if part:
            norm_part = norm_text(part)
            if norm_part and norm_part not in candidates:
                candidates.append(norm_part)
    
    # Handle "First Last" -> "Last" variants
    if ' ' in norm_name:
        words = norm_name.split()
        if len(words) >= 2:
            # Add last name only
            last_name = words[-1]
            if last_name not in candidates:
                candidates.append(last_name)
            
            # Add first name only
            first_name = words[0]
            if first_name not in candidates:
                candidates.append(first_name)
    
    return list(set(candidates))  # Remove duplicates

def name_candidates_from_json(json_doc: dict) -> List[str]:
    """Generate name candidates from JSON document."""
    candidates = []
    
    # Common name fields to check
    name_fields = [
        'common_name', 'name', 'full_name', 'display_name', 
        'first_name', 'last_name', 'short_name', 'player_name'
    ]
    
    for field in name_fields:
        value = json_doc.get(field)
        if value:
            norm_name = norm_text(str(value))
            if norm_name and norm_name not in candidates:
                candidates.append(norm_name)
    
    # Handle web_name if present
    web_name = json_doc.get('web_name')
    if web_name:
        norm_web_name = norm_text(str(web_name))
        if norm_web_name and norm_web_name not in candidates:
            candidates.append(norm_web_name)
    
    # Generate additional variants from full names
    for candidate in list(candidates):
        if ' ' in candidate:
            words = candidate.split()
            if len(words) >= 2:
                # Add last name only
                last_name = words[-1]
                if last_name not in candidates:
                    candidates.append(last_name)
                
                # Add first name only
                first_name = words[0]
                if first_name not in candidates:
                    candidates.append(first_name)
    
    return list(set(candidates))  # Remove duplicates

# Manual aliases for known problematic mappings
MANUAL_ALIASES = {
    # Add manual name mappings here if needed
    # "problematic_name": "correct_name"
}

def apply_manual_alias(name: str) -> str:
    """Apply manual name aliases for known problematic mappings."""
    return MANUAL_ALIASES.get(name, name)

# --- Fuzzy Matching Helpers ---
def best_fuzzy_match(candidate: str, options: List[str]) -> Tuple[Optional[str], float]:
    """Return best fuzzy match and score against candidate using fuzzywuzzy."""
    best_opt, best_score = None, 0.0
    for opt in options:
        score = fuzz.ratio(candidate, opt)
        if score > best_score:
            best_opt, best_score = opt, score
    return best_opt, best_score

# --- Main mapping ---
def main():
    client = MongoClient(INPUT_DB_URI)

    src_db = client[SRC_DB_NAME]
    csv_coll = src_db[CSV_COLL_NAME]
    json_coll = src_db[JSON_COLL_NAME]
    out_mapped_coll = src_db[OUT_MAPPED_COLL_NAME]

    unmapped_db = client[UNMAPPED_DB_NAME]
    unmapped_coll = unmapped_db[UNMAPPED_COLL_NAME]

    # Ensure unique indexes exist for upsert operations
    print("Ensuring unique indexes exist...")
    out_mapped_coll.create_index([("csv__id", 1)], unique=True)
    unmapped_coll.create_index([("csv__id", 1)], unique=True)

    print("Loading JSON documents and building indices...")
    
    # Load JSON documents and build index by team+position
    json_docs = list(json_coll.find({}))
    print(f"Loaded {len(json_docs)} JSON documents")
    
    index_by_team_pos: Dict[Tuple[str, Optional[str]], List[dict]] = {}
    name_index_by_team_pos: Dict[Tuple[str, Optional[str]], Dict[str, List[dict]]] = {}

    for jd in json_docs:
        team_norm = normalize_team_name(jd.get("team_name"))
        pos_group = normalize_json_pos(jd.get("position"), jd.get("position_group"))
        key = (team_norm, pos_group)

        index_by_team_pos.setdefault(key, []).append(jd)

        for nm in name_candidates_from_json(jd):
            name_index_by_team_pos.setdefault(key, {}).setdefault(nm, []).append(jd)

    print("Indices built. Starting mapping process...")

    # Iterate CSV rows and map
    to_upsert_mapped: List[dict] = []
    to_upsert_unmapped: List[dict] = []

    total = csv_coll.count_documents({})
    processed = 0
    mapped = 0
    fuzzy_mapped = 0

    for row in csv_coll.find({}):
        processed += 1

        if processed % 100 == 0:
            print(f"Progress: {processed}/{total} ({processed/total*100:.1f}%)")

        csv_name_raw = row.get("name")
        csv_team_raw = row.get("team")
        csv_pos_raw = row.get("position")

        csv_team = normalize_team_name(csv_team_raw)
        csv_pos_group = normalize_csv_pos(csv_pos_raw)
        csv_name_cands = name_candidates_from_csv(csv_name_raw or "")

        csv_name_cands = [apply_manual_alias(n) for n in csv_name_cands]

        key = (csv_team, csv_pos_group)
        pool = index_by_team_pos.get(key, [])

        chosen_json = None
        match_mode = None
        match_score = None

        # Exact name match within team+pos
        if pool and csv_name_cands:
            name_map = name_index_by_team_pos.get(key, {})
            for cand in csv_name_cands:
                if cand in name_map:
                    chosen_json = name_map[cand][0]
                    match_mode = "exact"
                    match_score = 1.0
                    break

        # Fuzzy name match within team+pos
        if chosen_json is None and pool and csv_name_cands:
            pool_name_variants = []
            doc_by_variant = {}
            for jd in pool:
                variants = name_candidates_from_json(jd)
                for v in variants:
                    pool_name_variants.append(v)
                    doc_by_variant[v] = jd

            best_overall = (None, 0.0, None)
            for cand in csv_name_cands:
                opt, score = best_fuzzy_match(cand, pool_name_variants)
                if opt is not None and score > best_overall[1]:
                    best_overall = (doc_by_variant[opt], score, opt)

            if best_overall[0] is not None and best_overall[1] >= 90:
                chosen_json = best_overall[0]
                match_mode = "fuzzy"
                match_score = best_overall[1]

        # Last-resort: Ignore team if still unmatched
        if chosen_json is None and csv_pos_group:
            pool2 = [doc for docs in index_by_team_pos.values() 
                    for doc in docs 
                    if normalize_json_pos(doc.get("position"), doc.get("position_group")) == csv_pos_group]
            
            if pool2 and csv_name_cands:
                pool_name_variants = []
                doc_by_variant = {}
                for jd in pool2:
                    variants = name_candidates_from_json(jd)
                    for v in variants:
                        pool_name_variants.append(v)
                        doc_by_variant[v] = jd

                best_overall = (None, 0.0, None)
                for cand in csv_name_cands:
                    opt, score = best_fuzzy_match(cand, pool_name_variants)
                    if opt is not None and score > best_overall[1]:
                        best_overall = (doc_by_variant[opt], score, opt)

                if best_overall[0] is not None and best_overall[1] >= 96:
                    chosen_json = best_overall[0]
                    match_mode = "fuzzy_no_team"
                    match_score = best_overall[1]

        if chosen_json is not None:
            mapped += 1
            if match_mode != "exact":
                fuzzy_mapped += 1

            enriched = dict(row)
            if "_id" in enriched:
                enriched["csv__id"] = enriched.pop("_id")

            json_id: ObjectId = chosen_json.get("_id")
            enriched["fantasy__id"] = json_id
            enriched["api_player_id"] = chosen_json.get("api_player_id")
            enriched["common_name"] = chosen_json.get("common_name")
            enriched["image"] = chosen_json.get("image")
            enriched["team_name_json"] = chosen_json.get("team_name")
            enriched["team_id"] = chosen_json.get("team_id")
            enriched["team_api_id"] = chosen_json.get("team_api_id")
            enriched["team_short_code"] = chosen_json.get("team_short_code")

            enriched["_match"] = {
                "mode": match_mode,
                "score": match_score,
                "team_norm_csv": csv_team,
                "team_norm_json": normalize_team_name(chosen_json.get("team_name")),
                "pos_group_csv": csv_pos_group,
                "pos_group_json": normalize_json_pos(chosen_json.get("position"), chosen_json.get("position_group")),
            }

            to_upsert_mapped.append(enriched)
        else:
            unmapped_doc = dict(row)
            if "_id" in unmapped_doc:
                unmapped_doc["csv__id"] = unmapped_doc.pop("_id")
            unmapped_doc["_unmapped_info"] = {
                "reason": "no_match",
                "team_norm": csv_team,
                "pos_group": csv_pos_group,
                "name_candidates": csv_name_cands,
            }
            to_upsert_unmapped.append(unmapped_doc)

        # Batch upsert to avoid memory issues
        if len(to_upsert_mapped) >= 1000:
            for doc in to_upsert_mapped:
                out_mapped_coll.update_one({'csv__id': doc['csv__id']}, {'$set': doc}, upsert=True)
            to_upsert_mapped.clear()
        if len(to_upsert_unmapped) >= 1000:
            for doc in to_upsert_unmapped:
                unmapped_coll.update_one({'csv__id': doc['csv__id']}, {'$set': doc}, upsert=True)
            to_upsert_unmapped.clear()

    # Upsert remaining documents
    if to_upsert_mapped:
        for doc in to_upsert_mapped:
            out_mapped_coll.update_one({'csv__id': doc['csv__id']}, {'$set': doc}, upsert=True)
    if to_upsert_unmapped:
        for doc in to_upsert_unmapped:
            unmapped_coll.update_one({'csv__id': doc['csv__id']}, {'$set': doc}, upsert=True)

    print(f"\n=== MAPPING RESULTS ===")
    print(f"Processed: {processed}/{total}")
    print(f"Mapped: {mapped} ({mapped/processed*100:.1f}%)")
    print(f"  - Exact matches: {mapped - fuzzy_mapped}")
    print(f"  - Fuzzy matches: {fuzzy_mapped}")
    print(f"Unmapped: {processed - mapped} ({(processed - mapped)/processed*100:.1f}%)")
    print(f"\nOutput collections:")
    print(f"  - {SRC_DB_NAME}.{OUT_MAPPED_COLL_NAME} (updated with {mapped} documents)")
    print(f"  - {UNMAPPED_DB_NAME}.{UNMAPPED_COLL_NAME} (updated with {processed - mapped} documents)")

    client.close()

if __name__ == "__main__":
    main()