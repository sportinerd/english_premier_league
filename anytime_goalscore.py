import json
import numpy as np
import os
import sys
import re
from datetime import datetime as datetime_cls
from pymongo import MongoClient
from typing import List, Dict, Any, Optional, Tuple

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# --- MongoDB Connection Settings for your Brazilian Cup Data ---
INPUT_DB_URI = "mongodb://localhost:27017/"
INPUT_DB_NAME = "Brazil_cup"
INPUT_COLLECTION_NAME = "brazil_cup"

# --- Output File ---
OUTPUT_JSON_FILE = "brazilian_cup_player_predictions.json"

# --- Model Tuning Constants ---
PROBABILITY_CAPS = {
    'ags_max': 75.0, 'ags_min': 0, 'aas_max': 60.0, 'aas_min': 0,
    'cs_max': 85.0, 'cs_min': 5.0
}
AVERAGE_TOTAL_GOALS_IN_MATCH = 2.7

# ==============================================================================
# 2. DYNAMIC DATA HANDLING & MAPPINGS (NO HARDCODED DATA)
# ==============================================================================

# Minimal mapping for aliases found *within* the Brazil Cup data itself.
BRAZIL_CUP_TEAM_NAME_MAPPING = {
    "Flamengo": "CR Flamengo", "Palmeiras": "SE Palmeiras", "Sao Paulo": "Sao Paulo FC",
    "Fluminense": "Fluminense FC", "Botafogo RJ": "Botafogo FR", "Internacional": "SC Internacional",
    "Bahia": "EC Bahia", "Atletico MG": "Atletico Mineiro", "Vasco Da Gama": "Vasco da Gama",
    "Bragantino": "Red Bull Bragantino", "Athletico-PR": "Athletico Paranaense",
    "CSA AL": "CSA", "Bragantino SP": "Red Bull Bragantino",
}

# ==============================================================================
# 3. DATA LOADING AND PARSING FUNCTIONS (ADAPTED FOR YOUR DATA)
# ==============================================================================

def convert_american_to_decimal(american_odds_str: str) -> Optional[float]:
    """Converts American odds string (e.g., '+110', '-120') to decimal odds."""
    if not isinstance(american_odds_str, str) or not american_odds_str: return None
    try:
        val = float(american_odds_str.replace('+', ''))
        return (val / 100) + 1 if val > 0 else (100 / abs(val)) + 1
    except (ValueError, TypeError): return None

def load_brazilian_cup_data_from_mongo() -> Dict:
    """Loads the Brazilian Cup data document from MongoDB, handling list or dict format."""
    try:
        client = MongoClient(INPUT_DB_URI)
        raw_data = client[INPUT_DB_NAME][INPUT_COLLECTION_NAME].find_one()
        client.close()
        
        if not raw_data:
            print(f"FATAL: No data found in '{INPUT_DB_NAME}.{INPUT_COLLECTION_NAME}'. Exiting.")
            sys.exit(1)
            
        # FIX: Handle if the data is stored as a list with one element
        if isinstance(raw_data, list) and raw_data:
            cup_data = raw_data[0]
        elif isinstance(raw_data, dict):
            cup_data = raw_data
        else:
            print(f"FATAL: Loaded data is in an unexpected format: {type(raw_data)}. Exiting.")
            sys.exit(1)

        print(f"INFO: ✅ Successfully loaded data from '{INPUT_DB_NAME}.{INPUT_COLLECTION_NAME}'.")
        return cup_data
            
    except Exception as e:
        print(f"FATAL: Could not connect to MongoDB or load data: {e}. Exiting.")
        sys.exit(1)

def parse_match_name(match_name: str, team_map: Dict) -> Tuple[str, str]:
    """Parses a match name string like 'Team A v Team B' into standardized names."""
    if not isinstance(match_name, str): return "Unknown", "Unknown"
    delimiters = [' v ', ' vs ', ' - ']
    for d in delimiters:
        if d in match_name:
            h_raw, a_raw = [p.strip() for p in match_name.split(d, 1)]
            return team_map.get(h_raw, h_raw), team_map.get(a_raw, a_raw)
    return "Unknown", "Unknown"

def extract_all_odds_lookups(cup_data: Dict, team_map: Dict) -> Tuple[Dict, Dict]:
    """Extracts CS, AGS, and AAS odds into structured lookups."""
    cs_lookup, player_odds_lookup = {}, {}
    for match_details in cup_data.get('matches', {}).values():
        h, a = parse_match_name(match_details.get('match_name', ''), team_map)
        if h == "Unknown" or a == "Unknown": continue
        
        date_str = datetime_cls.fromisoformat(match_details['start_time'].replace('Z', '+00:00')).strftime('%Y-%m-%d')
        match_key = (h, a, date_str)
        player_odds_lookup[match_key] = {'ags': {}, 'aas': {}}
        
        markets = match_details.get('markets', {})
        if 'Correct Score' in markets:
            cs_lookup[match_key] = {s: convert_american_to_decimal(o) for s, o in markets['Correct Score'].items() if o}
        if 'Anytime Goalscorer' in markets:
            player_odds_lookup[match_key]['ags'] = {p: convert_american_to_decimal(o) for p, o in markets['Anytime Goalscorer'].items() if o}
        if 'Anytime Assist' in markets:
            player_odds_lookup[match_key]['aas'] = {p: convert_american_to_decimal(o) for p, o in markets['Anytime Assist'].items() if o}
            
    print(f"INFO: ✅ Extracted Correct Score odds for {len(cs_lookup)} matches.")
    print(f"INFO: ✅ Extracted Player (AGS/AAS) odds for {len(player_odds_lookup)} matches.")
    return cs_lookup, player_odds_lookup

# ==============================================================================
# 4. CORE PREDICTION LOGIC (ADAPTED FOR DIRECT ODDS)
# ==============================================================================

def estimate_team_xg(cs_odds: Optional[Dict]) -> Tuple[float, float]:
    """Estimates team xG from Correct Score odds."""
    if not cs_odds: return (AVERAGE_TOTAL_GOALS_IN_MATCH / 2, AVERAGE_TOTAL_GOALS_IN_MATCH / 2)
    
    h_xg, a_xg, total_inv = 0.0, 0.0, 0.0
    items = []
    for score, odd in cs_odds.items():
        if odd and odd > 1.0:
            inv = 1.0 / odd
            try:
                h, a = map(int, score.split('-'))
                items.append({'h': h, 'a': a, 'inv': inv})
                total_inv += inv
            except ValueError:
                continue
    
    if total_inv == 0: return (AVERAGE_TOTAL_GOALS_IN_MATCH / 2, AVERAGE_TOTAL_GOALS_IN_MATCH / 2)
    
    for item in items:
        norm_p = item['inv'] / total_inv
        h_xg += item['h'] * norm_p
        a_xg += item['a'] * norm_p
        
    return h_xg, a_xg

def calculate_player_probabilities(player_name: str, match_odds: Dict) -> Dict:
    """Calculates player probabilities directly from odds provided in the data."""
    
    # --- Anytime Goalscorer (AGS) Calculation ---
    ags_odds = match_odds.get('ags', {}).get(player_name)
    if ags_odds and ags_odds > 1.0:
        ags_prob = (1 / ags_odds) * 100
        ags_prob_source = "direct_odds"
    else:
        ags_prob = 0.0
        ags_prob_source = "no_odds_data"

    # --- Anytime Assist (AAS) Calculation ---
    aas_odds = match_odds.get('aas', {}).get(player_name)
    if aas_odds and aas_odds > 1.0:
        aas_prob = (1 / aas_odds) * 100
        aas_prob_source = "direct_odds"
    else:
        aas_prob = 0.0
        aas_prob_source = "no_odds_data"
        
    return {
        'player_name': player_name,
        'anytime_goalscorer_probability': round(np.clip(ags_prob, PROBABILITY_CAPS['ags_min'], PROBABILITY_CAPS['ags_max']), 2),
        'ags_prob_source': ags_prob_source,
        'anytime_assist_probability': round(np.clip(aas_prob, PROBABILITY_CAPS['aas_min'], PROBABILITY_CAPS['aas_max']), 2),
        'aas_prob_source': aas_prob_source
    }

# ==============================================================================
# 5. MAIN ORCHESTRATION SCRIPT
# ==============================================================================

def main():
    """Main function to orchestrate the entire data processing and prediction pipeline."""
    print("--- Starting Brazilian Cup Player Prediction Script ---")

    cup_data = load_brazilian_cup_data_from_mongo()
    cs_lookup, player_odds_lookup = extract_all_odds_lookups(cup_data, BRAZIL_CUP_TEAM_NAME_MAPPING)

    all_match_predictions = []
    print("\n--- Processing Each Fixture to Calculate Player Probabilities ---")
    
    # Iterate through matches in the order they appear in the data
    for match_id, match_details in cup_data.get('matches', {}).items():
        home_c, away_c = parse_match_name(match_details.get('match_name', ''), BRAZIL_CUP_TEAM_NAME_MAPPING)
        if home_c == "Unknown": continue # Skip if match name is unparsable

        date_str = datetime_cls.fromisoformat(match_details['start_time'].replace('Z', '+00:00')).strftime('%Y-%m-%d')
        match_key = (home_c, away_c, date_str)
        
        print(f"Processing: {home_c} vs {away_c} on {date_str}")
        
        match_player_odds = player_odds_lookup.get(match_key)
        if not match_player_odds:
            print(f"  -> WARNING: No player odds found for this match. Skipping.")
            continue
            
        cs_odds = cs_lookup.get(match_key)
        home_xg, away_xg = estimate_team_xg(cs_odds)
        
        all_players_in_match = set(match_player_odds.get('ags', {}).keys()) | set(match_player_odds.get('aas', {}).keys())

        match_output = {
            "fixture_id": f"{home_c.replace(' ','')}_{away_c.replace(' ','')}_{date_str}",
            "date_str": date_str, "home_team_canonical": home_c, "away_team_canonical": away_c,
            "home_team_xg": round(home_xg, 2), "away_team_xg": round(away_xg, 2),
            "xg_source": "cs_odds" if cs_odds else "default_average",
            "players": []
        }

        for player_name in sorted(list(all_players_in_match)):
            player_stats = calculate_player_probabilities(player_name, match_player_odds)
            match_output["players"].append(player_stats)
            
        all_match_predictions.append(match_output)

    if all_match_predictions:
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_match_predictions, f, indent=4)
        print(f"\n--- SUCCESS ---")
        print(f"Processed {len(all_match_predictions)} matches.")
        print(f"Final predictions saved to '{OUTPUT_JSON_FILE}'.")
    else:
        print("\n--- FAILED ---")
        print("No predictions were generated. Check data source and script logs.")

if __name__ == "__main__":
    main()