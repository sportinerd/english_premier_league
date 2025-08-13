import pandas as pd
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
import uvicorn
import os

# ==============================================================================
# 1. CONFIGURATION SETTINGS
# ==============================================================================

# --- MongoDB Connection Settings for EPL Data ---
INPUT_DB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
INPUT_DB_NAME = "English_premier_league"
EPL_ODDS_COLLECTION = "epl"
SPORTSMONK_FIXTURES_COLLECTION = "sportsmonk_fixture"

# --- Constants for FDR Calculation ---
OUTRIGHT_COMPONENT_WEIGHTS = {
    'base_strength_from_odds': 0.70,
    'venue_impact': 0.20,
    'fatigue': 0.10
}
FINAL_FDR_WEIGHTS = {
    'outright': 0.40,
    'correct_score': 0.60
}

# --- Constants for xG and AFD/DFD Calculations ---
TOTAL_XG_FOR_MATCH_FROM_STRENGTHS = 2.7
HOME_ADVANTAGE_XG_MULTIPLIER = 1.15
AWAY_DISADVANTAGE_XG_MULTIPLIER = 0.85
MIN_XG_PER_TEAM = 0.2
MAX_EXPECTED_XG_PER_TEAM = 3.5

# ==============================================================================
# 2. DYNAMIC DATA HANDLING & MAPPINGS
# ==============================================================================

EPL_TEAM_NAME_MAPPING = {
    # Odds Data Name : SportsMonk Official Name
    "Man City": "Manchester City", "Man United": "Manchester United", "Man Utd": "Manchester United",
    "Spurs": "Tottenham Hotspur", "Tottenham": "Tottenham Hotspur", "Newcastle": "Newcastle United",
    "West Ham": "West Ham United", "Brighton": "Brighton & Hove Albion", "Wolves": "Wolverhampton Wanderers",
    "Nottm Forest": "Nottingham Forest", "Sheff Utd": "Sheffield United", "Bournemouth": "AFC Bournemouth",
    "Leeds": "Leeds United", "Liverpool": "Liverpool", "Aston Villa": "Aston Villa",
    "Fulham": "Fulham", "Brentford": "Brentford", "Burnley": "Burnley",
    "Arsenal": "Arsenal", "Chelsea": "Chelsea", "Crystal Palace": "Crystal Palace",
    "Everton": "Everton", "Sunderland": "Sunderland",
}

def get_all_teams_from_data(sportsmonk_fixtures):
    team_names = set(fix['home_team_name'] for fix in sportsmonk_fixtures if 'home_team_name' in fix)
    team_names.update(fix['away_team_name'] for fix in sportsmonk_fixtures if 'away_team_name' in fix)
    print(f"API: Discovered {len(team_names)} unique teams from the SportsMonk data source.")
    return list(team_names)

def generate_team_details(team_list):
    team_details = {}
    for i, team_name in enumerate(team_list):
        short_code_parts = [word[0] for word in team_name.split() if word[0].isupper()]
        short_code = "".join(short_code_parts) if short_code_parts else team_name[:3].upper()
        api_id = 1000 + i
        team_details[team_name] = {"short_code": short_code, "api_id": api_id, "image": "https://placeholder.com/logo.png"}
    return team_details

DEFAULT_TEAM_DETAIL = {"short_code": "N/A", "api_id": None, "image": "https://placeholder.com/logo.png"}

# ==============================================================================
# 3. DATA LOADING AND PARSING FUNCTIONS
# ==============================================================================

def convert_american_to_decimal(american_odds_str):
    if not isinstance(american_odds_str, str) or not american_odds_str: return None
    try:
        val = float(american_odds_str.replace('+', ''))
        return (val / 100) + 1 if val > 0 else (100 / abs(val)) + 1
    except (ValueError, TypeError): return None

def load_data_from_mongo(collection_name):
    try:
        client = MongoClient(INPUT_DB_URI)
        db = client[INPUT_DB_NAME]
        data = list(db[collection_name].find({}))
        client.close()
        if not data: raise HTTPException(status_code=404, detail=f"No data in '{INPUT_DB_NAME}.{collection_name}'.")
        print(f"API: ✅ Loaded {len(data)} documents from '{INPUT_DB_NAME}.{collection_name}'.")
        return data[0] if len(data) == 1 and collection_name == EPL_ODDS_COLLECTION else data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB error: {e}")

def parse_match_name(match_name, team_map):
    for d in [' v ', ' vs ', ' - ']:
        if d in match_name:
            h_raw, a_raw = [p.strip() for p in match_name.split(d, 1)]
            return team_map.get(h_raw, h_raw), team_map.get(a_raw, a_raw)
    return "Unknown", "Unknown"

def map_fixtures_and_odds(epl_data, sportsmonk_fixtures, team_map):
    sportsmonk_lookup = {}
    for fix in sportsmonk_fixtures:
        start_time = fix.get('starting_at')
        if not start_time: continue
        st_obj = datetime.fromisoformat(start_time['$date'].replace('Z', '+00:00')) if isinstance(start_time, dict) else start_time
        if st_obj:
            fix['parsed_start_time'] = st_obj
            sportsmonk_lookup[(fix['home_team_name'], fix['away_team_name'], st_obj.strftime('%Y-%m-%d'))] = fix

    mapped_fixtures = []
    for match_id, odds in epl_data.get('matches', {}).items():
        h, a = parse_match_name(odds.get('match_name', ''), team_map)
        st_str = odds.get('start_time')
        if h == "Unknown" or not st_str: continue
        try:
            date_obj = datetime.fromisoformat(st_str.replace('Z', '+00:00'))
            key = (h, a, date_obj.strftime('%Y-%m-%d'))
            s_fix = sportsmonk_lookup.get(key)
            if s_fix:
                p_time = s_fix['parsed_start_time']
                mapped_fixtures.append({
                    'fixture_id': s_fix.get('fixture_id'), 'home_team_id': s_fix.get('home_team_id'),
                    'away_team_id': s_fix.get('away_team_id'), 'home_team': s_fix['home_team_name'],
                    'away_team': s_fix['away_team_name'], 'date': p_time.date(), 'time': p_time.time(),
                    'starting_at': p_time.isoformat() + "Z", 'gameweek': s_fix.get('GW'),
                    'correct_score_odds': odds.get('markets', {}).get('Correct Score')
                })
        except ValueError: continue
    print(f"API: ✅ Successfully mapped {len(mapped_fixtures)} fixtures with odds.")
    return mapped_fixtures

def extract_outright_odds_from_data(epl_data, team_map):
    odds_data = []
    outright_key = next(iter(epl_data.get('outright_winner', {})), None)
    if not outright_key: return pd.DataFrame()
    for team, american in epl_data['outright_winner'][outright_key].items():
        dec = convert_american_to_decimal(american)
        if dec: odds_data.append({'team_name': team_map.get(team, team), 'decimal_odds': dec})
    return pd.DataFrame(odds_data)

# ==============================================================================
# 4. CORE CALCULATION LOGIC
# ==============================================================================
TIER_DISPLAY_MAPPING = {"level1":"Very Easy", "level2":"Easy", "level3":"Average Difficulty", "level4":"Difficult", "level5":"Very Difficult"}
STANDARD_DIFFICULTY_TIERS = [
    {"threshold": 20, "name": TIER_DISPLAY_MAPPING["level1"], "color": "#1A651A"},
    {"threshold": 35, "name": TIER_DISPLAY_MAPPING["level2"], "color": "#8FBC8F"},
    {"threshold": 50, "name": TIER_DISPLAY_MAPPING["level3"], "color": "#D3D3D3"},
    {"threshold": 65, "name": TIER_DISPLAY_MAPPING["level4"], "color": "#F08080"},
    {"threshold": 100, "name": TIER_DISPLAY_MAPPING["level5"], "color": "#A52A2A"}
]
TIER_DETAILS_BY_NAME = {tier['name']: tier for tier in STANDARD_DIFFICULTY_TIERS}
DEFAULT_TIER_NAME, DEFAULT_TIER_COLOR = "N/A", "#FFFFFF"

def normalize_tournament_implied_probs(df_odds, all_teams):
    if df_odds.empty: return {team: 10.0 for team in all_teams}
    df_odds['implied_prob'] = 1 / df_odds['decimal_odds']
    total_prob = df_odds['implied_prob'].sum()
    df_odds['norm_prob'] = df_odds['implied_prob'] / total_prob if total_prob > 0 else 0
    min_n, max_n = df_odds['norm_prob'].min(), df_odds['norm_prob'].max()
    df_odds['strength'] = 10 + 90 * (df_odds['norm_prob'] - min_n) / (max_n - min_n) if (max_n - min_n) > 1e-9 else 55
    strengths = pd.Series(df_odds.strength.values, index=df_odds.team_name).to_dict()
    for team in all_teams:
        if team not in strengths: strengths[team] = 10.0
    return strengths

def calculate_fatigue_impact(match_date, last_match):
    if not last_match: return 0
    days = (match_date - last_match['date']).days
    if days >= 7: return -10
    if days >= 5: return -5
    if days >= 3: return 0
    return 15 if days < 2 else 8

def calculate_outright_fdr(fixture, strengths, history):
    h, a, dt = fixture['home_team'], fixture['away_team'], fixture['date']
    h_str, a_str = strengths.get(h, 10.0), strengths.get(a, 10.0)
    fat_h = calculate_fatigue_impact(dt, history.get(h))
    fat_a = calculate_fatigue_impact(dt, history.get(a))
    h_fdr = OUTRIGHT_COMPONENT_WEIGHTS['base_strength_from_odds'] * a_str + OUTRIGHT_COMPONENT_WEIGHTS['fatigue'] * fat_h
    a_fdr = OUTRIGHT_COMPONENT_WEIGHTS['base_strength_from_odds'] * h_str + OUTRIGHT_COMPONENT_WEIGHTS['fatigue'] * fat_a
    return {'home_fdr': max(1, min(99, h_fdr)), 'away_fdr': max(1, min(99, a_fdr)), 'h_str': h_str, 'a_str': a_str}

def get_cs_fdr_and_probs(cs_odds_american):
    if not cs_odds_american: return 50.0, 50.0, 0.333, 0.334, 0.333
    valid_odds, total_inv = [], 0.0
    for score, odd_str in cs_odds_american.items():
        odd = convert_american_to_decimal(odd_str)
        if odd and odd > 1.0:
            try:
                h_g, a_g = map(int, score.split('-'))
                inv = 1.0 / odd
                valid_odds.append({'h_g': h_g, 'a_g': a_g, 'inv': inv})
                total_inv += inv
            except ValueError: continue
    if total_inv <= 1e-6: return 50.0, 50.0, 0.333, 0.334, 0.333
    P_h, P_d, P_a = 0.0, 0.0, 0.0
    for item in valid_odds:
        prob = item['inv'] / total_inv
        if item['h_g'] > item['a_g']: P_h += prob
        elif item['h_g'] < item['a_g']: P_a += prob
        else: P_d += prob
    return max(0, 100 * (1 - (3 * P_h + P_d) / 3)), max(0, 100 * (1 - (3 * P_a + P_d) / 3)), P_h, P_d, P_a

def calculate_top_score_predictions(cs_odds_american):
    if not cs_odds_american: return []
    scores, total_inv = [], 0
    for score, odd_str in cs_odds_american.items():
        odd = convert_american_to_decimal(odd_str)
        if odd and odd > 1.0:
            inv = 1.0 / odd
            scores.append({'score': score, 'inv': inv})
            total_inv += inv
    if total_inv <= 1e-6: return []
    for s in scores: s['probability'] = s['inv'] / total_inv
    top_5 = sorted(scores, key=lambda x: x['probability'], reverse=True)[:5]
    return [{'score': item['score'], 'probability': f"{item['probability']*100:.2f}%"} for item in top_5]

def get_afd_dfd(h_str, a_str):
    if h_str + a_str == 0: return 50.0, 50.0, 50.0, 50.0
    h_base = (h_str / (h_str + a_str)) * TOTAL_XG_FOR_MATCH_FROM_STRENGTHS * HOME_ADVANTAGE_XG_MULTIPLIER
    a_base = (a_str / (h_str + a_str)) * TOTAL_XG_FOR_MATCH_FROM_STRENGTHS * AWAY_DISADVANTAGE_XG_MULTIPLIER
    h_xg = max(MIN_XG_PER_TEAM, h_base)
    a_xg = max(MIN_XG_PER_TEAM, a_base)
    return tuple(round(max(0, min(100, v)), 1) for v in [100*(1-h_xg/MAX_EXPECTED_XG_PER_TEAM), 100*(a_xg/MAX_EXPECTED_XG_PER_TEAM),
                                                        100*(1-a_xg/MAX_EXPECTED_XG_PER_TEAM), 100*(h_xg/MAX_EXPECTED_XG_PER_TEAM)])

def get_competitiveness(h_fdr, a_fdr):
    if h_fdr is None or pd.isna(h_fdr): return (DEFAULT_TIER_NAME, DEFAULT_TIER_COLOR), (DEFAULT_TIER_NAME, DEFAULT_TIER_COLOR), 0.0, "Unknown"
    diff = abs(h_fdr - a_fdr)
    if diff <= 6: label, low, high = "Minimal", "level4", "level4"
    elif diff <= 15: label, low, high = "Medium", "level3", "level4"
    elif diff <= 30: label, low, high = "Large", "level2", "level5"
    else: label, low, high = "Massive", "level1", "level5"
    ld, hd = TIER_DETAILS_BY_NAME[TIER_DISPLAY_MAPPING[low]], TIER_DETAILS_BY_NAME[TIER_DISPLAY_MAPPING[high]]
    h_tier, a_tier = (((ld['name'], ld['color']), (hd['name'], hd['color'])) if h_fdr <= a_fdr else ((hd['name'], hd['color']), (ld['name'], ld['color'])))
    return h_tier, a_tier, round(diff, 1), label

# ==============================================================================
# 5. MAIN PROCESSING & ORCHESTRATION (REFACTORED)
# ==============================================================================

def process_all_epl_data():
    print("API: --- Starting Shared EPL Data Processing ---")
    epl_odds_data = load_data_from_mongo(EPL_ODDS_COLLECTION)
    s_fixtures = load_data_from_mongo(SPORTSMONK_FIXTURES_COLLECTION)
    all_teams = get_all_teams_from_data(s_fixtures)
    team_details = generate_team_details(all_teams)
    base_fixtures = map_fixtures_and_odds(epl_odds_data, s_fixtures, EPL_TEAM_NAME_MAPPING)
    
    df_outright = extract_outright_odds_from_data(epl_odds_data, EPL_TEAM_NAME_MAPPING)
    strengths = normalize_tournament_implied_probs(df_outright, all_teams)
    
    fixtures_sorted = sorted(base_fixtures, key=lambda x: (x['date'], x['time']))
    history_tracker = {team: None for team in all_teams}
    
    results = []
    print("\nAPI: --- Calculating metrics for each mapped fixture ---")
    for fix in fixtures_sorted:
        h, a = fix['home_team'], fix['away_team']
        history = {h: history_tracker[h], a: history_tracker[a]}
        history_tracker[h] = history_tracker[a] = {'date': fix['date']}

        outright_calcs = calculate_outright_fdr(fix, strengths, history)
        cs_odds = fix.get('correct_score_odds')
        top_5_scores = calculate_top_score_predictions(cs_odds)
        
        if cs_odds:
            h_fdr_cs, a_fdr_cs, p_h, p_d, p_a = get_cs_fdr_and_probs(cs_odds)
            h_fdr = FINAL_FDR_WEIGHTS['outright'] * outright_calcs['home_fdr'] + FINAL_FDR_WEIGHTS['correct_score'] * h_fdr_cs
            a_fdr = FINAL_FDR_WEIGHTS['outright'] * outright_calcs['away_fdr'] + FINAL_FDR_WEIGHTS['correct_score'] * a_fdr_cs
            method = "Combined"
        else:
            h_fdr, a_fdr = outright_calcs['home_fdr'], outright_calcs['away_fdr']
            p_h, p_d, p_a = None, None, None
            method = "Outright Only"

        (h_afd, h_dfd, a_afd, a_dfd) = get_afd_dfd(outright_calcs['h_str'], outright_calcs['a_str'])
        h_details, a_details = team_details.get(h, DEFAULT_TEAM_DETAIL), team_details.get(a, DEFAULT_TEAM_DETAIL)
        (h_tier_d, a_tier_d, diff, comp) = get_competitiveness(h_fdr, a_fdr)
        
        results.append({
            'fixture_id': fix['fixture_id'], 'gameweek': fix['gameweek'], 'starting_at': fix['starting_at'],
            'home_team_id': fix['home_team_id'], 'home_team_name': h, 'away_team_id': fix['away_team_id'], 'away_team_name': a,
            'home_team_short_code': h_details['short_code'], 'home_team_logo': h_details['image'],
            'away_team_short_code': a_details['short_code'], 'away_team_logo': a_details['image'],
            'final_home_fdr': round(h_fdr, 1), 'home_fdr_tier': h_tier_d[0], 'home_fdr_tier_color': h_tier_d[1],
            'final_away_fdr': round(a_fdr, 1), 'away_fdr_tier': a_tier_d[0], 'away_fdr_tier_color': a_tier_d[1],
            'fdr_diff': diff, 'match_competitiveness_tier': comp, 'calculation_method': method,
            'prob_home_win': round(p_h, 3) if p_h else None, 'prob_draw': round(p_d, 3) if p_d else None, 'prob_away_win': round(p_a, 3) if p_a else None,
            'home_afd': h_afd, 'home_dfd': h_dfd, 'away_afd': a_afd, 'away_dfd': a_dfd,
            'top_5_score_predictions': top_5_scores
        })
    print(f"\nAPI: --- Processing Finished. Generated data for {len(results)} fixtures. ---")
    return results

# ==============================================================================
# 6. FASTAPI APPLICATION SETUP
# ==============================================================================
app = FastAPI(
    title="EPL Analysis API",
    description="Provides Fantasy Difficulty Ratings (FDR) and match predictions for the English Premier League.",
    version="2.0.0"
)

@app.get("/epl-fdr-results", summary="Get detailed FDR and team strength metrics for all fixtures")
async def get_epl_fdr_data():
    try:
        all_data = process_all_epl_data()
        # Return all calculated data for the FDR endpoint
        return all_data
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred in FDR endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during FDR calculation.")

@app.get("/epl-top-score-predictions", summary="Get top 5 score predictions for all fixtures")
async def get_epl_match_predictions():
    try:
        all_data = process_all_epl_data()
        # Tailor the response for the predictions endpoint
        prediction_results = []
        for match in all_data:
            prediction_results.append({
                'fixture_id': match['fixture_id'], 'gameweek': match['gameweek'], 'starting_at': match['starting_at'],
                'home_team_id': match['home_team_id'], 'home_team_name': match['home_team_name'],
                'away_team_id': match['away_team_id'], 'away_team_name': match['away_team_name'],
                'home_team_short_code': match['home_team_short_code'], 'home_team_logo': match['home_team_logo'],
                'away_team_short_code': match['away_team_short_code'], 'away_team_logo': match['away_team_logo'],
                'top_5_score_predictions': match['top_5_score_predictions']
            })
        return prediction_results
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred in predictions endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction calculation.")

if __name__ == "__main__":
    uvicorn.run("fdr:app", host="0.0.0.0", port=8000, reload=True)