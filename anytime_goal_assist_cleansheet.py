import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
import uvicorn
from scipy.stats import poisson
from typing import List, Dict, Any, Optional, Tuple
from bson import ObjectId

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# --- MongoDB Connection Settings ---
INPUT_DB_URI = "mongodb://localhost:27017/"
INPUT_DB_NAME = "English_premier_league"
EPL_ODDS_COLLECTION = "epl"
SPORTSMONK_FIXTURES_COLLECTION = "sportsmonk_fixture"
PLAYER_STAT_COLLECTION = "player_stat_24_25_mapped"

# --- Model Tuning Constants ---
PROBABILITY_CAPS = {
    'ags_max': 75.0, 'ags_min': 0, 'aas_max': 60.0, 'aas_min': 0,
    'cs_max': 85.0, 'cs_min': 5.0
}
AAS_POSITIONAL_MODIFIERS = {
    'Midfielder': 1.4, 'Forward': 0.9, 'Defender': 0.7, 'Goalkeeper': 0.05
}
AGS_POSITIONAL_MODIFIERS = {
    'Forward': 1.2, 'Midfielder': 0.8, 'Defender': 0.35, 'Goalkeeper': 0.02
}
DEFENSIVE_POSITIONS = ["Goalkeeper", "Defender", "DEF"]
AVERAGE_TOTAL_GOALS_IN_MATCH = 2.7
INDIVIDUAL_CS_INFLUENCE_FACTOR = 0.1

# ==============================================================================
# 2. TEAM NAME & PLAYER MAPPINGS
# ==============================================================================

EPL_TEAM_NAME_MAPPING = {
    "Man City": "Manchester City", "Man United": "Manchester United", "Man Utd": "Manchester United",
    "Spurs": "Tottenham Hotspur", "Tottenham": "Tottenham Hotspur", "Newcastle": "Newcastle United",
    "West Ham": "West Ham United", "Brighton": "Brighton & Hove Albion", "Wolves": "Wolverhampton Wanderers",
    "Nottm Forest": "Nottingham Forest", "Sheff Utd": "Sheffield United", "Bournemouth": "AFC Bournemouth",
    "Leeds": "Leeds United", "Liverpool": "Liverpool", "Aston Villa": "Aston Villa", "Fulham": "Fulham",
    "Brentford": "Brentford", "Burnley": "Burnley", "Arsenal": "Arsenal", "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace", "Everton": "Everton", "Sunderland": "Sunderland",
}

def generate_team_details(all_teams: List[str]) -> Dict[str, Dict]:
    team_details = {}
    for i, team_name in enumerate(all_teams):
        short_code_parts = [word[0] for word in team_name.split() if word[0].isupper()]
        short_code = "".join(short_code_parts) if short_code_parts else team_name[:3].upper()
        team_details[team_name] = {
            "team_id": f"TEAM_{1000 + i}",
            "short_code": short_code,
            "api_id": 2000 + i,
            "image": f"https://fantasyfootball.sgp1.cdn.digitaloceanspaces.com/epl-logos/{team_name.replace(' ', '-')}.png"
        }
    return team_details

# ==============================================================================
# 3. DATA LOADING & PREPARATION
# ==============================================================================

def convert_mongo_ids(data: Any) -> Any:
    if isinstance(data, list):
        return [convert_mongo_ids(item) for item in data]
    if isinstance(data, dict):
        return {key: convert_mongo_ids(value) for key, value in data.items()}
    if isinstance(data, ObjectId):
        return str(data)
    return data

def convert_american_to_decimal(american_odds_str: str) -> Optional[float]:
    if not isinstance(american_odds_str, str) or not american_odds_str: return None
    try:
        val = float(american_odds_str.replace('+', ''))
        return (val / 100) + 1 if val > 0 else (100 / abs(val)) + 1
    except (ValueError, TypeError): return None

def load_data_from_mongo(collection_name: str, is_single_doc: bool = False) -> Any:
    try:
        client = MongoClient(INPUT_DB_URI)
        collection = client[INPUT_DB_NAME][collection_name]
        data = collection.find_one() if is_single_doc else list(collection.find({}))
        client.close()
        if data:
            print(f"API: ✅ Loaded data from '{INPUT_DB_NAME}.{collection_name}'.")
            return convert_mongo_ids(data)
        raise HTTPException(status_code=404, detail=f"No data found in '{collection_name}'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB Error for {collection_name}: {e}")

def parse_match_name(match_name: str, team_map: Dict) -> Tuple[str, str]:
    delimiters = [' v ', ' vs ', ' - ']
    for d in delimiters:
        if d in match_name:
            h_raw, a_raw = [p.strip() for p in match_name.split(d, 1)]
            return team_map.get(h_raw, h_raw), team_map.get(a_raw, a_raw)
    return "Unknown", "Unknown"

def load_and_prepare_player_stats() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]], set]:
    player_stats_raw = load_data_from_mongo(PLAYER_STAT_COLLECTION)
    df = pd.DataFrame(player_stats_raw)
    df['team_canonical'] = df['team'].apply(lambda x: EPL_TEAM_NAME_MAPPING.get(x, x))
    for col in ['goals_scored', 'assists', 'clean_sheets']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    team_season_stats = {}
    for team_name, group in df.groupby('team_canonical'):
        def_players = group[group['position'].isin(DEFENSIVE_POSITIONS)]
        num_def_players = len(def_players)
        total_cs = def_players['clean_sheets'].sum()
        team_season_stats[team_name] = {
            "total_goals": group['goals_scored'].sum(),
            "total_assists": group['assists'].sum(),
            "total_clean_sheets": total_cs,
            "avg_cs_per_def_player": (total_cs / num_def_players) if num_def_players > 0 else 0
        }
    players_by_team = {team: group for team, group in df.groupby('team_canonical')}
    all_player_teams = set(df['team_canonical'].unique())
    print(f"API: ✅ Prepared player stats for {len(players_by_team)} teams.")
    return players_by_team, team_season_stats, all_player_teams

def map_fixtures_and_odds(epl_data, sportsmonk_fixtures):
    sportsmonk_lookup = {}
    all_fixture_teams = set()
    for fixture in sportsmonk_fixtures:
        all_fixture_teams.add(fixture['home_team_name'])
        all_fixture_teams.add(fixture['away_team_name'])
        start_time_val = fixture.get('starting_at')
        if not start_time_val: continue

        start_time_obj = None
        if isinstance(start_time_val, datetime):
            start_time_obj = start_time_val
        elif isinstance(start_time_val, str):
            try:
                start_time_obj = datetime.fromisoformat(start_time_val.replace('Z', '+00:00'))
            except ValueError:
                print(f"WARNING: Could not parse date string from SportsMonk: {start_time_val}")
                continue
        
        if start_time_obj:
            if start_time_obj.tzinfo is None:
                start_time_obj = start_time_obj.replace(tzinfo=timezone.utc)
            fixture['parsed_start_time'] = start_time_obj
            date_str = start_time_obj.strftime('%Y-%m-%d')
            sportsmonk_lookup[(fixture['home_team_name'], fixture['away_team_name'], date_str)] = fixture

    mapped_fixtures = []
    for odds_details in epl_data.get('matches', {}).values():
        h_odds, a_odds = parse_match_name(odds_details.get('match_name', ''), EPL_TEAM_NAME_MAPPING)
        time_str = odds_details.get('start_time')
        if h_odds == "Unknown" or not time_str: continue
        
        try:
            date_obj = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            date_str = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            print(f"WARNING: Could not parse date string from odds data: {time_str}")
            continue
        
        sm_fixture = sportsmonk_lookup.get((h_odds, a_odds, date_str))
        if sm_fixture:
            mapped_fixtures.append({
                'fixture_id': sm_fixture.get('fixture_id'), 'home_team_id': sm_fixture.get('home_team_id'),
                'away_team_id': sm_fixture.get('away_team_id'), 'home_team': sm_fixture['home_team_name'],
                'away_team': sm_fixture['away_team_name'], 'date': date_obj.date(), 'time': date_obj.time(),
                'starting_at': sm_fixture['parsed_start_time'].isoformat(), 'gameweek': sm_fixture.get('GW'),
                'odds_details': odds_details,
            })
    print(f"API: ✅ Successfully mapped {len(mapped_fixtures)} fixtures with odds.")
    return mapped_fixtures, all_fixture_teams

# ==============================================================================
# 4. CORE PREDICTION LOGIC
# ==============================================================================

def estimate_team_xg_cs(cs_odds_american: Optional[Dict]) -> Tuple[float, float, float, float]:
    if not cs_odds_american: return AVERAGE_TOTAL_GOALS_IN_MATCH/2, AVERAGE_TOTAL_GOALS_IN_MATCH/2, 30.0, 30.0
    cs_odds = {s: convert_american_to_decimal(o) for s, o in cs_odds_american.items()}
    h_xg, a_xg, h_cs_prob, a_cs_prob, total_inv = 0.0, 0.0, 0.0, 0.0, 0.0
    items = []
    for score, odd in cs_odds.items():
        if odd and odd > 1.0:
            try:
                inv = 1.0 / odd
                h, a = map(int, score.split('-'))
                items.append({'h': h, 'a': a, 'inv': inv}); total_inv += inv
            except (ValueError, TypeError): continue
    if total_inv == 0: return AVERAGE_TOTAL_GOALS_IN_MATCH/2, AVERAGE_TOTAL_GOALS_IN_MATCH/2, 30.0, 30.0
    for item in items:
        norm_p = item['inv'] / total_inv
        h_xg += item['h'] * norm_p; a_xg += item['a'] * norm_p
        if item['a'] == 0: h_cs_prob += norm_p
        if item['h'] == 0: a_cs_prob += norm_p
    return h_xg, a_xg, h_cs_prob * 100, a_cs_prob * 100

def get_position_modifier(position: str, modifier_dict: Dict, default: float) -> float:
    return modifier_dict.get(position, default)

def calculate_player_predictions(player_row: pd.Series, team_stats: Dict, team_xg: float, opp_xg: float, team_cs_prob: float, match_odds: Dict) -> Dict:
    player_name = player_row.get('name')
    position = player_row.get('position')
    
    direct_ags_odds = match_odds.get('Anytime Goalscorer', {}).get(player_name)
    direct_aas_odds = match_odds.get('Anytime Assist', {}).get(player_name)

    ags_prob, aas_prob, ags_prob_source, aas_prob_source = 0.0, 0.0, "no_data", "no_data"

    if direct_ags_odds:
        ags_prob = (1 / convert_american_to_decimal(direct_ags_odds)) * 100; ags_prob_source = "direct_odds"
    if direct_aas_odds:
        aas_prob = (1 / convert_american_to_decimal(direct_aas_odds)) * 100; aas_prob_source = "direct_odds"

    if ags_prob_source == "no_data" and team_stats.get('total_goals', 0) > 0:
        goal_share = player_row['goals_scored'] / team_stats['total_goals']
        pos_modifier = get_position_modifier(position, AGS_POSITIONAL_MODIFIERS, 1.0)
        individual_xg = goal_share * team_xg * pos_modifier
        if individual_xg > 0:
            ags_prob = (1 - poisson.pmf(0, individual_xg)) * 100; ags_prob_source = "poisson_model"
            
    if aas_prob_source == "no_data" and team_stats.get('total_assists', 0) > 0:
        assist_share = player_row['assists'] / team_stats['total_assists']
        pos_modifier = get_position_modifier(position, AAS_POSITIONAL_MODIFIERS, 1.0)
        individual_xa = assist_share * team_xg * 0.8 * pos_modifier
        if individual_xa > 0:
            aas_prob = (1 - poisson.pmf(0, individual_xa)) * 100; aas_prob_source = "poisson_model"
            
    cs_prob, cs_prob_source = 0.0, "not_applicable"
    if position in DEFENSIVE_POSITIONS:
        cs_prob_source = "team_model"
        base_cs_prob = team_cs_prob
        avg_cs = team_stats.get('avg_cs_per_def_player', 0)
        if avg_cs > 0:
            player_cs = player_row['clean_sheets']
            ratio_to_avg = player_cs / avg_cs
            individual_modifier = 1 + ((ratio_to_avg - 1) * INDIVIDUAL_CS_INFLUENCE_FACTOR)
            individual_modifier = np.clip(individual_modifier, 0.9, 1.15)
            base_cs_prob *= individual_modifier
            cs_prob_source = "hybrid_model"
        xg_adjustment_factor = max(0.5, 1 - (opp_xg / (AVERAGE_TOTAL_GOALS_IN_MATCH * 1.5)))
        calculated_cs_prob = base_cs_prob * xg_adjustment_factor
        cs_prob = np.clip(calculated_cs_prob, PROBABILITY_CAPS['cs_min'], PROBABILITY_CAPS['cs_max'])
        
    player_data = player_row.to_dict()
    player_data.update({
        'anytime_goalscorer_probability': round(np.clip(ags_prob, PROBABILITY_CAPS['ags_min'], PROBABILITY_CAPS['ags_max']), 2),
        'ags_prob_source': ags_prob_source,
        'anytime_assist_probability': round(np.clip(aas_prob, PROBABILITY_CAPS['aas_min'], PROBABILITY_CAPS['aas_max']), 2),
        'aas_prob_source': aas_prob_source,
        'clean_sheet_probability': round(cs_prob, 2),
        'cs_prob_source': cs_prob_source
    })
    return player_data

# ==============================================================================
# 5. MAIN ORCHESTRATION SCRIPT
# ==============================================================================

def process_epl_predictions():
    epl_odds_data = load_data_from_mongo(EPL_ODDS_COLLECTION, is_single_doc=True)
    sportsmonk_fixtures = load_data_from_mongo(SPORTSMONK_FIXTURES_COLLECTION)
    players_by_team, team_season_stats, all_player_teams = load_and_prepare_player_stats()

    mapped_fixtures, all_fixture_teams = map_fixtures_and_odds(epl_odds_data, sportsmonk_fixtures)
    all_known_teams = list(all_fixture_teams | all_player_teams)
    team_details_lookup = generate_team_details(all_known_teams)

    all_match_predictions = []
    print("\n--- Processing Each Mapped Fixture ---")

    for fixture in mapped_fixtures:
        home_team, away_team = fixture['home_team'], fixture['away_team']
        print(f"Processing: {home_team} vs {away_team}")
        
        home_details = team_details_lookup.get(home_team, {})
        away_details = team_details_lookup.get(away_team, {})
        
        cs_odds = fixture['odds_details'].get('markets', {}).get('Correct Score')
        home_xg, away_xg, home_cs_prob, away_cs_prob = estimate_team_xg_cs(cs_odds)
        
        match_output = {
            "fixture_id": fixture.get('fixture_id'), "gameweek": fixture.get('gameweek'),
            "starting_at": fixture.get('starting_at'), "home_team_id": fixture.get('home_team_id'),
            "home_team_name": home_team, "away_team_id": fixture.get('away_team_id'),
            "away_team_name": away_team, "home_team_short_code": home_details.get('short_code'),
            "home_team_logo": home_details.get('image'), "away_team_short_code": away_details.get('short_code'),
            "away_team_logo": away_details.get('image'), "home_team_xg": round(home_xg, 2),
            "away_team_xg": round(away_xg, 2), "xg_source": "cs_odds" if cs_odds else "default_average",
            "players": []
        }

        for team_name, team_xg, opp_xg, team_cs_prob in [(home_team, home_xg, away_xg, home_cs_prob), (away_team, away_xg, home_xg, away_cs_prob)]:
            if team_name in players_by_team:
                team_players_df = players_by_team[team_name]
                team_stats = team_season_stats.get(team_name, {"total_goals": 0, "total_assists": 0, "total_clean_sheets": 0, "avg_cs_per_def_player": 0})
                
                for _, player_row in team_players_df.iterrows():
                    player_predictions = calculate_player_predictions(player_row, team_stats, team_xg, opp_xg, team_cs_prob, fixture['odds_details'].get('markets', {}))
                    player_predictions.pop('_id', None); player_predictions.pop('csv__id', None)
                    player_predictions.pop('_match', None); player_predictions.pop('team_canonical', None)
                    match_output["players"].append(player_predictions)
        
        all_match_predictions.append(match_output)
    
    print(f"\n--- SUCCESS: Processed {len(all_match_predictions)} matches. ---")
    return all_match_predictions

# ==============================================================================
# 6. FASTAPI APPLICATION
# ==============================================================================
app = FastAPI(
    title="EPL Player Prediction API",
    description="Calculates AGS, AAS, and CS probabilities for EPL players using a hybrid model.",
    version="2.5.1" # Version bump for the fix
)

@app.get("/epl-player-predictions", summary="Get EPL Player Predictions")
async def get_epl_player_predictions():
    try:
        results = process_epl_predictions()
        return results
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

if __name__ == "__main__":
    uvicorn.run("epl_predictions_api:app", host="0.0.0.0", port=8000, reload=True)