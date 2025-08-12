import os
import sys
import subprocess
import json
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from scipy.stats import poisson
from bson import ObjectId
from contextlib import contextmanager
import backoff
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('epl_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration - using environment variables with defaults
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DB_NAME = os.getenv("DB_NAME", "English_premier_league")

# Team name mapping (common to all scripts)
EPL_TEAM_NAME_MAPPING = {
    "Man City": "Manchester City", "Man United": "Manchester United", "Man Utd": "Manchester United",
    "Spurs": "Tottenham Hotspur", "Tottenham": "Tottenham Hotspur", "Newcastle": "Newcastle United",
    "West Ham": "West Ham United", "Brighton": "Brighton & Hove Albion", "Wolves": "Wolverhampton Wanderers",
    "Nottm Forest": "Nottingham Forest", "Sheff Utd": "Sheffield United", "Bournemouth": "AFC Bournemouth",
    "Leeds": "Leeds United", "Liverpool": "Liverpool", "Aston Villa": "Aston Villa",
    "Fulham": "Fulham", "Brentford": "Brentford", "Burnley": "Burnley",
    "Arsenal": "Arsenal", "Chelsea": "Chelsea", "Crystal Palace": "Crystal Palace",
    "Everton": "Everton", "Sunderland": "Sunderland",
}

# ============================================================================
# PART 1: RUN THE FIRST TWO SCRIPTS SEQUENTIALLY
# ============================================================================

def run_process_all_league_data():
    """Run the process_all_league_data_mongo.py script"""
    logger.info("=== Running process_all_league_data_mongo.py ===")
    try:
        # Set environment to handle UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Run without capturing output to avoid encoding issues
        result = subprocess.run(
            [sys.executable, "process_all_league_data_mongo.py"], 
            check=True, 
            env=env,
            timeout=300  # 5 minute timeout
        )
        logger.info("process_all_league_data_mongo.py completed successfully")
    except subprocess.TimeoutExpired:
        logger.error("Timeout running process_all_league_data_mongo.py")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running process_all_league_data_mongo.py: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error running process_all_league_data_mongo.py: {e}")
        sys.exit(1)

def run_from_cleandata_epl():
    """Run the from_cleandata_epl.py script"""
    logger.info("=== Running from_cleandata_epl.py ===")
    try:
        # Set environment to handle UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Run without capturing output to avoid encoding issues
        result = subprocess.run(
            [sys.executable, "from_cleandata_epl.py"], 
            check=True, 
            env=env,
            timeout=300  # 5 minute timeout
        )
        logger.info("from_cleandata_epl.py completed successfully")
    except subprocess.TimeoutExpired:
        logger.error("Timeout running from_cleandata_epl.py")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running from_cleandata_epl.py: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error running from_cleandata_epl.py: {e}")
        sys.exit(1)

# ============================================================================
# PART 2: COMBINED API FUNCTIONALITY
# ============================================================================

# Context manager for MongoDB connections
@contextmanager
def get_mongo_connection():
    """Context manager for MongoDB connections"""
    client = None
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Test the connection
        client.admin.command('ping')
        yield client
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection error: {e}")
        raise HTTPException(status_code=503, detail="Database service unavailable")
    except PyMongoError as e:
        logger.error(f"MongoDB error: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        if client:
            client.close()

# Common utility functions
def convert_mongo_ids(data):
    """Recursively convert MongoDB ObjectId to string"""
    if isinstance(data, list):
        return [convert_mongo_ids(item) for item in data]
    if isinstance(data, dict):
        return {key: convert_mongo_ids(value) for key, value in data.items()}
    if isinstance(data, ObjectId):
        return str(data)
    return data

@lru_cache(maxsize=1000)
def convert_american_to_decimal(american_odds_str: str) -> Optional[float]:
    """Convert American odds to decimal with caching for performance"""
    if not isinstance(american_odds_str, str) or not american_odds_str:
        return None
    try:
        val = float(american_odds_str.replace('+', ''))
        return (val / 100) + 1 if val > 0 else (100 / abs(val)) + 1
    except (ValueError, TypeError):
        return None

@backoff.on_exception(backoff.expo, PyMongoError, max_tries=3, max_time=30)
def load_data_from_mongo(collection_name: str, db_name: str = DB_NAME, is_single_doc: bool = False) -> Any:
    """Load data from MongoDB with retries and proper error handling"""
    try:
        with get_mongo_connection() as client:
            collection = client[db_name][collection_name]
            data = collection.find_one() if is_single_doc else list(collection.find({}))
            
            if not data:
                logger.warning(f"No data found in '{collection_name}'")
                raise HTTPException(status_code=404, detail=f"No data found in '{collection_name}'.")
            
            logger.info(f"Loaded data from '{db_name}.{collection_name}'")
            return convert_mongo_ids(data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data from MongoDB: {e}")
        raise HTTPException(status_code=500, detail=f"MongoDB Error for {collection_name}: {e}")

def parse_match_name(match_name: str, team_map: Dict) -> Tuple[str, str]:
    """Parse match name into home and away teams using mapping"""
    delimiters = [' v ', ' vs ', ' - ']
    for d in delimiters:
        if d in match_name:
            h_raw, a_raw = [p.strip() for p in match_name.split(d, 1)]
            return team_map.get(h_raw, h_raw), team_map.get(a_raw, a_raw)
    return "Unknown", "Unknown"

# FDR API Functions
def fdr_process_all_epl_data():
    """Process all EPL data for FDR calculation"""
    logger.info("API: --- Starting Shared EPL Data Processing for FDR ---")
    
    # Load data with error handling
    try:
        epl_odds_data = load_data_from_mongo("epl", is_single_doc=True)
        sportsmonk_fixtures = load_data_from_mongo("sportsmonk_fixture")
    except HTTPException:
        raise
    
    # Generate team details
    all_teams = set(fix['home_team_name'] for fix in sportsmonk_fixtures if 'home_team_name' in fix)
    all_teams.update(fix['away_team_name'] for fix in sportsmonk_fixtures if 'away_team_name' in fix)
    
    team_details = {}
    for i, team_name in enumerate(all_teams):
        short_code_parts = [word[0] for word in team_name.split() if word[0].isupper()]
        short_code = "".join(short_code_parts) if short_code_parts else team_name[:3].upper()
        team_details[team_name] = {
            "short_code": short_code, 
            "api_id": 1000 + i, 
            "image": "https://placeholder.com/logo.png"
        }
    
    # Map fixtures and odds
    sportsmonk_lookup = {}
    for fix in sportsmonk_fixtures:
        start_time = fix.get('starting_at')
        if not start_time: 
            continue
            
        try:
            st_obj = datetime.fromisoformat(start_time['$date'].replace('Z', '+00:00')) if isinstance(start_time, dict) else start_time
            if st_obj:
                fix['parsed_start_time'] = st_obj
                sportsmonk_lookup[(fix['home_team_name'], fix['away_team_name'], st_obj.strftime('%Y-%m-%d'))] = fix
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing start time for fixture: {e}")
            continue
    
    mapped_fixtures = []
    for match_id, odds in epl_odds_data.get('matches', {}).items():
        h, a = parse_match_name(odds.get('match_name', ''), EPL_TEAM_NAME_MAPPING)
        st_str = odds.get('start_time')
        if h == "Unknown" or not st_str: 
            continue
            
        try:
            date_obj = datetime.fromisoformat(st_str.replace('Z', '+00:00'))
            key = (h, a, date_obj.strftime('%Y-%m-%d'))
            s_fix = sportsmonk_lookup.get(key)
            if s_fix:
                p_time = s_fix['parsed_start_time']
                mapped_fixtures.append({
                    'fixture_id': s_fix.get('fixture_id'), 
                    'home_team_id': s_fix.get('home_team_id'),
                    'away_team_id': s_fix.get('away_team_id'), 
                    'home_team': s_fix['home_team_name'],
                    'away_team': s_fix['away_team_name'], 
                    'date': p_time.date(), 
                    'time': p_time.time(),
                    'starting_at': p_time.isoformat() + "Z", 
                    'gameweek': s_fix.get('GW'),
                    'correct_score_odds': odds.get('markets', {}).get('Correct Score')
                })
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing match {match_id}: {e}")
            continue
    
    # Extract outright odds
    df_outright = pd.DataFrame()
    outright_key = next(iter(epl_odds_data.get('outright_winner', {})), None)
    if outright_key:
        odds_data = []
        for team, american in epl_odds_data['outright_winner'][outright_key].items():
            dec = convert_american_to_decimal(american)
            if dec: 
                odds_data.append({'team_name': EPL_TEAM_NAME_MAPPING.get(team, team), 'decimal_odds': dec})
        df_outright = pd.DataFrame(odds_data)
    
    # Normalize tournament implied probabilities
    strengths = {}
    if not df_outright.empty:
        df_outright['implied_prob'] = 1 / df_outright['decimal_odds']
        total_prob = df_outright['implied_prob'].sum()
        df_outright['norm_prob'] = df_outright['implied_prob'] / total_prob if total_prob > 0 else 0
        min_n, max_n = df_outright['norm_prob'].min(), df_outright['norm_prob'].max()
        df_outright['strength'] = 10 + 90 * (df_outright['norm_prob'] - min_n) / (max_n - min_n) if (max_n - min_n) > 1e-9 else 55
        strengths = pd.Series(df_outright.strength.values, index=df_outright.team_name).to_dict()
    
    for team in all_teams:
        if team not in strengths: 
            strengths[team] = 10.0
    
    # Calculate FDR for each fixture
    fixtures_sorted = sorted(mapped_fixtures, key=lambda x: (x['date'], x['time']))
    history_tracker = {team: None for team in all_teams}
    results = []
    
    for fix in fixtures_sorted:
        h, a = fix['home_team'], fix['away_team']
        history = {h: history_tracker[h], a: history_tracker[a]}
        history_tracker[h] = history_tracker[a] = {'date': fix['date']}
        
        # Calculate outright FDR
        h_str, a_str = strengths.get(h, 10.0), strengths.get(a, 10.0)
        fatigue_impact_h = 0
        fatigue_impact_a = 0
        if history[h]:
            days = (fix['date'] - history[h]['date']).days
            if days < 2: fatigue_impact_h = 15
            elif days < 3: fatigue_impact_h = 8
            elif days < 5: fatigue_impact_h = 0
            elif days < 7: fatigue_impact_h = -5
            else: fatigue_impact_h = -10
        if history[a]:
            days = (fix['date'] - history[a]['date']).days
            if days < 2: fatigue_impact_a = 15
            elif days < 3: fatigue_impact_a = 8
            elif days < 5: fatigue_impact_a = 0
            elif days < 7: fatigue_impact_a = -5
            else: fatigue_impact_a = -10
        
        h_fdr = 0.7 * a_str + 0.1 * fatigue_impact_h
        a_fdr = 0.7 * h_str + 0.1 * fatigue_impact_a
        
        # Get CS FDR and probabilities
        cs_odds = fix.get('correct_score_odds')
        if cs_odds:
            valid_odds, total_inv = [], 0.0
            for score, odd_str in cs_odds.items():
                odd = convert_american_to_decimal(odd_str)
                if odd and odd > 1.0:
                    try:
                        h_g, a_g = map(int, score.split('-'))
                        inv = 1.0 / odd
                        valid_odds.append({'h_g': h_g, 'a_g': a_g, 'inv': inv})
                        total_inv += inv
                    except ValueError: 
                        continue
            
            if total_inv > 1e-6:
                P_h, P_d, P_a = 0.0, 0.0, 0.0
                for item in valid_odds:
                    prob = item['inv'] / total_inv
                    if item['h_g'] > item['a_g']: 
                        P_h += prob
                    elif item['h_g'] < item['a_g']: 
                        P_a += prob
                    else: 
                        P_d += prob
                
                h_fdr_cs = max(0, 100 * (1 - (3 * P_h + P_d) / 3))
                a_fdr_cs = max(0, 100 * (1 - (3 * P_a + P_d) / 3))
                
                h_fdr = 0.4 * h_fdr + 0.6 * h_fdr_cs
                a_fdr = 0.4 * a_fdr + 0.6 * a_fdr_cs
            else:
                P_h, P_d, P_a = 0.333, 0.334, 0.333
        else:
            P_h, P_d, P_a = 0.333, 0.334, 0.333
        
        # Get AFD/DFD
        if h_str + a_str == 0:
            h_afd, h_dfd, a_afd, a_dfd = 50.0, 50.0, 50.0, 50.0
        else:
            h_base = (h_str / (h_str + a_str)) * 2.7 * 1.15
            a_base = (a_str / (h_str + a_str)) * 2.7 * 0.85
            h_xg = max(0.2, h_base)
            a_xg = max(0.2, a_base)
            h_afd = round(max(0, min(100, 100*(1-h_xg/3.5))), 1)
            h_dfd = round(max(0, min(100, 100*(a_xg/3.5))), 1)
            a_afd = round(max(0, min(100, 100*(1-a_xg/3.5))), 1)
            a_dfd = round(max(0, min(100, 100*(h_xg/3.5))), 1)
        
        # Get competitiveness
        diff = abs(h_fdr - a_fdr)
        if diff <= 6: 
            label, low, high = "Minimal", "level4", "level4"
        elif diff <= 15: 
            label, low, high = "Medium", "level3", "level4"
        elif diff <= 30: 
            label, low, high = "Large", "level2", "level5"
        else: 
            label, low, high = "Massive", "level1", "level5"
        
        tier_map = {
            "level1": ("Very Easy", "#1A651A"),
            "level2": ("Easy", "#8FBC8F"),
            "level3": ("Average Difficulty", "#D3D3D3"),
            "level4": ("Difficult", "#F08080"),
            "level5": ("Very Difficult", "#A52A2A")
        }
        
        h_tier = tier_map[low]
        a_tier = tier_map[high]
        if h_fdr <= a_fdr:
            h_tier_d, a_tier_d = h_tier, a_tier
        else:
            h_tier_d, a_tier_d = a_tier, h_tier
        
        # Get top score predictions
        top_5_scores = []
        if cs_odds:
            scores, total_inv = [], 0
            for score, odd_str in cs_odds.items():
                odd = convert_american_to_decimal(odd_str)
                if odd and odd > 1.0:
                    inv = 1.0 / odd
                    scores.append({'score': score, 'inv': inv})
                    total_inv += inv
            
            if total_inv > 1e-6:
                for s in scores: 
                    s['probability'] = s['inv'] / total_inv
                top_5 = sorted(scores, key=lambda x: x['probability'], reverse=True)[:5]
                top_5_scores = [{'score': item['score'], 'probability': f"{item['probability']*100:.2f}%"} for item in top_5]
        
        # Add to results
        h_details = team_details.get(h, {"short_code": "N/A", "api_id": None, "image": "https://placeholder.com/logo.png"})
        a_details = team_details.get(a, {"short_code": "N/A", "api_id": None, "image": "https://placeholder.com/logo.png"})
        
        results.append({
            'fixture_id': fix['fixture_id'], 
            'gameweek': fix['gameweek'], 
            'starting_at': fix['starting_at'],
            'home_team_id': fix['home_team_id'], 
            'home_team_name': h, 
            'away_team_id': fix['away_team_id'], 
            'away_team_name': a,
            'home_team_short_code': h_details['short_code'], 
            'home_team_logo': h_details['image'],
            'away_team_short_code': a_details['short_code'], 
            'away_team_logo': a_details['image'],
            'final_home_fdr': round(h_fdr, 1), 
            'home_fdr_tier': h_tier_d[0], 
            'home_fdr_tier_color': h_tier_d[1],
            'final_away_fdr': round(a_fdr, 1), 
            'away_fdr_tier': a_tier_d[0], 
            'away_fdr_tier_color': a_tier_d[1],
            'fdr_diff': round(diff, 1), 
            'match_competitiveness_tier': label,
            'prob_home_win': round(P_h, 3), 
            'prob_draw': round(P_d, 3), 
            'prob_away_win': round(P_a, 3),
            'home_afd': h_afd, 
            'home_dfd': h_dfd, 
            'away_afd': a_afd, 
            'away_dfd': a_dfd,
            'top_5_score_predictions': top_5_scores
        })
    
    logger.info(f"API: --- FDR Processing Finished. Generated data for {len(results)} fixtures. ---")
    return results

# Player Predictions API Functions
def aga_process_epl_predictions():
    """Process EPL player predictions"""
    logger.info("API: --- Starting EPL Player Predictions Processing ---")
    
    try:
        epl_odds_data = load_data_from_mongo("epl", is_single_doc=True)
        sportsmonk_fixtures = load_data_from_mongo("sportsmonk_fixture")
        player_stats_raw = load_data_from_mongo("player_stat_24_25_mapped")
    except HTTPException:
        raise
    
    # Prepare player stats
    df = pd.DataFrame(player_stats_raw)
    df['team_canonical'] = df['team'].apply(lambda x: EPL_TEAM_NAME_MAPPING.get(x, x))
    for col in ['goals_scored', 'assists', 'clean_sheets']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    team_season_stats = {}
    for team_name, group in df.groupby('team_canonical'):
        def_players = group[group['position'].isin(["Goalkeeper", "Defender", "DEF"])]
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
    
    # Map fixtures and odds
    sportsmonk_lookup = {}
    all_fixture_teams = set()
    for fixture in sportsmonk_fixtures:
        all_fixture_teams.add(fixture['home_team_name'])
        all_fixture_teams.add(fixture['away_team_name'])
        start_time_val = fixture.get('starting_at')
        if not start_time_val: 
            continue
        
        start_time_obj = None
        try:
            if isinstance(start_time_val, datetime):
                start_time_obj = start_time_val.replace(tzinfo=timezone.utc)
            elif isinstance(start_time_val, str):
                start_time_obj = datetime.fromisoformat(start_time_val.replace('Z', '+00:00'))
            
            if start_time_obj:
                fixture['parsed_start_time'] = start_time_obj
                date_str = start_time_obj.strftime('%Y-%m-%d')
                sportsmonk_lookup[(fixture['home_team_name'], fixture['away_team_name'], date_str)] = fixture
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing fixture start time: {e}")
            continue
    
    mapped_fixtures = []
    for odds_details in epl_odds_data.get('matches', {}).values():
        h_odds, a_odds = parse_match_name(odds_details.get('match_name', ''), EPL_TEAM_NAME_MAPPING)
        time_str = odds_details.get('start_time')
        if h_odds == "Unknown" or not time_str: 
            continue
        
        try:
            date_obj = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            date_str = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            continue
        
        sm_fixture = sportsmonk_lookup.get((h_odds, a_odds, date_str))
        if sm_fixture:
            mapped_fixtures.append({
                'fixture_id': sm_fixture.get('fixture_id'), 
                'home_team_id': sm_fixture.get('home_team_id'),
                'away_team_id': sm_fixture.get('away_team_id'), 
                'home_team': sm_fixture['home_team_name'],
                'away_team': sm_fixture['away_team_name'], 
                'date': date_obj.date(), 
                'time': date_obj.time(),
                'starting_at': sm_fixture['parsed_start_time'].isoformat(), 
                'gameweek': sm_fixture.get('GW'),
                'odds_details': odds_details,
            })
    
    # Generate team details
    all_known_teams = list(all_fixture_teams | all_player_teams)
    team_details_lookup = {}
    for i, team_name in enumerate(all_known_teams):
        short_code_parts = [word[0] for word in team_name.split() if word[0].isupper()]
        short_code = "".join(short_code_parts) if short_code_parts else team_name[:3].upper()
        team_details_lookup[team_name] = {
            "team_id": f"TEAM_{1000 + i}",
            "short_code": short_code,
            "api_id": 2000 + i,
            "image": f"https://fantasyfootball.sgp1.cdn.digitaloceanspaces.com/epl-logos/{team_name.replace(' ', '-')}.png"
        }
    
    # Process each fixture
    all_match_predictions = []
    for fixture in mapped_fixtures:
        home_team, away_team = fixture['home_team'], fixture['away_team']
        logger.info(f"Processing: {home_team} vs {away_team}")
        
        home_details = team_details_lookup.get(home_team, {})
        away_details = team_details_lookup.get(away_team, {})
        
        # Estimate team xG and CS probabilities
        cs_odds = fixture['odds_details'].get('markets', {}).get('Correct Score')
        if cs_odds:
            cs_odds_decimal = {s: convert_american_to_decimal(o) for s, o in cs_odds.items() if convert_american_to_decimal(o)}
            h_xg, a_xg, h_cs_prob, a_cs_prob, total_inv = 0.0, 0.0, 0.0, 0.0, 0.0
            items = []
            
            for score, odd in cs_odds_decimal.items():
                try:
                    inv = 1.0 / odd
                    h, a = map(int, score.split('-'))
                    items.append({'h': h, 'a': a, 'inv': inv})
                    total_inv += inv
                except (ValueError, TypeError):
                    continue
            
            if total_inv > 0:
                for item in items:
                    norm_p = item['inv'] / total_inv
                    h_xg += item['h'] * norm_p
                    a_xg += item['a'] * norm_p
                    if item['a'] == 0: 
                        h_cs_prob += norm_p
                    if item['h'] == 0: 
                        a_cs_prob += norm_p
        else:
            h_xg, a_xg, h_cs_prob, a_cs_prob = 2.7/2, 2.7/2, 30.0, 30.0
        
        # Create match output
        match_output = {
            "fixture_id": fixture.get('fixture_id'), 
            "gameweek": fixture.get('gameweek'),
            "starting_at": fixture.get('starting_at'), 
            "home_team_id": fixture.get('home_team_id'),
            "home_team_name": home_team, 
            "away_team_id": fixture.get('away_team_id'),
            "away_team_name": away_team, 
            "home_team_short_code": home_details.get('short_code'),
            "home_team_logo": home_details.get('image'), 
            "away_team_short_code": away_details.get('short_code'),
            "away_team_logo": away_details.get('image'), 
            "home_team_xg": round(h_xg, 2),
            "away_team_xg": round(a_xg, 2), 
            "xg_source": "cs_odds" if cs_odds else "default_average",
            "players": []
        }
        
        # Process players for each team
        for team_name, team_xg, opp_xg, team_cs_prob in [
            (home_team, h_xg, a_xg, h_cs_prob), 
            (away_team, a_xg, h_xg, a_cs_prob)
        ]:
            if team_name in players_by_team:
                team_players_df = players_by_team[team_name]
                team_stats = team_season_stats.get(team_name, {
                    "total_goals": 0, 
                    "total_assists": 0, 
                    "total_clean_sheets": 0, 
                    "avg_cs_per_def_player": 0
                })
                
                for _, player_row in team_players_df.iterrows():
                    player_name = player_row.get('name')
                    position = player_row.get('position')
                    
                    # Get direct odds if available
                    match_odds = fixture['odds_details'].get('markets', {})
                    direct_ags_odds = match_odds.get('Anytime Goalscorer', {}).get(player_name)
                    direct_aas_odds = match_odds.get('Anytime Assist', {}).get(player_name)
                    
                    # Calculate probabilities
                    ags_prob, aas_prob, ags_prob_source, aas_prob_source = 0.0, 0.0, "no_data", "no_data"
                    
                    if direct_ags_odds:
                        ags_prob = (1 / convert_american_to_decimal(direct_ags_odds)) * 100
                        ags_prob_source = "direct_odds"
                    elif team_stats.get('total_goals', 0) > 0:
                        goal_share = player_row['goals_scored'] / team_stats['total_goals']
                        pos_modifier = {
                            'Midfielder': 0.8, 
                            'Forward': 1.2, 
                            'Defender': 0.35, 
                            'Goalkeeper': 0.02
                        }.get(position, 1.0)
                        individual_xg = goal_share * team_xg * pos_modifier
                        if individual_xg > 0:
                            ags_prob = (1 - poisson.pmf(0, individual_xg)) * 100
                            ags_prob_source = "poisson_model"
                    
                    if direct_aas_odds:
                        aas_prob = (1 / convert_american_to_decimal(direct_aas_odds)) * 100
                        aas_prob_source = "direct_odds"
                    elif team_stats.get('total_assists', 0) > 0:
                        assist_share = player_row['assists'] / team_stats['total_assists']
                        pos_modifier = {
                            'Midfielder': 1.4, 
                            'Forward': 0.9, 
                            'Defender': 0.7, 
                            'Goalkeeper': 0.05
                        }.get(position, 1.0)
                        individual_xa = assist_share * team_xg * 0.8 * pos_modifier
                        if individual_xa > 0:
                            aas_prob = (1 - poisson.pmf(0, individual_xa)) * 100
                            aas_prob_source = "poisson_model"
                    
                    # Calculate clean sheet probability
                    cs_prob, cs_prob_source = 0.0, "not_applicable"
                    if position in ["Goalkeeper", "Defender", "DEF"]:
                        cs_prob_source = "team_model"
                        base_cs_prob = team_cs_prob
                        avg_cs = team_stats.get('avg_cs_per_def_player', 0)
                        if avg_cs > 0:
                            player_cs = player_row['clean_sheets']
                            ratio_to_avg = player_cs / avg_cs
                            individual_modifier = 1 + ((ratio_to_avg - 1) * 0.1)
                            individual_modifier = np.clip(individual_modifier, 0.9, 1.15)
                            base_cs_prob *= individual_modifier
                            cs_prob_source = "hybrid_model"
                        
                        xg_adjustment_factor = max(0.5, 1 - (opp_xg / (2.7 * 1.5)))
                        calculated_cs_prob = base_cs_prob * xg_adjustment_factor
                        cs_prob = np.clip(calculated_cs_prob, 5.0, 85.0)
                    
                    # Add player data
                    player_data = player_row.to_dict()
                    player_data.update({
                        'anytime_goalscorer_probability': round(np.clip(ags_prob, 0, 75.0), 2),
                        'ags_prob_source': ags_prob_source,
                        'anytime_assist_probability': round(np.clip(aas_prob, 0, 60.0), 2),
                        'aas_prob_source': aas_prob_source,
                        'clean_sheet_probability': round(cs_prob, 2),
                        'cs_prob_source': cs_prob_source
                    })
                    
                    # Remove unnecessary fields
                    for field in ['_id', 'csv__id', '_match', 'team_canonical']:
                        player_data.pop(field, None)
                    
                    match_output["players"].append(player_data)
        
        all_match_predictions.append(match_output)
    
    logger.info(f"--- SUCCESS: Processed {len(all_match_predictions)} matches. ---")
    return all_match_predictions

# Player Points API Functions
def pp_process_epl_player_points():
    """Process EPL player points with enhanced model"""
    logger.info("API: --- Starting Enhanced EPL Player Points Processing ---")
    
    try:
        epl_odds_data = load_data_from_mongo("epl", is_single_doc=True)
        sportsmonk_fixtures = load_data_from_mongo("sportsmonk_fixture")
        player_stats_raw = load_data_from_mongo("player_stat_24_25_mapped")
    except HTTPException:
        raise
    
    # Prepare player stats with enhanced metrics
    df = pd.DataFrame(player_stats_raw)
    df['team_canonical'] = df['team'].apply(lambda x: EPL_TEAM_NAME_MAPPING.get(x, x))
    
    # Position categorization
    def get_position_category(position_str: str) -> str:
        if not isinstance(position_str, str):
            return 'Forward'
        pos_l = position_str.lower()
        if 'goalkeeper' in pos_l:
            return 'Goalkeeper'
        if 'defender' in pos_l or 'back' in pos_l:
            return 'Defender'
        if 'midfield' in pos_l:
            return 'Midfielder'
        return 'Forward'
    
    df['PositionCategory'] = df['position'].apply(get_position_category)
    
    for col in ['goals_scored', 'assists', 'clean_sheets']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculate player quality metrics
    def calculate_player_quality_metrics(player_stats: Dict, games_played: int = 25) -> Dict:
        goals = float(player_stats.get('goals_scored', 0))
        assists = float(player_stats.get('assists', 0))
        clean_sheets = float(player_stats.get('clean_sheets', 0))
        
        estimated_games = max(games_played, 1)
        goals_per_game = goals / estimated_games
        assists_per_game = assists / estimated_games
        combined_per_game = goals_per_game + assists_per_game
        cs_per_game = clean_sheets / estimated_games
        
        # Determine player tier
        if combined_per_game >= 0.8:
            tier = "premium"
            quality_multiplier = 1.4
        elif combined_per_game >= 0.4:
            tier = "good"
            quality_multiplier = 1.2
        else:
            tier = "regular"
            quality_multiplier = 1.0
        
        return {
            'goals_per_game': goals_per_game,
            'assists_per_game': assists_per_game,
            'combined_per_game': combined_per_game,
            'cs_per_game': cs_per_game,
            'tier': tier,
            'quality_multiplier': quality_multiplier,
            'estimated_games': estimated_games
        }
    
    df_with_quality = []
    for _, row in df.iterrows():
        player_data = row.to_dict()
        quality_metrics = calculate_player_quality_metrics(player_data)
        player_data.update(quality_metrics)
        df_with_quality.append(player_data)
    
    df = pd.DataFrame(df_with_quality)
    
    # Team season stats
    team_season_stats = {}
    for team_name, group in df.groupby('team_canonical'):
        def_players = group[group['position'].isin(["Goalkeeper", "Defender", "DEF"])]
        num_def_players = len(def_players)
        total_cs = def_players['clean_sheets'].sum()
        
        team_season_stats[team_name] = {
            "total_goals": group['goals_scored'].sum(),
            "total_assists": group['assists'].sum(),
            "total_clean_sheets": total_cs,
            "avg_cs_per_def_player": (total_cs / num_def_players) if num_def_players > 0 else 0,
            "avg_goals_per_game": group['goals_scored'].sum() / 25,
            "avg_assists_per_game": group['assists'].sum() / 25,
            "premium_players": len(group[group['tier'] == 'premium']),
            "team_strength": group['quality_multiplier'].mean()
        }
    
    players_by_team = {team: group for team, group in df.groupby('team_canonical')}
    all_player_teams = set(df['team_canonical'].unique())
    
    # Map fixtures and odds
    sportsmonk_lookup = {}
    all_fixture_teams = set()
    for fixture in sportsmonk_fixtures:
        all_fixture_teams.add(fixture['home_team_name'])
        all_fixture_teams.add(fixture['away_team_name'])
        start_time_val = fixture.get('starting_at')
        if not start_time_val: 
            continue
        
        start_time_obj = None
        try:
            if isinstance(start_time_val, datetime):
                start_time_obj = start_time_val.replace(tzinfo=timezone.utc)
            elif isinstance(start_time_val, str):
                start_time_obj = datetime.fromisoformat(start_time_val.replace('Z', '+00:00'))
            
            if start_time_obj:
                fixture['parsed_start_time'] = start_time_obj
                date_str = start_time_obj.strftime('%Y-%m-%d')
                sportsmonk_lookup[(fixture['home_team_name'], fixture['away_team_name'], date_str)] = fixture
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing fixture start time: {e}")
            continue
    
    mapped_fixtures = []
    for odds_details in epl_odds_data.get('matches', {}).values():
        h_odds, a_odds = parse_match_name(odds_details.get('match_name', ''), EPL_TEAM_NAME_MAPPING)
        time_str = odds_details.get('start_time')
        if h_odds == "Unknown" or not time_str: 
            continue
        
        try:
            date_obj = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            date_str = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            continue
        
        sm_fixture = sportsmonk_lookup.get((h_odds, a_odds, date_str))
        if sm_fixture:
            mapped_fixtures.append({
                'fixture_id': sm_fixture.get('fixture_id'), 
                'home_team_id': sm_fixture.get('home_team_id'),
                'away_team_id': sm_fixture.get('away_team_id'), 
                'home_team_name': sm_fixture['home_team_name'],
                'away_team_name': sm_fixture['away_team_name'], 
                'date': date_obj.date(),
                'starting_at': sm_fixture['parsed_start_time'].isoformat(), 
                'gameweek': sm_fixture.get('GW'),
                'odds_details': odds_details,
            })
    
    # Generate team details
    all_known_teams = list(all_fixture_teams | all_player_teams)
    team_details_lookup = {}
    for i, team_name in enumerate(all_known_teams):
        short_code_parts = [word[0] for word in team_name.split() if word[0].isupper()]
        short_code = "".join(short_code_parts) if short_code_parts else team_name[:3].upper()
        team_details_lookup[team_name] = {
            "team_id": f"TEAM_{1000 + i}",
            "short_code": short_code,
            "api_id": 2000 + i,
            "image": f"https://fantasyfootball.sgp1.cdn.digitaloceanspaces.com/epl-logos/{team_name.replace(' ', '-')}.png"
        }
    
    # Process each fixture
    all_match_predictions = []
    for fixture in mapped_fixtures:
        home_team, away_team = fixture['home_team_name'], fixture['away_team_name']
        logger.info(f"Processing: {home_team} vs {away_team}")
        
        home_details = team_details_lookup.get(home_team, {})
        away_details = team_details_lookup.get(away_team, {})
        
        # Estimate team xG and CS probabilities
        match_markets = fixture['odds_details'].get('markets', {})
        cs_odds = match_markets.get('Correct Score')
        
        if cs_odds:
            cs_odds_decimal = {s: convert_american_to_decimal(o) for s, o in cs_odds.items() if convert_american_to_decimal(o)}
            h_xg, a_xg, h_cs_prob, a_cs_prob, total_inv = 0.0, 0.0, 0.0, 0.0, 0.0
            items = []
            
            for score, odd in cs_odds_decimal.items():
                try:
                    inv = 1.0 / odd
                    h, a = map(int, score.split('-'))
                    items.append({'h': h, 'a': a, 'inv': inv})
                    total_inv += inv
                except (ValueError, TypeError):
                    continue
            
            if total_inv > 0:
                for item in items:
                    norm_p = item['inv'] / total_inv
                    h_xg += item['h'] * norm_p
                    a_xg += item['a'] * norm_p
                    if item['a'] == 0: 
                        h_cs_prob += norm_p
                    if item['h'] == 0: 
                        a_cs_prob += norm_p
        else:
            h_xg, a_xg, h_cs_prob, a_cs_prob = 2.7/2, 2.7/2, 25.0, 25.0
        
        # Create match output
        match_output = {
            "fixture_id": fixture.get('fixture_id'), 
            "gameweek": fixture.get('gameweek'),
            "starting_at": fixture.get('starting_at'), 
            "home_team_id": fixture.get('home_team_id'),
            "home_team_name": home_team, 
            "away_team_id": fixture.get('away_team_id'),
            "away_team_name": away_team, 
            "home_team_short_code": home_details.get('short_code'),
            "home_team_logo": home_details.get('image'), 
            "away_team_short_code": away_details.get('short_code'),
            "away_team_logo": away_details.get('image'),
            "home_expected_goals": round(h_xg, 2), 
            "away_expected_goals": round(a_xg, 2),
            "home_cs_probability": round(h_cs_prob, 1), 
            "away_cs_probability": round(a_cs_prob, 1),
            "players": []
        }
        
        # Process players for each team
        for team_name, is_home in [(home_team, True), (away_team, False)]:
            if team_name in players_by_team:
                team_players_df = players_by_team[team_name]
                team_stats = team_season_stats.get(team_name, {})
                team_xg = h_xg if is_home else a_xg
                opponent_xg = a_xg if is_home else h_xg
                cs_prob = (h_cs_prob if is_home else a_cs_prob) / 100
                
                for _, player_row in team_players_df.iterrows():
                    player_dict = player_row.to_dict()
                    player_name = player_row['name']
                    position = player_row['PositionCategory']
                    
                    # Try direct odds first
                    if "Anytime Goalscorer" in match_markets and player_name in match_markets["Anytime Goalscorer"]:
                        ags_odds = match_markets.get("Anytime Goalscorer", {}).get(player_name)
                        score_2_plus_odds = match_markets.get("To Score 2 or More Goals", {}).get(player_name)
                        hat_trick_odds = match_markets.get("To Score a Hat-Trick", {}).get(player_name)
                        assist_odds = match_markets.get("Anytime Assist", {}).get(player_name)
                        
                        prob_1 = 1/convert_american_to_decimal(ags_odds) if ags_odds else 0
                        prob_2 = min(prob_1, 1/convert_american_to_decimal(score_2_plus_odds)) if score_2_plus_odds else 0
                        prob_3 = min(prob_2, 1/convert_american_to_decimal(hat_trick_odds)) if hat_trick_odds else 0
                        prob_a = 1/convert_american_to_decimal(assist_odds) if assist_odds else 0
                        
                        prob_e1 = prob_1 - prob_2  # Exactly 1 goal
                        prob_e2 = prob_2 - prob_3  # Exactly 2 goals
                        
                        # Fantasy points config
                        goal_pts = {
                            'Goalkeeper': 10, 
                            'Defender': 6, 
                            'Midfielder': 5, 
                            'Forward': 4
                        }.get(position, 4)
                        
                        exp_pts = 2.0  # Minutes played
                        exp_pts += 1.0  # Enhanced baseline for players with odds
                        exp_pts += prob_e1 * goal_pts
                        exp_pts += prob_e2 * (goal_pts * 2)
                        exp_pts += prob_3 * (goal_pts * 3 + 2)  # Hat-trick bonus
                        exp_pts += prob_a * 3  # Assist points
                        
                        # Bonus for having goal odds
                        if prob_1 > 0.2:
                            exp_pts += 0.5
                        
                        points_data = {
                            "expected_points": round(exp_pts, 2),
                            "points_calc_method": "direct_odds_enhanced",
                            "goal_probability": round(prob_1, 3),
                            "assist_probability": round(prob_a, 3)
                        }
                    else:
                        # Use enhanced probability model
                        # Calculate baseline points
                        base_points = 2.0  # Minutes played
                        
                        # Add quality-based baseline
                        tier = player_dict.get('tier', 'regular')
                        if tier == 'premium':
                            base_points += 2.0
                        elif tier == 'good':
                            base_points += 1.0
                        
                        # Position-specific baseline
                        if position == 'Forward':
                            base_points += 0.5
                        elif position == 'Midfielder':
                            base_points += 0.3
                        
                        # Calculate individual goal probability
                        goals_per_game = player_dict.get('goals_per_game', 0)
                        quality_multiplier = player_dict.get('quality_multiplier', 1.0)
                        team_avg_goals = team_stats.get('avg_goals_per_game', 1.5)
                        
                        if team_avg_goals > 0:
                            match_multiplier = team_xg / team_avg_goals
                            goal_prob = goals_per_game * match_multiplier * quality_multiplier
                        else:
                            goal_prob = goals_per_game * quality_multiplier
                        
                        goal_prob = min(goal_prob, 0.85)
                        
                        # Calculate individual assist probability
                        assists_per_game = player_dict.get('assists_per_game', 0)
                        team_avg_assists = team_stats.get('avg_assists_per_game', 1.5)
                        
                        if team_avg_assists > 0:
                            match_multiplier = team_xg / team_avg_assists
                            assist_prob = assists_per_game * match_multiplier * quality_multiplier
                        else:
                            assist_prob = assists_per_game * quality_multiplier
                        
                        assist_prob = min(assist_prob, 0.75)
                        
                        # Calculate expected points
                        total_points = base_points
                        goal_pts = {
                            'Goalkeeper': 10, 
                            'Defender': 6, 
                            'Midfielder': 5, 
                            'Forward': 4
                        }.get(position, 4)
                        
                        # Expected goals
                        if player_dict.get('tier') == 'premium':
                            expected_goals = goal_prob + (goal_prob * 0.3)
                        else:
                            expected_goals = goal_prob
                        
                        total_points += expected_goals * goal_pts
                        
                        # Hat-trick bonus for premium forwards
                        if position == 'Forward' and goal_prob > 0.3:
                            hat_trick_prob = goal_prob * 0.1
                            total_points += hat_trick_prob * (goal_pts + 2)
                        
                        # Assist points
                        total_points += assist_prob * 3
                        
                        # Clean sheet points for defensive players
                        if position in ['Goalkeeper', 'Defender']:
                            cs_points = {
                                'Goalkeeper': 4, 
                                'Defender': 4, 
                                'Midfielder': 1, 
                                'Forward': 0
                            }.get(position, 0)
                            total_points += cs_prob * cs_points
                            
                            # Penalty for goals conceded
                            total_points -= opponent_xg * 0.5
                        
                        # Bonus points for quality players
                        if player_dict.get('tier') in ['premium', 'good']:
                            bonus_multiplier = 1.5 if player_dict.get('tier') == 'premium' else 1.2
                            total_points += 0.3 * bonus_multiplier
                        
                        # Form bonus
                        combined_per_game = player_dict.get('combined_per_game', 0)
                        if combined_per_game > 0.6:
                            form_bonus = min(combined_per_game * 0.5, 1.0)
                            total_points += form_bonus
                        
                        points_data = {
                            'expected_points': round(total_points, 2),
                            'goal_probability': round(goal_prob, 3),
                            'assist_probability': round(assist_prob, 3),
                            'expected_goals': round(expected_goals, 2),
                            'points_calc_method': 'enhanced_probability_model'
                        }
                    
                    # Combine player data with calculated points
                    final_player_data = player_dict.copy()
                    final_player_data.update(points_data)
                    
                    # Clean up unnecessary fields
                    for field in ['_id', 'csv__id', '_match', 'team_canonical', 'estimated_games']:
                        final_player_data.pop(field, None)
                    
                    match_output["players"].append(final_player_data)
        
        # Sort players by expected points
        match_output["players"].sort(key=lambda x: x.get('expected_points', 0), reverse=True)
        all_match_predictions.append(match_output)
    
    logger.info(f"--- SUCCESS: Processed {len(all_match_predictions)} matches with enhanced model. ---")
    return all_match_predictions

# ============================================================================
# PART 3: CREATE UNIFIED FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Unified EPL Analysis API",
    description="Combined API for FDR, Player Predictions, and Player Points",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler for consistent error responses
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

# FDR Endpoints
@app.get("/epl-fdr-results", summary="Get detailed FDR and team strength metrics for all fixtures", response_model=List[Dict[str, Any]])
async def get_epl_fdr_data():
    """Get detailed FDR and team strength metrics for all fixtures"""
    try:
        all_data = fdr_process_all_epl_data()
        return all_data
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred in FDR endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during FDR calculation.")

@app.get("/epl-top-score-predictions", summary="Get top 5 score predictions for all fixtures", response_model=List[Dict[str, Any]])
async def get_epl_match_predictions():
    """Get top 5 score predictions for all fixtures"""
    try:
        all_data = fdr_process_all_epl_data()
        prediction_results = []
        for match in all_data:
            prediction_results.append({
                'fixture_id': match['fixture_id'], 
                'gameweek': match['gameweek'], 
                'starting_at': match['starting_at'],
                'home_team_id': match['home_team_id'], 
                'home_team_name': match['home_team_name'],
                'away_team_id': match['away_team_id'], 
                'away_team_name': match['away_team_name'],
                'home_team_short_code': match['home_team_short_code'], 
                'home_team_logo': match['home_team_logo'],
                'away_team_short_code': match['away_team_short_code'], 
                'away_team_logo': match['away_team_logo'],
                'top_5_score_predictions': match['top_5_score_predictions']
            })
        return prediction_results
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred in predictions endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction calculation.")

# Player Predictions Endpoints
@app.get("/epl-player-predictions", summary="Get EPL Player Predictions (AGS, AAS, CS)", response_model=List[Dict[str, Any]])
async def get_epl_player_predictions():
    """Get EPL Player Predictions (Anytime Goalscorer, Anytime Assist, Clean Sheet)"""
    try:
        results = aga_process_epl_predictions()
        return results
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred in player predictions endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# Player Points Endpoints
@app.get("/epl-player-points", summary="Get Enhanced EPL Player Expected Points", response_model=List[Dict[str, Any]])
async def get_epl_player_points():
    """Get Enhanced EPL Player Expected Points"""
    try:
        results = pp_process_epl_player_points()
        return results
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred in player points endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Health check endpoint to verify the API is running"""
    return {
        "status": "healthy", 
        "model_version": "1.0.0", 
        "features": [
            "fdr_analysis", 
            "player_predictions", 
            "enhanced_player_points"
        ],
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# PART 4: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("=== Starting Unified EPL Analysis System ===")
    
    # Run the first two scripts sequentially
    run_process_all_league_data()
    run_from_cleandata_epl()
    
    logger.info("=== Starting Unified API Server ===")
    logger.info(f"API will be available at: http://{API_HOST}:{API_PORT}")
    logger.info(f"API Documentation at: http://{API_HOST}:{API_PORT}/docs")
    
    # Start the FastAPI application
    uvicorn.run(
        app, 
        host=API_HOST, 
        port=API_PORT, 
        reload=False,
        log_level="info",
        access_log=True
    )