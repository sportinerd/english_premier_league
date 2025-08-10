import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
import uvicorn
from scipy.stats import poisson
from typing import List, Dict, Any, Optional, Tuple
from bson import ObjectId
import math

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# --- MongoDB Connection Settings ---
INPUT_DB_URI = "mongodb://localhost:27017/"
INPUT_DB_NAME = "English_premier_league"
EPL_ODDS_COLLECTION = "epl"
SPORTSMONK_FIXTURES_COLLECTION = "sportsmonk_fixture"
PLAYER_STAT_COLLECTION = "player_stat_24_25_mapped"

# --- Enhanced Fantasy Points System ---
FANTASY_POINTS_CONFIG = {
    'goal': {'Goalkeeper': 10, 'Defender': 6, 'Midfielder': 5, 'Forward': 4},
    'assist': 3,
    'clean_sheet': {'Goalkeeper': 4, 'Defender': 4, 'Midfielder': 1, 'Forward': 0},
    'hat_trick_bonus': 2,
    'minutes_played': 2,
    'conceded_penalty': {'Goalkeeper': -2, 'Defender': -1},
    'bonus_points_avg': 0.3,  # Average bonus points per game for good performers
    'captain_multiplier': 2.0,  # For premium players
}

# --- Enhanced Model Constants ---
AVERAGE_TOTAL_GOALS_IN_MATCH = 2.7
DEFENSIVE_POSITIONS = ["Goalkeeper", "Defender", "DEF"]
INDIVIDUAL_CS_INFLUENCE_FACTOR = 0.1

# Player quality thresholds (goals + assists per game)
PREMIUM_PLAYER_THRESHOLD = 0.8  # 0.8+ goals+assists per game
GOOD_PLAYER_THRESHOLD = 0.4     # 0.4+ goals+assists per game

# Form weighting (more recent performance weighted higher)
FORM_WEIGHT_RECENT = 0.6
FORM_WEIGHT_OVERALL = 0.4

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
            "team_id": f"TEAM_{1000 + i}", "short_code": short_code, "api_id": 2000 + i,
            "image": f"https://fantasyfootball.sgp1.cdn.digitaloceanspaces.com/epl-logos/{team_name.replace(' ', '-')}.png"
        }
    return team_details

# ==============================================================================
# 3. DATA LOADING & PREPARATION
# ==============================================================================

def convert_mongo_ids(data):
    if isinstance(data, list):
        return [convert_mongo_ids(item) for item in data]
    if isinstance(data, dict):
        return {key: convert_mongo_ids(value) for key, value in data.items()}
    if isinstance(data, ObjectId):
        return str(data)
    return data

def convert_american_to_decimal(american_odds_str: str) -> Optional[float]:
    if not isinstance(american_odds_str, str) or not american_odds_str:
        return None
    try:
        val = float(american_odds_str.replace('+', ''))
        return (val / 100) + 1 if val > 0 else (100 / abs(val)) + 1
    except (ValueError, TypeError):
        return None

def load_data_from_mongo(collection_name: str, is_single_doc: bool = False) -> Any:
    try:
        client = MongoClient(INPUT_DB_URI)
        collection = client[INPUT_DB_NAME][collection_name]
        data = collection.find_one() if is_single_doc else list(collection.find({}))
        client.close()
        if data:
            # The following line was causing the SyntaxError
            print(f"API: ✅ Successfully loaded data from '{INPUT_DB_NAME}.{collection_name}'.")
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

def calculate_player_quality_metrics(player_stats: Dict, games_played: int = 25) -> Dict:
    """Calculate enhanced player quality metrics"""
    goals = float(player_stats.get('goals_scored', 0))
    assists = float(player_stats.get('assists', 0))
    clean_sheets = float(player_stats.get('clean_sheets', 0))

    # Estimate games played if not provided (assuming mid-season)
    estimated_games = max(games_played, 1)

    goals_per_game = goals / estimated_games
    assists_per_game = assists / estimated_games
    combined_per_game = goals_per_game + assists_per_game
    cs_per_game = clean_sheets / estimated_games

    # Determine player tier
    if combined_per_game >= PREMIUM_PLAYER_THRESHOLD:
        tier = "premium"
        quality_multiplier = 1.4
    elif combined_per_game >= GOOD_PLAYER_THRESHOLD:
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

def load_and_prepare_player_stats() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]], set]:
    player_stats_raw = load_data_from_mongo(PLAYER_STAT_COLLECTION)
    df = pd.DataFrame(player_stats_raw)
    df['team_canonical'] = df['team'].apply(lambda x: EPL_TEAM_NAME_MAPPING.get(x, x))
    df['PositionCategory'] = df['position'].apply(get_position_category)

    for col in ['goals_scored', 'assists', 'clean_sheets']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Add player quality metrics
    df_with_quality = []
    for _, row in df.iterrows():
        player_data = row.to_dict()
        quality_metrics = calculate_player_quality_metrics(player_data)
        player_data.update(quality_metrics)
        df_with_quality.append(player_data)

    df = pd.DataFrame(df_with_quality)

    team_season_stats = {}
    for team_name, group in df.groupby('team_canonical'):
        def_players = group[group['position'].isin(DEFENSIVE_POSITIONS)]
        num_def_players = len(def_players)
        total_cs = def_players['clean_sheets'].sum()

        team_season_stats[team_name] = {
            "total_goals": group['goals_scored'].sum(),
            "total_assists": group['assists'].sum(),
            "total_clean_sheets": total_cs,
            "avg_cs_per_def_player": (total_cs / num_def_players) if num_def_players > 0 else 0,
            "avg_goals_per_game": group['goals_scored'].sum() / 25,  # Assuming 25 games played
            "avg_assists_per_game": group['assists'].sum() / 25,
            "premium_players": len(group[group['tier'] == 'premium']),
            "team_strength": group['quality_multiplier'].mean()
        }

    players_by_team = {team: group for team, group in df.groupby('team_canonical')}
    all_player_teams = set(df['team_canonical'].unique())
    print(f"API: ✅ Prepared enhanced player stats for {len(players_by_team)} teams.")
    return players_by_team, team_season_stats, all_player_teams

def map_fixtures_and_odds(epl_data, sportsmonk_fixtures):
    sportsmonk_lookup = {}
    all_fixture_teams = set()
    for fixture in sportsmonk_fixtures:
        all_fixture_teams.add(fixture['home_team_name'])
        all_fixture_teams.add(fixture['away_team_name'])
        start_time_val = fixture.get('starting_at')
        if not start_time_val:
            continue

        start_time_obj = None
        if isinstance(start_time_val, datetime):
            start_time_obj = start_time_val.replace(tzinfo=timezone.utc)
        elif isinstance(start_time_val, str):
            try:
                start_time_obj = datetime.fromisoformat(start_time_val.replace('Z', '+00:00'))
            except ValueError:
                print(f"WARNING: Could not parse date string from SportsMonk: {start_time_val}")
                continue

        if start_time_obj:
            fixture['parsed_start_time'] = start_time_obj
            date_str = start_time_obj.strftime('%Y-%m-%d')
            sportsmonk_lookup[(fixture['home_team_name'], fixture['away_team_name'], date_str)] = fixture

    mapped_fixtures = []
    for odds_details in epl_data.get('matches', {}).values():
        h_odds, a_odds = parse_match_name(odds_details.get('match_name', ''), EPL_TEAM_NAME_MAPPING)
        time_str = odds_details.get('start_time')
        if h_odds == "Unknown" or not time_str:
            continue
        try:
            date_obj = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            date_str = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            print(f"WARNING: Could not parse date string: {time_str}")
            continue

        sm_fixture = sportsmonk_lookup.get((h_odds, a_odds, date_str))
        if sm_fixture:
            mapped_fixtures.append({
                'fixture_id': sm_fixture.get('fixture_id'), 'home_team_id': sm_fixture.get('home_team_id'),
                'away_team_id': sm_fixture.get('away_team_id'), 'home_team_name': sm_fixture['home_team_name'],
                'away_team_name': sm_fixture['away_team_name'], 'date': date_obj.date(),
                'starting_at': sm_fixture['parsed_start_time'].isoformat(), 'gameweek': sm_fixture.get('GW'),
                'odds_details': odds_details,
            })
    print(f"API: ✅ Successfully mapped {len(mapped_fixtures)} fixtures with odds.")
    return mapped_fixtures, all_fixture_teams

# ==============================================================================
# 4. ENHANCED CORE CALCULATION LOGIC
# ==============================================================================

def get_score_probabilities_from_cs(cs_odds: Dict) -> Optional[Dict[str, float]]:
    if not cs_odds:
        return None
    raw_probs = {s: 1.0/convert_american_to_decimal(o) for s, o in cs_odds.items() if convert_american_to_decimal(o)}
    total_p = sum(raw_probs.values())
    return {s: p/total_p for s, p in raw_probs.items()} if total_p else None

def get_score_probabilities_poisson(xg_h, xg_a, max_g=6):
    probs = {f"{hg}-{ag}": poisson.pmf(hg, xg_h) * poisson.pmf(ag, xg_a) for hg in range(max_g+1) for ag in range(max_g+1)}
    total_p = sum(probs.values())
    return {s: p/total_p for s,p in probs.items()} if total_p else {"0-0":1.0}

# FIX: Corrected function definition
def calculate_enhanced_baseline_points(player_data: Dict) -> float:
    """Calculate enhanced baseline points based on player quality"""
    base_points = FANTASY_POINTS_CONFIG['minutes_played']  # 2 points for playing

    # Add quality-based baseline
    tier = player_data.get('tier', 'regular')
    if tier == 'premium':
        base_points += 2.0  # Premium players get additional baseline
    elif tier == 'good':
        base_points += 1.0  # Good players get moderate baseline

    # Add position-specific baseline
    position = player_data.get('PositionCategory', 'Forward')
    if position == 'Forward':
        base_points += 0.5  # Forwards slightly favored for attacking returns
    elif position == 'Midfielder':
        base_points += 0.3  # Midfielders get some bonus for versatility

    return base_points

# FIX: Corrected function definition
def calculate_individual_goal_probability(player_data: Dict, team_expected_goals: float, team_stats: Dict) -> float:
    """Calculate individual player's goal probability for the match"""
    goals_per_game = player_data.get('goals_per_game', 0)
    quality_multiplier = player_data.get('quality_multiplier', 1.0)

    # Base probability from season performance
    base_prob = goals_per_game

    # Adjust based on team's expected performance in this match
    team_avg_goals = team_stats.get('avg_goals_per_game', 1.5)
    if team_avg_goals > 0:
        match_multiplier = team_expected_goals / team_avg_goals
        base_prob *= match_multiplier

    # Apply quality multiplier for premium players
    final_prob = base_prob * quality_multiplier

    # Cap the probability at reasonable levels
    return min(final_prob, 0.85)  # Max 85% chance of scoring

# FIX: Corrected function definition
def calculate_individual_assist_probability(player_data: Dict, team_expected_goals: float, team_stats: Dict) -> float:
    """Calculate individual player's assist probability for the match"""
    assists_per_game = player_data.get('assists_per_game', 0)
    quality_multiplier = player_data.get('quality_multiplier', 1.0)

    # Base probability from season performance
    base_prob = assists_per_game

    # Adjust based on team's expected performance
    team_avg_assists = team_stats.get('avg_assists_per_game', 1.5)
    if team_avg_assists > 0:
        match_multiplier = team_expected_goals / team_avg_assists
        base_prob *= match_multiplier

    # Apply quality multiplier
    final_prob = base_prob * quality_multiplier

    # Cap the probability
    return min(final_prob, 0.75)  # Max 75% chance

# FIX: Corrected function definition
def calculate_enhanced_points_from_probabilities(player_data: Dict, team_expected_goals: float,
                                                 opponent_expected_goals: float, team_stats: Dict,
                                                 clean_sheet_prob: float) -> Dict:
    """Enhanced points calculation using individual probabilities"""
    position = player_data.get('PositionCategory', 'Forward')

    # Start with enhanced baseline
    total_points = calculate_enhanced_baseline_points(player_data)

    # Individual goal probability and points
    goal_prob = calculate_individual_goal_probability(player_data, team_expected_goals, team_stats)
    goal_points = FANTASY_POINTS_CONFIG['goal'].get(position, 4)

    # Expected goals (including multiple goals for premium players)
    expected_goals = 0.0
    if player_data.get('tier') == 'premium':
        # Premium players have higher chance of multiple goals
        expected_goals = goal_prob + (goal_prob * 0.3)  # 30% chance of second goal if they score
        total_points += expected_goals * goal_points

        # Hat-trick bonus for premium forwards
        if position == 'Forward' and goal_prob > 0.3:
            hat_trick_prob = goal_prob * 0.1  # 10% chance relative to scoring
            total_points += hat_trick_prob * (goal_points + FANTASY_POINTS_CONFIG['hat_trick_bonus'])
    else:
        expected_goals = goal_prob
        total_points += expected_goals * goal_points

    # Individual assist probability and points
    assist_prob = calculate_individual_assist_probability(player_data, team_expected_goals, team_stats)
    total_points += assist_prob * FANTASY_POINTS_CONFIG['assist']

    # Clean sheet points for defensive players
    if position in ['Goalkeeper', 'Defender']:
        cs_points = FANTASY_POINTS_CONFIG['clean_sheet'].get(position, 0)
        total_points += clean_sheet_prob * cs_points

        # Penalty for goals conceded
        expected_goals_conceded = opponent_expected_goals
        penalty_per_goal = 0.5 if position == 'Goalkeeper' else 0.5
        total_points -= expected_goals_conceded * penalty_per_goal

    # Bonus points for quality players
    if player_data.get('tier') in ['premium', 'good']:
        bonus_multiplier = 1.5 if player_data.get('tier') == 'premium' else 1.2
        total_points += FANTASY_POINTS_CONFIG['bonus_points_avg'] * bonus_multiplier

    # Form bonus (for players performing above average)
    combined_per_game = player_data.get('combined_per_game', 0)
    if combined_per_game > 0.6:  # High performers get form bonus
        form_bonus = min(combined_per_game * 0.5, 1.0)
        total_points += form_bonus

    return {
        'expected_points': round(total_points, 2),
        'goal_probability': round(goal_prob, 3),
        'assist_probability': round(assist_prob, 3),
        'expected_goals': round(expected_goals, 2)
    }

def calculate_points_from_direct_odds(player_name, position, match_markets):
    """Enhanced direct odds calculation with quality bonuses"""
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

    goal_pts = FANTASY_POINTS_CONFIG['goal'].get(position, 4)
    exp_pts = FANTASY_POINTS_CONFIG['minutes_played']

    # Enhanced baseline for players with odds (implies they're good enough to have odds)
    exp_pts += 1.0

    exp_pts += prob_e1 * goal_pts
    exp_pts += prob_e2 * (goal_pts * 2)
    exp_pts += prob_3 * (goal_pts * 3 + FANTASY_POINTS_CONFIG['hat_trick_bonus'])
    exp_pts += prob_a * FANTASY_POINTS_CONFIG['assist']

    # Bonus for having goal odds (indicates bookmaker confidence)
    if prob_1 > 0.2:  # If >20% chance to score
        exp_pts += 0.5  # Bonus for being a goal threat

    return {
        "expected_points": round(exp_pts, 2),
        "points_calc_method": "direct_odds",
        "goal_probability": round(prob_1, 3),
        "assist_probability": round(prob_a, 3)
    }

def estimate_team_xg_cs(cs_odds_american: Optional[Dict]) -> Tuple[float, float, float, float]:
    """Enhanced xG estimation with better defaults"""
    if not cs_odds_american:
        return AVERAGE_TOTAL_GOALS_IN_MATCH/2, AVERAGE_TOTAL_GOALS_IN_MATCH/2, 25.0, 25.0

    cs_odds = {s: convert_american_to_decimal(o) for s, o in cs_odds_american.items() if convert_american_to_decimal(o)}
    h_xg, a_xg, h_cs_prob, a_cs_prob, total_inv = 0.0, 0.0, 0.0, 0.0, 0.0

    items = []
    for score, odd in cs_odds.items():
        if odd and odd > 1.0:
            try:
                inv = 1.0 / odd
                h, a = map(int, score.split('-'))
                items.append({'h': h, 'a': a, 'inv': inv})
                total_inv += inv
            except (ValueError, TypeError):
                continue

    if total_inv == 0:
        return AVERAGE_TOTAL_GOALS_IN_MATCH/2, AVERAGE_TOTAL_GOALS_IN_MATCH/2, 25.0, 25.0

    for item in items:
        norm_p = item['inv'] / total_inv
        h_xg += item['h'] * norm_p
        a_xg += item['a'] * norm_p
        if item['a'] == 0:
            h_cs_prob += norm_p
        if item['h'] == 0:
            a_cs_prob += norm_p

    # Ensure minimum xG values for more realistic expectations
    h_xg = max(h_xg, 0.5)
    a_xg = max(a_xg, 0.5)

    return h_xg, a_xg, h_cs_prob * 100, a_cs_prob * 100

# ==============================================================================
# 5. MAIN ORCHESTRATION SCRIPT
# ==============================================================================

def process_epl_player_points():
    epl_odds_data = load_data_from_mongo(EPL_ODDS_COLLECTION, is_single_doc=True)
    sportsmonk_fixtures = load_data_from_mongo(SPORTSMONK_FIXTURES_COLLECTION)
    players_by_team, team_season_stats, all_player_teams = load_and_prepare_player_stats()

    mapped_fixtures, all_fixture_teams = map_fixtures_and_odds(epl_odds_data, sportsmonk_fixtures)
    all_known_teams = list(all_fixture_teams | all_player_teams)
    team_details_lookup = generate_team_details(all_known_teams)

    all_match_predictions = []
    print("\n--- Processing Each Mapped Fixture with Enhanced Model ---")

    for fixture in mapped_fixtures:
        home_team, away_team = fixture['home_team_name'], fixture['away_team_name']
        print(f"Processing: {home_team} vs {away_team}")

        home_details = team_details_lookup.get(home_team, {})
        away_details = team_details_lookup.get(away_team, {})
        match_markets = fixture['odds_details'].get('markets', {})

        cs_odds = match_markets.get('Correct Score')
        h_xg, a_xg, home_cs_prob, away_cs_prob = estimate_team_xg_cs(cs_odds)

        print(f"  Expected Goals - {home_team}: {h_xg:.2f}, {away_team}: {a_xg:.2f}")
        print(f"  Clean Sheet Prob - {home_team}: {home_cs_prob:.1f}%, {away_team}: {away_cs_prob:.1f}%")

        match_output = {
            "fixture_id": fixture.get('fixture_id'), "gameweek": fixture.get('gameweek'),
            "starting_at": fixture.get('starting_at'), "home_team_id": fixture.get('home_team_id'),
            "home_team_name": home_team, "away_team_id": fixture.get('away_team_id'),
            "away_team_name": away_team, "home_team_short_code": home_details.get('short_code'),
            "home_team_logo": home_details.get('image'), "away_team_short_code": away_details.get('short_code'),
            "away_team_logo": away_details.get('image'),
            "home_expected_goals": round(h_xg, 2), "away_expected_goals": round(a_xg, 2),
            "home_cs_probability": round(home_cs_prob, 1), "away_cs_probability": round(away_cs_prob, 1),
            "players": []
        }

        for team_name, is_home in [(home_team, True), (away_team, False)]:
            if team_name in players_by_team:
                team_players_df = players_by_team[team_name]
                team_stats = team_season_stats.get(team_name, {})
                team_xg = h_xg if is_home else a_xg
                opponent_xg = a_xg if is_home else h_xg
                cs_prob = (home_cs_prob if is_home else away_cs_prob) / 100

                for _, player_row in team_players_df.iterrows():
                    player_dict = player_row.to_dict()

                    # Try direct odds first (for players with betting odds)
                    if "Anytime Goalscorer" in match_markets and player_row['name'] in match_markets["Anytime Goalscorer"]:
                        points_data = calculate_points_from_direct_odds(
                            player_row['name'], player_row['PositionCategory'], match_markets
                        )
                        points_data['points_calc_method'] = 'direct_odds_enhanced'
                    else:
                        # Use enhanced probability model
                        points_data = calculate_enhanced_points_from_probabilities(
                            player_dict, team_xg, opponent_xg, team_stats, cs_prob
                        )
                        points_data['points_calc_method'] = 'enhanced_probability_model'

                    # Combine player data with calculated points
                    final_player_data = player_dict.copy()
                    final_player_data.update(points_data)

                    # Clean up unnecessary fields
                    for field in ['_id', 'csv__id', '_match', 'team_canonical', 'estimated_games']:
                        final_player_data.pop(field, None)

                    match_output["players"].append(final_player_data)

        # Sort players by expected points (highest first)
        match_output["players"].sort(key=lambda x: x.get('expected_points', 0), reverse=True)
        all_match_predictions.append(match_output)

    print(f"\n--- SUCCESS: Processed {len(all_match_predictions)} matches with enhanced model. ---")
    return all_match_predictions

# ==============================================================================
# 6. FASTAPI APPLICATION
# ==============================================================================
app = FastAPI(
    title="Enhanced EPL Player Expected Points API",
    description="Calculates expected fantasy points for EPL players using an enhanced hybrid model with quality-based adjustments.",
    version="2.0.0"
)

@app.get("/epl-player-points", summary="Get Enhanced EPL Player Expected Points")
async def get_epl_player_points():
    try:
        results = process_epl_player_points()
        return results
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": "2.0.0", "features": ["enhanced_baselines", "quality_tiers", "individual_probabilities", "form_weighting"]}

if __name__ == "__main__":
    uvicorn.run("player_points:app", host="0.0.0.0", port=8000, reload=True)