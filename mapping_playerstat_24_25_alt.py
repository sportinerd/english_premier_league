import pymongo
import unicodedata
import re
from pymongo import UpdateOne
from thefuzz import fuzz  # Import the fuzzy matching library

# --- MongoDB Connection Details ---
DB_URI = "mongodb://localhost:27017/"
DB_NAME = "English_premier_league"

# --- Source Collections ---
MASTER_PLAYER_COLLECTION = "mapped_fantasy_player"
PLAYER_STATS_COLLECTION = "player_stat_24_25"

# --- Output Collections ---
MAPPED_OUTPUT_COLLECTION = "mapped_player_stats_24_25"
UNMAPPED_OUTPUT_COLLECTION = "unmapped_player_stats_24_25"

# --- MAPPING & NORMALIZATION DICTIONARIES ---
POSITION_MAP = {
    "Defender": "DEF", "Midfielder": "MID", "Attacker": "FWD", "Goalkeeper": "GK",
    # Add any other long-form positions if they exist
}

TEAM_NAME_MAP = {
    # Key: Name from CSV file (lowercase) | Value: Name from JSON file (lowercase)
    "west ham": "west ham united",
    "spurs": "tottenham hotspur",
    "man utd": "manchester united",
    "nott'm forest": "nottingham forest",
    "man city": "manchester city",
    "brighton": "brighton & hove albion",
    "wolves": "wolverhampton wanderers",
    # This map can be expanded by analyzing the unmapped results
}

def normalize_text(text):
    """
    Normalizes text by making it lowercase, removing accents, handling special characters,
    and stripping extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Handle special characters like Ø -> o
    text = text.replace('ø', 'o').replace('Ø', 'o')
    nfkd_form = unicodedata.normalize('NFKD', text.lower())
    text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return ' '.join(text.split()) # Consolidate whitespace

def create_advanced_mapped_collections():
    """
    Intelligently merges player stats with master data using a multi-pass, fuzzy
    matching strategy and provides detailed logging for any failures.
    """
    try:
        client = pymongo.MongoClient(DB_URI)
        db = client[DB_NAME]

        master_players = db[MASTER_PLAYER_COLLECTION]
        player_stats = db[PLAYER_STATS_COLLECTION]
        mapped_output = db[MAPPED_OUTPUT_COLLECTION]
        unmapped_output = db[UNMAPPED_OUTPUT_COLLECTION]
        
        print("Successfully connected to MongoDB.")

        print(f"Clearing old output collections...")
        mapped_output.delete_many({})
        unmapped_output.delete_many({})
        print("Collections cleared.")

        # --- Build Multiple Lookup Dictionaries for Advanced Matching ---
        player_lookup = {}
        team_to_players_lookup = {}
        name_only_lookup = {}

        print("Building advanced lookup maps from the master player collection...")
        for player in master_players.find():
            team_name_raw = player.get("team_name", "").strip().lower()
            normalized_team = TEAM_NAME_MAP.get(team_name_raw, team_name_raw)
            short_pos = POSITION_MAP.get(player.get("position"), player.get("position"))

            merge_data = {
                "_id": player["_id"], "api_player_id": player.get("api_player_id"),
                "common_name": player.get("common_name"), "image": player.get("image"),
                "team_name": player.get("team_name"), "team_id": player.get("team_id"),
                "team_api_id": player.get("team_api_id"), "team_short_code": player.get("team_short_code"),
            }

            names_to_try = set(filter(None, [player.get("display_name"), player.get("common_name"), player.get("name")]))
            
            for name in names_to_try:
                normalized_name = normalize_text(name)
                if not all([normalized_name, normalized_team]): continue

                # 1. For exact name + team + position match
                if short_pos:
                    player_lookup[(normalized_name, normalized_team, short_pos)] = merge_data
                
                # 2. For exact name + team match (position relaxed)
                player_lookup[(normalized_name, normalized_team)] = merge_data

                # 3. For fuzzy name matching within a team
                if normalized_team not in team_to_players_lookup:
                    team_to_players_lookup[normalized_team] = []
                team_to_players_lookup[normalized_team].append((normalized_name, merge_data))

                # 4. For potential transfer matching (name only)
                if normalized_name not in name_only_lookup:
                    name_only_lookup[normalized_name] = []
                name_only_lookup[normalized_name].append(merge_data)

        print(f"Lookup maps created. Unique keys: {len(player_lookup)}")

        # --- Iterate and perform multi-pass matching ---
        unmapped_docs = []
        bulk_upsert_ops = []

        print("Processing and mapping player stats...")
        for stat_record in player_stats.find():
            stat_name_norm = normalize_text(stat_record.get("name", ""))
            stat_team_raw = stat_record.get("team", "").strip().lower()
            stat_team_norm = TEAM_NAME_MAP.get(stat_team_raw, stat_team_raw)
            stat_pos = stat_record.get("position", "").strip()

            master_data = None
            
            # PASS 1 & 2: Exact name + team (+/- position)
            key_with_pos = (stat_name_norm, stat_team_norm, stat_pos)
            key_without_pos = (stat_name_norm, stat_team_norm)
            master_data = player_lookup.get(key_with_pos) or player_lookup.get(key_without_pos)
            
            # PASS 3: Fuzzy name match within the same team
            if not master_data and stat_team_norm in team_to_players_lookup:
                best_match = None
                highest_score = 88  # Set a high threshold to avoid bad matches
                for master_name, player_data in team_to_players_lookup[stat_team_norm]:
                    score = fuzz.ratio(stat_name_norm, master_name)
                    if score > highest_score:
                        highest_score = score
                        best_match = player_data
                if best_match:
                    master_data = best_match

            # DECISION and LOGGING
            if master_data:
                merged_doc = stat_record.copy()
                merged_doc.update(master_data)
                operation = UpdateOne({"_id": master_data["_id"]}, {"$set": merged_doc}, upsert=True)
                bulk_upsert_ops.append(operation)
            else:
                # If no match, determine the reason for the log
                log_reason = "Player name not found in master data."
                if stat_team_norm not in team_to_players_lookup:
                    log_reason = f"Team name '{stat_record.get('team')}' could not be normalized or found."
                else:
                    # PASS 4: Check for potential transfers
                    potential_matches = name_only_lookup.get(stat_name_norm)
                    if potential_matches and len(potential_matches) == 1:
                        original_team = potential_matches[0]['team_name']
                        log_reason = f"Potential Transfer: Player found on team '{original_team}' in master data, but on '{stat_record.get('team')}' in stats."
                    else:
                        log_reason = f"Player '{stat_record.get('name')}' not found at team '{stat_record.get('team')}'."

                stat_record['mapping_failure_reason'] = log_reason
                unmapped_docs.append(stat_record)

        print("Mapping complete.")

        # --- Execute bulk operations ---
        if bulk_upsert_ops:
            print(f"\nPerforming bulk upsert for {len(bulk_upsert_ops)} documents into '{MAPPED_OUTPUT_COLLECTION}'...")
            result = mapped_output.bulk_write(bulk_upsert_ops)
            print("Bulk upsert complete.")
            print(f"  - New players inserted: {result.upserted_count}")
            print(f"  - Existing players updated: {result.modified_count}")
        
        if unmapped_docs:
            print(f"\nInserting {len(unmapped_docs)} unmapped documents into '{UNMAPPED_OUTPUT_COLLECTION}'...")
            unmapped_output.insert_many(unmapped_docs)
            print("Insertion complete.")

        print("\n--- Final Summary ---")
        print(f"Successfully mapped players: {len(bulk_upsert_ops)}")
        print(f"Unmapped players (check '{UNMAPPED_OUTPUT_COLLECTION}' for reasons): {len(unmapped_docs)}")
        print("---------------------")

    except pymongo.errors.ConnectionFailure as e:
        print(f"Error: Could not connect to MongoDB. Please ensure it is running.\nDetails: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'client' in locals() and client:
            client.close()
            print("MongoDB connection closed.")

if __name__ == "__main__":
    create_advanced_mapped_collections()