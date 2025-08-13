import pymongo
from pymongo import MongoClient
from difflib import SequenceMatcher
from bson import ObjectId
import logging
from datetime import datetime
import re
from collections import defaultdict
import unicodedata
import Levenshtein
from fuzzywuzzy import fuzz, process
import pandas as pd
import json
import os

# --- Configuration ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Input JSON file path
INPUT_JSON_FILE = "2024-25_processed.json"

# Input Player Database (Sportsmonk)
INPUT_DB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
INPUT_DB_NAME = "English_premier_league"
INPUT_PLAYER_COLLECTION = "Sportsmonk_player"

# Output Database
OUTPUT_DB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
OUTPUT_DB_NAME = "EPL_Player_Stats_Mapped"
MAPPED_COLLECTION_NAME = "mapped_players_2024_25"
UNMAPPED_COLLECTION_NAME = "unmapped_players_2024_25"


class RobustPlayerMatcher:
    """A class to handle robust player matching using various string comparison techniques."""
    def __init__(self):
        # Enhanced nickname mapping from soccer conventions
        self.nickname_map = {
            'alex': ['alexander', 'alejandro', 'alessandro', 'alexandre', 'alexis'],
            'bob': ['robert', 'roberto'],
            'bill': ['william', 'billy', 'willy'],
            'chris': ['christopher', 'christian', 'cristiano', 'cristian'],
            'dan': ['daniel', 'danilo', 'danny'],
            'dave': ['david', 'davide'],
            'ed': ['edward', 'eduardo', 'edouard', 'edison', 'edson'],
            'frank': ['francisco', 'francesco', 'franco', 'franck'],
            'fred': ['frederico', 'frederick', 'alfredo'],
            'gabi': ['gabriel'],
            'jim': ['james', 'jimmy', 'jaime'],
            'joe': ['joseph', 'jose', 'josep'],
            'john': ['juan', 'joao', 'giovanni', 'johan', 'johannes', 'jon'],
            'leo': ['leonardo', 'leon', 'lionel'],
            'mike': ['michael', 'miguel', 'michele', 'mikhail', 'mikael'],
            'nick': ['nicholas', 'nicolas', 'nicola', 'nico'],
            'pete': ['peter', 'pedro', 'pietro', 'petr'],
            'rick': ['richard', 'ricardo', 'riccardo'],
            'rob': ['robert', 'roberto', 'robin'],
            'sam': ['samuel', 'samu', 'samir'],
            'steve': ['stephen', 'steven', 'stefano', 'esteban'],
            'tom': ['thomas', 'tomas', 'tommy', 'thom'],
            'tony': ['antonio', 'antoine', 'anton'],
            'will': ['william', 'wilhelm', 'guillermo', 'guillaume'],
            'rafa': ['rafael'], 'xavi': ['xavier'], 'pepe': ['jose', 'pedro'],
            'kaka': ['ricardo'], 'deco': ['anderson'], 'hulk': ['givanildo'],
            'juninho': ['junior'], 'ronaldinho': ['ronaldo']
        }
        self.reverse_nickname_map = self._build_reverse_nickname_map()
        self.name_suffixes = ['jr', 'junior', 'sr', 'senior', 'ii', 'iii', 'iv', 'filho', 'neto']
        self.all_players_df = None

    def _build_reverse_nickname_map(self):
        """Builds a reverse mapping from full names to nicknames."""
        reverse_map = defaultdict(list)
        for short, fulls in self.nickname_map.items():
            for full in fulls:
                reverse_map[full].append(short)
        return reverse_map

    def preload_players_optimized(self, player_info_coll):
        """Loads and preprocesses player data into a pandas DataFrame for efficient searching."""
        logger.info("Preloading player data for fast matching...")
        projection = {"_id": 1, "name": 1, "display_name": 1, "first_name": 1, "last_name": 1}
        
        try:
            players = list(player_info_coll.find({}, projection))
            if not players:
                logger.warning("No players found in the Sportsmonk_player collection!")
                self.all_players_df = pd.DataFrame()
                return

            self.all_players_df = pd.DataFrame(players)
            # Create a normalized full name column for matching
            self.all_players_df['normalized_name'] = self.all_players_df['name'].apply(
                lambda x: self.normalize_name_ultra(x) if isinstance(x, str) else ""
            )
            logger.info(f"Loaded and preprocessed {len(players)} players into memory.")
        except Exception as e:
            logger.error(f"Failed to preload players from MongoDB: {e}")
            self.all_players_df = pd.DataFrame()

    def normalize_name_ultra(self, name):
        """Performs comprehensive normalization of a player's name."""
        if not name:
            return ""
        
        name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
        name = name.lower().strip()
        
        for suffix in self.name_suffixes:
            name = re.sub(rf'\s+{suffix}\.?\s*$', '', name, flags=re.IGNORECASE)
            
        name = re.sub(r"[^\w\s\-']", '', name)
        name = ' '.join(name.split())
        return name

    def match_player(self, player_name, team_name):
        """
        Finds the best match for a player from the preloaded DataFrame.
        This simplified version matches on full name.
        """
        if self.all_players_df is None or self.all_players_df.empty:
            return None, 0, []

        normalized_search_name = self.normalize_name_ultra(player_name)
        
        # Use fuzzywuzzy's process module to find the best matches from the preloaded data
        # It returns a list of tuples: (matched_name, score, index)
        best_matches = process.extractBests(
            normalized_search_name, 
            self.all_players_df['normalized_name'],
            scorer=fuzz.token_sort_ratio,
            limit=5,
            score_cutoff=85  # A reasonably high threshold to ensure quality matches
        )
        
        if not best_matches:
            return None, 0, []

        # Get the details of the best match
        best_match_name, best_score, best_match_index = best_matches[0]
        best_match_player_doc = self.all_players_df.iloc[best_match_index].to_dict()
        
        # Clean up the document before returning
        best_match_player_doc.pop('normalized_name', None)

        # Prepare details for logging/debugging
        match_details = []
        for match_name, score, index in best_matches:
            player_doc = self.all_players_df.iloc[index]
            match_details.append({
                "matched_name": player_doc["name"],
                "score": score
            })

        return best_match_player_doc, best_score, match_details


def preprocess_json_data(filepath):
    """
    Loads, preprocesses, and structures the player statistics from the input JSON file.
    """
    logger.info(f"Preprocessing data from {filepath}...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found at path: {filepath}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Could not decode JSON from file: {filepath}")
        return []

    preprocessed_players = []
    for player_key, stats in data.items():
        # Extract player name and team name from the key (e.g., "Alex Scott (Bournemouth)")
        match = re.match(r'^(.*?) \((.*?)\)$', player_key)
        if not match:
            logger.warning(f"Could not parse player key: {player_key}. Skipping.")
            continue
        
        player_name = match.group(1).strip()
        team_name = match.group(2).strip()

        # Calculate total matches and points
        total_matches = sum(stats.get('games_played_by_opponent', {}).values())
        total_points = sum(stats.get('total_points_by_opponent', {}).values())
        
        player_record = {
            "source_player_name": player_name,
            "source_team_name": team_name,
            "total_matches": total_matches,
            "total_points": total_points,  # Note: This is a combined metric from the source file
            "stats_breakdown": {
                "total_points_by_opponent": stats.get("total_points_by_opponent", {}),
                "games_played_by_opponent": stats.get("games_played_by_opponent", {}),
                "was_home_counts_by_opponent": stats.get("was_home_counts_by_opponent", {}),
                "total_home_points_by_opponent": stats.get("total_home_points_by_opponent", {})
            }
        }
        preprocessed_players.append(player_record)
        
    logger.info(f"Successfully preprocessed {len(preprocessed_players)} players from the JSON file.")
    return preprocessed_players

def main():
    """
    Main function to run the player mapping process.
    """
    try:
        # --- Database Setup ---
        input_client = MongoClient(INPUT_DB_URI)
        input_db = input_client[INPUT_DB_NAME]
        player_info_coll = input_db[INPUT_PLAYER_COLLECTION]

        output_client = MongoClient(OUTPUT_DB_URI)
        output_db = output_client[OUTPUT_DB_NAME]
        mapped_coll = output_db[MAPPED_COLLECTION_NAME]
        unmapped_coll = output_db[UNMAPPED_COLLECTION_NAME]

        # Create unique index to prevent duplicate mappings
        mapped_coll.create_index([("sportsmonk_player_info._id", 1)], unique=True)
        unmapped_coll.create_index([("source_player_name", 1), ("source_team_name", 1)], unique=True)
        logger.info("Database indexes checked/created.")

        # --- Preprocessing and Matching ---
        
        # 1. Preload players from Sportsmonk DB
        matcher = RobustPlayerMatcher()
        matcher.preload_players_optimized(player_info_coll)
        
        # 2. Preprocess players from JSON file
        players_to_map = preprocess_json_data(INPUT_JSON_FILE)
        
        if not players_to_map:
            logger.warning("No players to map. Exiting.")
            return

        # 3. Iterate and map each player
        logger.info("Starting player mapping process...")
        stats = {
            'successful': 0,
            'failed': 0,
            'already_exists': 0,
            'total': len(players_to_map)
        }
        
        for player_data in players_to_map:
            player_name = player_data["source_player_name"]
            team_name = player_data["source_team_name"]
            
            best_match, best_score, match_details = matcher.match_player(player_name, team_name)
            
            if best_match:
                # Successful match
                mapped_document = {
                    "source_player_info": player_data,
                    "sportsmonk_player_info": best_match,
                    "mapping_details": {
                        "match_score": best_score,
                        "top_candidates": match_details,
                        "mapped_at": datetime.utcnow()
                    }
                }
                
                try:
                    mapped_coll.insert_one(mapped_document)
                    stats['successful'] += 1
                    logger.info(f"SUCCESS: Mapped '{player_name}' to '{best_match['name']}' with score {best_score}")
                except pymongo.errors.DuplicateKeyError:
                    stats['already_exists'] += 1
                    logger.info(f"SKIP: Mapping for '{player_name}' -> '{best_match['name']}' already exists.")
            
            else:
                # Failed match
                unmapped_document = {
                    "source_player_name": player_data["source_player_name"],
                    "source_team_name": player_data["source_team_name"],
                    "stats": player_data,
                    "mapping_failure_details": {
                        "reason": "No high-confidence match found.",
                        "top_candidates": match_details,
                        "failed_at": datetime.utcnow()
                    }
                }
                
                try:
                    unmapped_coll.insert_one(unmapped_document)
                    stats['failed'] += 1
                    logger.warning(f"FAILURE: Could not map '{player_name}'. Saved to unmapped collection.")
                except pymongo.errors.DuplicateKeyError:
                    stats['already_exists'] += 1
                    logger.info(f"SKIP: Unmapped record for '{player_name}' already exists.")

        # --- Final Report ---
        logger.info("\n" + "="*50)
        logger.info("          MAPPING PROCESS COMPLETE")
        logger.info("="*50)
        logger.info(f"Total players processed: {stats['total']}")
        logger.info(f"Successfully mapped: {stats['successful']}")
        logger.info(f"Failed to map: {stats['failed']}")
        logger.info(f"Already in database (skipped): {stats['already_exists']}")
        logger.info("="*50)

    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
    finally:
        if 'input_client' in locals():
            input_client.close()
        if 'output_client' in locals():
            output_client.close()
        logger.info("Database connections closed.")

if __name__ == "__main__":
    main()