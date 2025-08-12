import json
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# ==============================================================================
# 1. MONGODB CONFIGURATION
# ==============================================================================

# --- Source MongoDB Configuration (Where to READ raw data from) ---
SOURCE_DB_URI = "mongodb://localhost:27017/"
SOURCE_DB_NAME = "English_premier_league"
SOURCE_COLLECTION = "scrapped_all_league_data"

# --- Destination MongoDB Configuration (Where to WRITE cleaned data to) ---
DESTINATION_DB_NAME = "English_premier_league"
DESTINATION_COLLECTION = "cleaned_all_data"

# ==============================================================================
# 2. DATABASE LOADING FUNCTION
# ==============================================================================

def load_data_from_mongodb(uri, db_name, collection_name):
    """
    Loads the initial raw data from a specified MongoDB collection.
    """
    client = None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster')
        db = client[db_name]
        collection = db[collection_name]
        
        print(f"Successfully connected to MongoDB. Fetching data from '{db_name}.{collection_name}'...")
        data = list(collection.find({}))
        
        if not data:
            print(f"Warning: No documents found in the source collection '{collection_name}'.")
            return None
            
        print(f"✅ Successfully loaded {len(data)} documents from MongoDB.")
        return data
        
    except ConnectionFailure as e:
        print(f"Error: Could not connect to MongoDB at '{uri}'. Please ensure it is running.")
        print(f"Details: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading data from MongoDB: {e}")
        return None
    finally:
        if client:
            client.close()

# ==============================================================================
# 3. DATA PARSING AND CLEANING FUNCTIONS (UNCHANGED)
# ==============================================================================

def parse_market_runners(market):
    runner_data = {}
    for runner in market.get('runners', []):
        runner_name = runner.get('runnerName')
        odds_info = runner.get('winRunnerOdds', {}).get('americanDisplayOdds', {})
        odds = odds_info.get('americanOdds')
        if runner_name and odds is not None:
            runner_data[runner_name] = f"+{odds}" if odds > 0 else str(odds)
    return runner_data

def clean_and_represent_data(data):
    structured_data = {}
    desired_match_markets = [
        "Moneyline (3-way)", "Over/Under 1.5 Goals", "Over/Under 2.5 Goals",
        "Over/Under 3.5 Goals", "Over/Under 4.5 Goals", "Both Teams To Score",
        "Double Chance", "Half-Time Result", "Correct Score",
        "Correct Score Combinations", "Anytime Goalscorer", "First Goalscorer",
        "Anytime Assist", "To Score 2 or More Goals", "To Score a Hat-Trick",
        "To Score a Header", "Goalkeeper To Make 1 Or More Saves",
        "Goalkeeper To Make 2 Or More Saves", "Goalkeeper To Make 3 Or More Saves"
    ]
    
    for league in data:
        league_name = league.get('name')
        competition_id = league.get('competitionId')
        if not league_name or not competition_id: continue

        structured_data[league_name] = {
            "competitionId": competition_id,
            "outright_winner": {},
            "matches": {}
        }
        all_markets = league.get('futures', [])
        for event in league.get('events', []):
            all_markets.extend(event.get('futures', []))
        events_by_id = {event['eventId']: event for event in league.get('events', [])}

        for market in all_markets:
            market_type, market_name, event_id = market.get('marketType'), market.get('marketName'), market.get('eventId')
            if market_type == "OUTRIGHT_BETTING" and "Winner" in market_name:
                outright_odds = parse_market_runners(market)
                if outright_odds: structured_data[league_name]["outright_winner"][market_name] = outright_odds
            elif event_id and market_name in desired_match_markets:
                event_id_str = str(event_id)
                if event_id_str not in structured_data[league_name]["matches"]:
                    match_details = events_by_id.get(event_id, {})
                    structured_data[league_name]["matches"][event_id_str] = {
                        "eventId": event_id,
                        "match_name": match_details.get('name', "Unknown Event Name"),
                        "start_time": match_details.get('openDate', market.get('marketTime')),
                        "markets": {}
                    }
                market_runners = parse_market_runners(market)
                if market_runners:
                    structured_data[league_name]["matches"][event_id_str]["markets"][market_name] = market_runners
            
    return structured_data

# ==============================================================================
# 4. DATABASE STORAGE FUNCTION (CORRECTED)
# ==============================================================================

def save_to_mongodb(data, uri, db_name, collection_name):
    """
    Saves the cleaned data to a MongoDB collection using an upsert operation.
    """
    client = None
    try:
        client = MongoClient(uri)
        client.admin.command('ismaster')
        
        # --- FIX IS HERE ---
        # The single-line assignment has been split into two lines.
        db = client[db_name]
        collection = db[collection_name]
        # --- END FIX ---

        upserted_count, modified_count = 0, 0
        
        for league_name, league_data in data.items():
            query = {'competitionId': league_data['competitionId']}
            update_payload = {'$set': {'league_name': league_name, **league_data}}
            result = collection.update_one(query, update_payload, upsert=True)
            if result.upserted_id: upserted_count += 1
            elif result.modified_count > 0: modified_count += 1
        
        print("\n--- MongoDB Operation Summary ---")
        print(f"Successfully connected to MongoDB database: '{db_name}'.")
        print(f"Data processed for collection: '{collection_name}'.")
        print(f"✅ Documents Inserted: {upserted_count}")
        print(f"✅ Documents Updated: {modified_count}")
        
    except ConnectionFailure as e:
        print(f"Error: Could not connect to MongoDB at '{uri}'. Please ensure it is running.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the database operation: {e}")
    finally:
        if client: client.close()

# ==============================================================================
# 5. MAIN EXECUTION BLOCK (UNCHANGED)
# ==============================================================================
if __name__ == "__main__":
    try:
        print(f"--- Starting Data Processing ---")
        
        raw_data = load_data_from_mongodb(SOURCE_DB_URI, SOURCE_DB_NAME, SOURCE_COLLECTION)

        if raw_data:
            print("\nCleaning and structuring the raw data...")
            cleaned_data = clean_and_represent_data(raw_data)
            print("✅ Data cleaning complete.")

            print("\nSaving structured data to destination MongoDB collection...")
            save_to_mongodb(cleaned_data, SOURCE_DB_URI, DESTINATION_DB_NAME, DESTINATION_COLLECTION)
        else:
            print("\nProcessing stopped due to failure in loading initial data.")
            
    except Exception as e:
        print(f"An unexpected error occurred in the main process: {e}")