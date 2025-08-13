
import pymongo
from pymongo import UpdateOne
from bson.objectid import ObjectId
import os

# --- MongoDB Connection Details ---
INPUT_DB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
INPUT_DB_NAME = "Fantasy_LiveScoring"
OUTPUT_COLLECTION_NAME = "mapped_fantasy_player"  # The name for the new collection

def create_or_update_mapped_collection():
    """
    Connects to MongoDB, merges player data with their corresponding team 
    information, and then "upserts" the data into the 'mapped_fantasy_player'
    collection. This avoids duplicates and only updates/adds records.
    """
    try:
        # 1. Establish connection to MongoDB
        client = pymongo.MongoClient(INPUT_DB_URI)
        db = client[INPUT_DB_NAME]

        # 2. Get collections
        # Note: Using the collection names from your files, not the filenames themselves
        players_collection = db["players"]
        teams_collection = db["teams"]
        teamofseasons_collection = db["teamofseasons"]

        output_collection = db[OUTPUT_COLLECTION_NAME]

        print("Successfully connected to MongoDB.")

        # 3. Create a mapping of team_id to team details for efficiency
        team_info_map = {}
        for team in teams_collection.find({}, {"_id": 1, "name": 1, "api_team_id": 1, "short_code": 1}):
            team_info_map[team['_id']] = {
                "team_name": team.get("name"),
                "team_api_id": team.get("api_team_id"),
                "team_short_code": team.get("short_code")
            }
        
        print(f"Created a map of details for {len(team_info_map)} teams.")

        # 4. Create a mapping from player_id to team_id
        player_to_team_map = {}
        for team_season in teamofseasons_collection.find({}, {"team_id": 1, "player_ids": 1}):
            team_id = team_season.get("team_id")
            if team_id:
                for player_id in team_season.get("player_ids", []):
                    player_to_team_map[player_id] = team_id

        print(f"Created a map for {len(player_to_team_map)} player-team relationships.")

        # 5. Prepare a list of bulk 'upsert' operations
        bulk_operations = []
        unmatched_player_count = 0
        
        all_players = list(players_collection.find())

        for player in all_players:
            player_id = player["_id"]
            team_id = player_to_team_map.get(player_id)
            
            # Start with the original player data
            enriched_doc = player.copy()

            if team_id and team_id in team_info_map:
                team_details = team_info_map[team_id]
                # Add the team fields to the document
                enriched_doc["team_id"] = team_id
                enriched_doc["team_name"] = team_details["team_name"]
                enriched_doc["team_api_id"] = team_details["team_api_id"]
                enriched_doc["team_short_code"] = team_details["team_short_code"]
            else:
                unmatched_player_count += 1
                print(f"Warning: Team not found for player {player.get('display_name', player_id)}.")

            # Create an UpdateOne operation for the bulk write
            # It will match on '_id' and either update it or insert it if it doesn't exist.
            operation = UpdateOne(
                {"_id": enriched_doc["_id"]},  # The filter to find the document
                {"$set": enriched_doc},         # The data to apply
                upsert=True                     # The magic flag: update or insert
            )
            bulk_operations.append(operation)

        # 6. Execute the bulk write operation if there are players to process
        if bulk_operations:
            print(f"\nPerforming bulk upsert for {len(bulk_operations)} documents into '{OUTPUT_COLLECTION_NAME}'...")
            result = output_collection.bulk_write(bulk_operations)
            print("Bulk operation complete.")
            
            print("\n--- Operation Summary ---")
            print(f"Collection '{OUTPUT_COLLECTION_NAME}' is now up-to-date.")
            print(f"New players inserted: {result.upserted_count}")
            print(f"Existing players updated: {result.modified_count}")
            print(f"Players that were already up-to-date: {result.matched_count - result.modified_count}")
            print(f"Players processed without a matching team: {unmatched_player_count}")
            print("--------------------------")
        else:
            print("No players found to process.")

    except pymongo.errors.ConnectionFailure as e:
        print(f"Error: Could not connect to MongoDB. Please ensure it is running.\nDetails: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # 7. Close the MongoDB connection
        if 'client' in locals() and client:
            client.close()
            print("MongoDB connection closed.")

if __name__ == "__main__":
    create_or_update_mapped_collection()