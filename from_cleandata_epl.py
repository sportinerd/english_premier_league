import pymongo
from pymongo.errors import ConnectionFailure

# --- MongoDB Connection Settings ---
# The source database and collection
INPUT_DB_URI = "mongodb://localhost:27017/"
INPUT_DB_NAME = "all_league"
INPUT_COLLECTION_NAME = "cleaned_all_data"  # The collection where all league data is stored.

# The new database and collection for EPL data
OUTPUT_DB_NAME = "English_premier_league"
OUTPUT_COLLECTION_NAME = "epl"

# The unique competitionId for the English Premier League
EPL_COMPETITION_ID = 10932509

def extract_and_store_epl():
    """
    Connects to MongoDB, finds the English Premier League document by its unique
    competitionId, and upserts it into a new, dedicated database and collection.
    This process is non-destructive and avoids creating duplicates.
    """
    print("--- Starting EPL Data Extraction ---")
    
    client = None

    try:
        # 1. Connect to MongoDB
        print(f"Attempting to connect to MongoDB at {INPUT_DB_URI}...")
        client = pymongo.MongoClient(INPUT_DB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster')
        print("✅ MongoDB connection successful.")

        # 2. Access the source database and collection
        input_db = client[INPUT_DB_NAME]
        input_collection = input_db[INPUT_COLLECTION_NAME]
        print(f"Accessed source: '{INPUT_DB_NAME}.{INPUT_COLLECTION_NAME}'")

        # 3. Find the EPL document using its unique competitionId (CORRECTED LOGIC)
        print(f"Searching for the EPL document with competitionId: {EPL_COMPETITION_ID}...")
        epl_document = input_collection.find_one({"competitionId": EPL_COMPETITION_ID})

        # 4. Check if the document was found
        if not epl_document:
            print(f"ERROR: No document found with competitionId {EPL_COMPETITION_ID} in '{INPUT_COLLECTION_NAME}'.")
            print("Please ensure the data has been processed and inserted correctly by the previous script.")
            return
            
        print("✅ Successfully found the English Premier League source document.")

        # 5. Access the destination database and collection
        output_db = client[OUTPUT_DB_NAME]
        output_collection = output_db[OUTPUT_COLLECTION_NAME]
        print(f"Accessed destination: '{OUTPUT_DB_NAME}.{OUTPUT_COLLECTION_NAME}'")

        # 6. Prepare data for upsert
        # The entire found document is the data we want to save.
        # We remove the original '_id' field because it cannot be updated.
        # The destination document will get its own unique _id upon insertion.
        epl_document.pop('_id', None)

        # 7. Upsert the data into the new collection
        # This is the core logic that meets your requirements:
        # - It finds a document with the matching competitionId.
        # - If found, it updates it with the new data ($set).
        # - If not found, it inserts a new document (upsert=True).
        # This prevents duplicates and does not remove existing data.
        print(f"Upserting data with competitionId: {EPL_COMPETITION_ID} into the destination collection...")
        
        result = output_collection.update_one(
            {"competitionId": EPL_COMPETITION_ID},
            {"$set": epl_document},
            upsert=True
        )

        print("\n--- Migration Complete! ---")
        if result.upserted_id:
            print("✅ A new document was CREATED in the destination collection.")
        elif result.modified_count > 0:
            print("✅ An existing document was UPDATED in the destination collection.")
        else:
            print("ℹ️ The data in the destination collection was already up-to-date. No changes were made.")
            
        print(f"The 'English Premier League' data is now stored in '{OUTPUT_DB_NAME}.{OUTPUT_COLLECTION_NAME}'.")

    except ConnectionFailure as e:
        print(f"\nERROR: Could not connect to MongoDB.")
        print(f"Please ensure your MongoDB server is running at '{INPUT_DB_URI}'.")
        print(f"Error details: {e}")
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    finally:
        # 8. Always close the connection
        if client:
            client.close()
            print("\nMongoDB connection closed.")

# This makes the script executable
if __name__ == "__main__":
    extract_and_store_epl()