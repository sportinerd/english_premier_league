import subprocess
import multiprocessing
import uvicorn
import time
import sys
from dotenv import load_dotenv
load_dotenv()
# --- Configuration ---
# List of scripts to run sequentially before starting the APIs.
SEQUENTIAL_SCRIPTS = [
    "process_all_league_data_mongo.py",
    "from_cleandata_epl.py"
]

def run_script(script_name):
    """Executes a python script and waits for it to complete."""
    print(f"--- Running sequential script: {script_name} ---")
    try:
        # Using sys.executable ensures we use the same python interpreter.
        process = subprocess.run(
            [sys.executable, script_name], 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"--- Finished script: {script_name} ---")
        print("Output:\n" + process.stdout)
        if process.stderr:
            print("Errors:\n" + process.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR: Script {script_name} failed with exit code {e.returncode} ---")
        print("Output:\n" + e.stdout)
        print("Errors:\n" + e.stderr)
        return False
    except FileNotFoundError:
        print(f"--- ERROR: Script {script_name} not found. Make sure it's in the same directory. ---")
        return False


def start_api_server(app_module_name, app_object_name, port, host="0.0.0.0"):
    """
    A target function for multiprocessing to start a Uvicorn server.
    
    Args:
        app_module_name (str): The name of the python file (e.g., 'fdr').
        app_object_name (str): The name of the FastAPI app instance (usually 'app').
        port (int): The port number for the API.
        host (str): The host for the API.
    """
    print(f"--- Starting API '{app_module_name}' on http://{host}:{port} ---")
    try:
        uvicorn.run(
            f"{app_module_name}:{app_object_name}", 
            host=host, 
            port=port, 
            reload=False
        )
    except Exception as e:
        print(f"--- ERROR starting API '{app_module_name}': {e} ---")

if __name__ == "__main__":
    # =================================================================
    # 1. RUN SEQUENTIAL DATA PROCESSING SCRIPTS
    # =================================================================
    print(">>> Starting Step 1: Sequential data processing...")
    for script in SEQUENTIAL_SCRIPTS:
        success = run_script(script)
        if not success:
            print(">>> Halting execution due to an error in a data processing script.")
            sys.exit(1)
    
    print("\n>>> All data processing scripts completed successfully.")
    
    # =================================================================
    # 2. DEFINE AND START PARALLEL API SERVERS
    # =================================================================
    print("\n>>> Starting Step 2: Launching all APIs in parallel...")
    
    apis_to_run = [
        ('fdr', 'app', 8000),
        ('anytime_goal_assist_cleansheet', 'app', 8001),
        ('player_points', 'app', 8002)
    ]
    
    processes = []

    for api_module, api_app, api_port in apis_to_run:
        process = multiprocessing.Process(
            target=start_api_server, 
            args=(api_module, api_app, api_port)
        )
        processes.append(process)
        process.start()
        time.sleep(2)

    print("\n--- ✅ All APIs have been launched successfully! ---")
    print(f"  - FDR API running on: http://localhost:{apis_to_run[0][2]}")
    print(f"  - Anytime Predictions API running on: http://localhost:{apis_to_run[1][2]}")
    print(f"  - Player Points API running on: http://localhost:{apis_to_run[2][2]}")
    print("\nPress Ctrl+C to stop all servers.")

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print("\n--- Shutting down all API servers... ---")
        for process in processes:
            process.terminate()
            process.join()
        print("--- All servers have been shut down. ---")