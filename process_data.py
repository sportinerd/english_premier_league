import json

def parse_market_runners(market):
    """
    Parses the runners (betting options) from a market object and extracts
    their names and odds into a simple dictionary, ensuring correct American
    odds formatting (e.g., +110, -150).

    Args:
        market (dict): The market dictionary containing runner information.

    Returns:
        dict: A dictionary of runner names and their formatted American odds strings.
    """
    runner_data = {}
    for runner in market.get('runners', []):
        runner_name = runner.get('runnerName')
        odds_info = runner.get('winRunnerOdds', {}).get('americanDisplayOdds', {})
        odds = odds_info.get('americanOdds')
        if runner_name and odds is not None:
            # --- MODIFICATION START ---
            # Format the odds to include the '+' sign for positive values
            if odds > 0:
                runner_data[runner_name] = f"+{odds}"
            else:
                runner_data[runner_name] = str(odds)
            # --- MODIFICATION END ---
    return runner_data

def clean_and_represent_data(data):
    """
    Cleans and restructures the raw sports betting data into a more
    understandable format, organized by league, outright winners, and matches.

    Args:
        data (list): The raw data loaded from the JSON file.

    Returns:
        dict: A cleaned and structured dictionary of the sports data.
    """
    structured_data = {}

    # A comprehensive list of all match-specific market names to look for
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
        if not league_name:
            continue

        structured_data[league_name] = {
            "competitionId": competition_id,
            "outright_winner": {},
            "matches": {}
        }

        all_markets = []
        if 'futures' in league and league['futures']:
            all_markets.extend(league['futures'])
        for event in league.get('events', []):
            if 'futures' in event and event['futures']:
                all_markets.extend(event['futures'])

        events_by_id = {event['eventId']: event for event in league.get('events', [])}

        for market in all_markets:
            market_type = market.get('marketType')
            market_name = market.get('marketName')
            event_id = market.get('eventId')

            # --- Handle Outright Winner (League Winner) Markets ---
            if market_type == "OUTRIGHT_BETTING" and "Winner" in market_name:
                outright_odds = parse_market_runners(market)
                if outright_odds:
                    structured_data[league_name]["outright_winner"][market_name] = outright_odds
                continue

            # --- Handle All Other Match-Specific Markets ---
            if event_id and market_name in desired_match_markets:
                if event_id not in structured_data[league_name]["matches"]:
                    match_details = events_by_id.get(event_id, {})
                    structured_data[league_name]["matches"][event_id] = {
                        "match_name": match_details.get('name', "Unknown Event Name"),
                        "start_time": match_details.get('openDate', market.get('marketTime')),
                        "markets": {}
                    }
                
                market_runners = parse_market_runners(market)
                if market_runners:
                    structured_data[league_name]["matches"][event_id]["markets"][market_name] = market_runners

    return structured_data

# --- Main execution block ---
if __name__ == "__main__":
    input_filename = 'all_league_data.json'  # Make sure this matches your input file name
    output_filename = 'cleaned_data.json'
    
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        cleaned_data = clean_and_represent_data(raw_data)

        with open(output_filename, 'w', encoding='utf-8') as outfile:
            json.dump(cleaned_data, outfile, indent=2, ensure_ascii=False)
        
        print(f"Successfully cleaned the data and saved it to '{output_filename}'")
        
    except FileNotFoundError:
        print(f"Error: '{input_filename}' not found. Please ensure the file is in the same directory.")
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{input_filename}'. Details: {e}")