import os
import time
import zipfile
import requests
from tqdm import tqdm

API_URL = "https://ps.waltheri.net/api/games/"
SGF_BASE_URL = "https://ps.waltheri.net"

SAVE_DIR = "sgf_files"
ZIP_NAME = "waltheri_sgf_collection.zip"
os.makedirs(SAVE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

def fetch_all_games():
    all_games = []
    limit = 100
    offset = 0

    print("Fetching game metadata...")
    while True:
        params = {"limit": limit, "offset": offset}
        response = requests.get(API_URL, params=params, headers=HEADERS)
        if response.status_code != 200:
            print(f"Error fetching games at offset {offset}")
            break

        data = response.json()
        games = data.get("results", [])
        if not games:
            break

        all_games.extend(games)
        offset += limit
        time.sleep(0.3)  # be polite to server

    print(f"Total games found: {len(all_games)}")
    return all_games

def download_sgf(game, idx):
    game_id = game.get("id")
    if not game_id:
        return False
    sgf_url = f"{SGF_BASE_URL}/game/{game_id}/sgf/"
    response = requests.get(sgf_url, headers=HEADERS)
    if response.status_code == 200:
        filename = os.path.join(SAVE_DIR, f"game_{idx:05d}.sgf")
        with open(filename, "wb") as f:
            f.write(response.content)
        return True
    return False

def create_zip(output_path, source_dir):
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, source_dir)
                zipf.write(filepath, arcname)

def main():
    games = fetch_all_games()

    for idx, game in enumerate(tqdm(games, desc="Downloading SGFs")):
        try:
            if not download_sgf(game, idx):
                print(f"Failed to download SGF for game ID {game.get('id')}")
        except Exception as e:
            print(f"Error on game {game.get('id')}: {e}")
        time.sleep(0.2)

    print("Creating ZIP archive...")
    create_zip(ZIP_NAME, SAVE_DIR)
    print(f"Done! Saved to: {ZIP_NAME}")

if __name__ == "__main__":
    main()
