
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup, Comment


# --- 1. OBTENER ESTADÍSTICAS DE JUGADORES POR PARTIDO ---
stats_per_game_url = 'https://www.basketball-reference.com/leagues/NBA_2025_per_game.html'
stats_per_game_csv_filename = 'nba_stats_per_game_regular_season.csv'

print(f"Obteniendo datos de estadísticas desde: {stats_per_game_url}")
try:
    stats_tables_list = pd.read_html(stats_per_game_url)
    stats_df_raw = stats_tables_list[0]
    stats_df_raw.to_csv(stats_per_game_csv_filename, index=False)
    print(f"Éxito. Datos de estadísticas guardados en: {stats_per_game_csv_filename}")
except Exception as e:
    print(f"Error al obtener datos de Basketball-Reference: {e}")

print("\n" + "-" * 30 + "\n")
time.sleep(2)

#-- OBTENER ESTADÍSTICAS AVANZADAS --
# --- Configuration ---
url = "https://www.basketball-reference.com/leagues/NBA_2025_advanced.html"
csv_filename = "nba_advanced_stats_2025.csv"

print(f"Scraping Advanced Stats from: {url}")

try:
    # 1. Read the HTML table directly into a list of DataFrames
    # This function grabs all <table> elements on the page.
    tables = pd.read_html(url)

    # 2. Select the first table
    # The 'Advanced' stats table is the main table on this page.
    advanced_df = tables[0]

    print(f"Table found with {len(advanced_df)} rows.")

    # 3. Clean the Data (Remove Repeated Headers)
    # Basketball-Reference repeats the header row (Rk, Player, Pos...) every 20 rows.
    # We simply filter out rows where the 'Rk' column equals the string 'Rk'.


    # 4. Save to CSV
    advanced_df.to_csv(csv_filename, index=False)

    print("\n--- SUCCESS ---")
    print(f"Cleaned data saved to '{csv_filename}'")
    print(f"Final row count: {len(advanced_df)}")
    print(advanced_df.head().to_markdown(index=False))

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"Failed to scrape data: {e}")
    print("Make sure you have 'lxml' installed (pip install lxml).")

# -- OBTENER LOS ENFRENTAMIENTOS ENTRE EQUIPOS
url = "https://www.basketball-reference.com/leagues/NBA_2025_standings.html"
csv_filename = "expanded_standings.csv"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all comments in the HTML
comments = soup.find_all(string=lambda text: isinstance(text, Comment))

target_table = None

# Loop through comments to find the one hiding our table
for comment in comments:
    if 'id="expanded_standings"' in comment:
        # This comment contains our table!
        # Parse the HTML *inside* the comment
        comment_soup = BeautifulSoup(comment, 'html.parser')
        target_table = comment_soup.find('table', id='expanded_standings')
        break

if target_table:
    # Use pandas to read the table HTML directly
    df = pd.read_html(str(target_table))[0]

    df.to_csv(csv_filename, index=False)

    print(df.head())
else:
    print("Table not found in comments.")


# -- OBTENER LOS SALARIOS DE LOS JUGADORES

# 1. Define the URL
# This page contains the salaries for the 2024-2025 season
salary_url = "https://www.basketball-reference.com/contracts/players.html"
csv_filename = "nba_salaries_2024_2025_raw.csv"

print(f"Attempting to scrape: {salary_url}")

try:
    # 2. Use pandas to read the HTML table directly
    # We assume the header is on the second row (header=1) to handle the multi-level headers
    tables = pd.read_html(salary_url, header=1)

    # The first table is the one we want
    salary_df = tables[0]

    # 4. Save to CSV
    salary_df.to_csv(csv_filename, index=False)

    print("\n--- SUCCESS ---")
    print(f"Data saved to '{csv_filename}'")
    print(f"Rows scraped: {len(salary_df)}")
    print(salary_df[['Player', '2025-26']].head().to_markdown(index=False))

except Exception as e:
    print(f"Error: {e}")
    print("If this fails, try Solution 2 (Manual Download).")