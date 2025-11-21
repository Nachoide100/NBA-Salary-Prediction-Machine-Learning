import pandas as pd
import matplotlib.pyplot as plt
#-- PHASE 1: PREPARING AND CLEANING THE DATA --
#Convert the .csv to dataframe
df_stats_per_game = pd.read_csv("data/nba_stats_per_game_regular_season.csv")
df_stats_per_game.info()
df_advanced_stats = pd.read_csv("data/nba_advanced_stats_2025.csv")
df_advanced_stats.info()
df_expanded_standings = pd.read_csv("data/expanded_standings.csv", header=1) #Header in row 2
df_expanded_standings.info()
df_salaries = pd.read_csv("data/nba_salaries_2024_2025_raw.csv")
df_salaries.info()

#Cleaning the data
#SALARIES
#Select the columns
df_salaries = df_salaries[["Player","2025-26"]]
#Change the name
df_salaries = df_salaries.rename(columns={"2025-26":"Salary"})
#Cleaning the column "Salary": String to int
df_salaries["Salary"] = df_salaries["Salary"].astype(str).str.replace(r'[$,]', '', regex=True)
df_salaries["Salary"] = pd.to_numeric(df_salaries["Salary"], errors='coerce')
#Drop the nan rows
df_salaries = df_salaries.dropna(subset=["Player"])
df_salaries =df_salaries.dropna(subset=["Salary"])
# --- NEW FIX: Remove Duplicate Players in Salary Data ---
# Sort by salary descending so we keep the highest value if there are duplicates
df_salaries = df_salaries.sort_values('Salary', ascending=False)

# Drop duplicates, keeping the first (highest salary) one
df_salaries = df_salaries.drop_duplicates(subset=['Player'], keep='first')

#Verfication
df_salaries.info()
print(df_salaries.head(10))

#EXPANDED STANDINGS
#Select the columns
df_expanded_standings = df_expanded_standings[["Team", "Overall"]]
#Map the team column
# Define the mapping dictionary
team_map = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BRK',
    'Charlotte Hornets': 'CHO', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHO',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

# Apply the mapping to create a new 'Team_Code' column
# This looks at the 'Team' column, finds the name in the dictionary, and returns the code.
df_expanded_standings['Team_Code'] = df_expanded_standings['Team'].map(team_map)

# Verification: Check for any spelling errors (NaN values)
print("Unmapped teams:", df_expanded_standings[df_expanded_standings['Team_Code'].isna()]['Team'].unique())

# 1. Split the "Overall" string into two new columns: 'Wins' and 'Losses'
df_expanded_standings[['Wins', 'Losses']] = df_expanded_standings['Overall'].str.split('-', expand=True)

# 2. Convert these new columns to Integers (they are currently strings)
df_expanded_standings['Wins'] = pd.to_numeric(df_expanded_standings['Wins'])
df_expanded_standings['Losses'] = pd.to_numeric(df_expanded_standings['Losses'])

# 3. Calculate the Win Percentage
df_expanded_standings['Win_Pct'] = (df_expanded_standings['Wins'] /
                                    (df_expanded_standings['Wins'] + df_expanded_standings['Losses']))

# Optional: Round to 3 decimal places (e.g., 0.744)
df_expanded_standings['Win_Pct'] = df_expanded_standings['Win_Pct'].round(3)

# View the result
df_expanded_standings.info()
print(df_expanded_standings[['Team', 'Team_Code', 'Overall', 'Win_Pct']].head())

#STATS PER GAME AND ADVANCED STATS

#Select the most important columns of stats per game
df_stats_per_game = df_stats_per_game[["Player", "Age", "Team",
                                       "G", "MP", "PTS",
                                       "FG%",
                                       "3P%", "FT%"]]
#Rename certain columns for clarity
df_stats_per_game = df_stats_per_game.rename(columns={"G": "Games Played",
                                                      "MP": "Minutes Played",
                                                      })
print("--- INFO STATS PER GAME ---")
print(df_stats_per_game.head())

#Remove duplicates

# --- Step 1: Define what "Total" looks like ---
# Basketball-Reference uses '2TM', '3TM', '4TM', or sometimes 'TOT'
total_tags = ['2TM', '3TM', '4TM', 'TOT']

# --- Step 2: Create a "Mapping Dictionary" ---
# We want to find the team where each player played the MOST games.

# Filter to get only the individual team rows (Exclude '2TM', etc.)
individual_teams_df = df_stats_per_game[~df_stats_per_game['Team'].isin(total_tags)].copy()

# Sort by Games ('G') in descending order
# So for a player, the team with the most games appears first
individual_teams_df = individual_teams_df.sort_values(by=['Player', 'Games Played'], ascending=[True, False])

# Drop duplicates to keep ONLY the team with the max games for each player
# This creates a list of unique players and their "Main Team"
main_team_map_df = individual_teams_df.drop_duplicates(subset=['Player'], keep='first')

# Convert this to a dictionary: {'Davion Mitchell': 'TOR', 'Caleb Martin': 'MIA', ...}
player_to_team_dict = dict(zip(main_team_map_df['Player'], main_team_map_df['Team']))

# --- Step 3: Update the "Total" Rows ---
# Now we go back to our main dataframe.

# Identify the rows that need fixing (the ones with '2TM', etc.)
mask_totals = df_stats_per_game['Team'].isin(total_tags)

# Apply the dictionary map ONLY to those rows
# This says: "Look at the Player name, find their Main Team in the dictionary, and paste it here."
df_stats_per_game.loc[mask_totals, 'Team'] = df_stats_per_game.loc[mask_totals, 'Player'].map(player_to_team_dict)

# If a player somehow wasn't in the map, fill NaNs back with original value
df_stats_per_game['Team'] = df_stats_per_game['Team'].fillna('Unknown')

# --- Step 4: Now Drop Duplicates (Keep 'First') ---
# Since the "Total" row usually comes first in Basketball-Reference,
# we can now safely keep it. It has the total stats AND the correct team.
df_stats_per_game_clean = df_stats_per_game.drop_duplicates(subset=['Player'], keep='first')

# --- Verify ---
print("--- STATS PER GAME DUPLICATES REMOVED ---")
print(df_stats_per_game_clean.head())

#ADVANCED STATS
#Select the most important columns of advanced stats
df_advanced_stats = df_advanced_stats[["Player", "Team","G",  "PER", "TS%",
                                       "3PAr", "TRB%", "AST%", "STL%", "BLK%",
                                       "TOV%", "USG%",
                                       "WS", "BPM", "VORP"]]
#Remae certain columns for clarity
df_advanced_stats = df_advanced_stats.rename(columns={"G": "Games Played",
                                                      "PER":"Offensive Production per Minute",
                                                      "TS%": "True Shooting",
                                                      "AST%": "Field Goals Assisted",
                                                      "STL%": "Possessions end by steal",
                                                      "BLK%": "Two-point attempts blocked",
                                                      "TOV%": "Turnovers per 100 plays",
                                                      "USG%": "Plays Used",
                                                      "WS": "Win Shares",
                                                      "BPM": "Box Plus/Minus",
                                                      "VORP": "Value Over Replacement"})

#Remove duplicates

# --- Step 1: Define what "Total" looks like ---
# Basketball-Reference uses '2TM', '3TM', '4TM', or sometimes 'TOT'
total_tags = ['2TM', '3TM', '4TM', 'TOT']

# --- Step 2: Create a "Mapping Dictionary" ---
# We want to find the team where each player played the MOST games.

# Filter to get only the individual team rows (Exclude '2TM', etc.)
individual_teams_df = df_advanced_stats[~df_advanced_stats['Team'].isin(total_tags)].copy()

# Sort by Games ('G') in descending order
# So for a player, the team with the most games appears first
individual_teams_df = individual_teams_df.sort_values(by=['Player', 'Games Played'], ascending=[True, False])

# Drop duplicates to keep ONLY the team with the max games for each player
# This creates a list of unique players and their "Main Team"
main_team_map_df = individual_teams_df.drop_duplicates(subset=['Player'], keep='first')

# Convert this to a dictionary: {'Davion Mitchell': 'TOR', 'Caleb Martin': 'MIA', ...}
player_to_team_dict = dict(zip(main_team_map_df['Player'], main_team_map_df['Team']))

# --- Step 3: Update the "Total" Rows ---
# Now we go back to our main dataframe.

# Identify the rows that need fixing (the ones with '2TM', etc.)
mask_totals = df_advanced_stats['Team'].isin(total_tags)

# Apply the dictionary map ONLY to those rows
# This says: "Look at the Player name, find their Main Team in the dictionary, and paste it here."
df_advanced_stats.loc[mask_totals, 'Team'] = df_advanced_stats.loc[mask_totals, 'Player'].map(player_to_team_dict)

# If a player somehow wasn't in the map, fill NaNs back with original value
df_advanced_stats['Team'] = df_advanced_stats['Team'].fillna('Unknown')

# --- Step 4: Now Drop Duplicates (Keep 'First') ---
# Since the "Total" row usually comes first in Basketball-Reference,
# we can now safely keep it. It has the total stats AND the correct team.
df_advanced_stats_clean = df_advanced_stats.drop_duplicates(subset=['Player'], keep='first')

# --- Verify ---
print("--- ADVANCED STATS DUPLICATES REMOVED ---")
print(df_advanced_stats_clean.head())
df_advanced_stats_clean.info()

#Merge the stats data frame
df_stats_merged = pd.merge(
         df_stats_per_game_clean,
         df_advanced_stats_clean,
         how="inner",
         on="Player")
df_stats_merged.info()
print(df_stats_merged.head())


df_stats_merged = df_stats_merged.drop(columns=["Team_y", "Games Played_y"])
df_stats_merged = df_stats_merged.rename(columns={"Team_x": "Team",
                                                  "Games Played_x": "Games Played"})

# --- Final Data Cleaning Step ---

# 1. Drop the "Ghost" Row
# We remove any row where 'Age' is missing.
# This removes the single junk row causing the 736 vs 735 mismatch.
df_stats_final = df_stats_merged.dropna(subset=['Age'])

# 2. Fill Missing Percentages with 0.0
# We identify columns where NaN implies "0 attempts"
percentage_cols = ['FG%', '3P%', 'FT%', 'True Shooting', 'Turnovers per 100 plays']

# We fill those specific columns with 0.0
df_stats_final[percentage_cols] = df_stats_final[percentage_cols].fillna(0.0)

# 3. Verification
# Now, EVERY column should show exactly 735 non-null values.
print("--- Final Clean DataFrame Info ---")
df_stats_final.info()
df_stats_final.to_csv("stats_final.csv")

#MERGE STATS AND SALARIES
df_stats_salaries = pd.merge(
    df_stats_final,
    df_salaries,
    how="inner",
    on="Player"
)

df_stats_salaries.info()

#MERGE WITH STANDINGS
#Select the relevant columns
df_standings_clean = df_expanded_standings[["Team_Code", "Win_Pct"]]
df_final = pd.merge(
    left=df_stats_salaries,
    right=df_standings_clean,
    left_on="Team",
    right_on="Team_Code",
    how="left"

)
df_final = df_final.drop(columns=["Team_Code"])
df_final.info()

#Final save
df_final.to_csv("final_nba_dataset.csv")
