import pandas as pd

df = pd.read_csv("/Users/aldinfazlic/Desktop/projects/nba/nba_games.csv")

df.dropna(inplace=True)  # Remove rows with missing value

df['wlb_home'] = df['wl_home'].map({'W': 1, 'L': 0})
df['wlb_away'] = df['wl_away'].map({'W': 1, 'L': 0})

# only use data from games after 2014
df["season_year"] = df['season_id'].astype(str).str[1:].astype(int)
filtered_df = df[df["season_year"] >= 2014]

# remove non-nba teams and duplicate clippers
filtered_df = filtered_df[~filtered_df["team_name_home"].isin(["Team Durant", "West NBA All Stars West", "Barcelona FC Barcelona Lassa", "Flamengo Flamengo", "Team Giannis", "Berlin Alba Berlin", "Team Stephen", "East NBA All Stars East", "Milano Olimpia Milano", "New Orleans/Oklahoma City Hornets", "Team LeBron", "Istanbul Fenerbahce Ulker", "Madrid Real Madrid"])]
filtered_df.replace("LA Clippers", "Los Angeles Clippers", inplace = True)

#print(filtered_df.head())
teams = set(filtered_df["team_name_home"].values)
championship_winners = [("Golden State Warriors", 2014), ("Golden State Warriors", 2015), ("Cleveland Cavaliers", 2016), 
                        ("Golden State Warriors", 2017), ("Golden State Warriors", 2018), ("Toronto Raptors", 2019), ("Los Angeles Lakers", 2020), 
                        ("Milwaukee Bucks", 2021), ("Golden State Warriors", 2022), ("Denver Nuggets", 2023)]

# organize data by team and season
season_stats = filtered_df.groupby(['season_year', 'team_name_home']).agg(
    total_points=('pts_home', 'sum'),
    assists=('ast_home', 'sum'),
    rebounds=('reb_home', 'sum'),
    steals=('stl_home', 'sum'),
    blocks=('blk_home', 'sum'),
    games_played=('game_id', 'count'),
    wins=('wlb_home', 'sum'), 
    total_points_against=('pts_away', 'sum')  # Total points allowed
).reset_index()

season_stats['points_per_game'] = season_stats['total_points'] / season_stats['games_played']
season_stats['assists_per_game'] = season_stats['assists'] / season_stats['games_played']
season_stats['rebounds_per_game'] = season_stats['rebounds'] / season_stats['games_played']
season_stats['win_loss_ratio'] = season_stats['wins'] / season_stats['games_played']
season_stats['point_differential'] = season_stats['total_points'] - season_stats['total_points_against']
season_stats['championship_winner'] = season_stats.apply(
    lambda row: 1 if (row['team_name_home'], row['season_year']) in championship_winners else 0, axis=1
)

print(season_stats.head())

# ML MODEL BELOW
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE


selected_features = ['points_per_game', 'assists_per_game', 'rebounds_per_game', 'win_loss_ratio', 'point_differential']
X = season_stats[selected_features] # features / input 
y = season_stats['championship_winner'] # target / output / prediction 


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80/20 split


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


#model initialization
model = LogisticRegression()

#train model
model.fit(X_train_resampled, y_train_resampled)

#evaluate
y_pred = model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Print a detailed classification report
print(classification_report(y_test, y_pred))




