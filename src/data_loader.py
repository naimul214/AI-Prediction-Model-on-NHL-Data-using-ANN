import os
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_dir=None):
    """
    Loads NHL data, aggregates statistics to the team-game level to prevent 
    Cartesian product data duplication, and computes historical rolling averages 
    to prevent data leakage.
    
    Parameters:
    -----------
    data_dir : str, optional
        Path to the directory containing the CSV files. If None, it defaults
        to the local 'nhl_data' folder.
        
    Returns:
    --------
    X : pd.DataFrame
        Historical rolling feature matrix.
    y : pd.Series
        Binary target labels (1 for home win, 0 for away win).
    """
    if data_dir is None:
        # Resolve path relative to this script
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, 'nhl_data')
        
    logger.info(f"Loading datasets from: {data_dir}")
    
    # Define file paths
    game_path = os.path.join(data_dir, 'game.csv')
    teams_stats_path = os.path.join(data_dir, 'game_teams_stats.csv')
    goalie_stats_path = os.path.join(data_dir, 'game_goalie_stats.csv')
    skater_stats_path = os.path.join(data_dir, 'game_skater_stats.csv')
    
    # 1. Load the files
    try:
        game = pd.read_csv(game_path)
        logger.info(f"Loaded game.csv with {len(game)} rows.")
    except Exception as e:
        logger.error(f"Failed to load game.csv: {e}")
        raise e
        
    try:
        game_teams_stats = pd.read_csv(teams_stats_path)
        logger.info(f"Loaded game_teams_stats.csv with {len(game_teams_stats)} rows.")
    except Exception as e:
        logger.error(f"Failed to load game_teams_stats.csv: {e}")
        raise e
        
    try:
        game_goalie_stats = pd.read_csv(goalie_stats_path)
        logger.info(f"Loaded game_goalie_stats.csv with {len(game_goalie_stats)} rows.")
    except Exception as e:
        logger.error(f"Failed to load game_goalie_stats.csv: {e}")
        raise e
        
    # Skater stats is optional since it might not be in the workspace due to file size
    has_skater_stats = False
    if os.path.exists(skater_stats_path):
        try:
            game_skater_stats = pd.read_csv(skater_stats_path)
            has_skater_stats = True
            logger.info(f"Loaded game_skater_stats.csv with {len(game_skater_stats)} rows.")
        except Exception as e:
            logger.warning(f"Found game_skater_stats.csv but failed to load it: {e}")
    else:
        logger.info("game_skater_stats.csv not found. Falling back to team and goalie metrics.")

    # 2. Aggregate goalie stats to the team-game level
    logger.info("Aggregating goalie statistics...")
    goalie_agg = game_goalie_stats.groupby(['game_id', 'team_id']).agg({
        'saves': 'sum',
        'shots': 'sum'
    }).reset_index().rename(columns={'saves': 'goalie_saves', 'shots': 'goalie_shots_faced'})
    
    # 3. Aggregate skater stats if available
    if has_skater_stats:
        logger.info("Aggregating skater statistics...")
        skater_agg = game_skater_stats.groupby(['game_id', 'team_id']).agg({
            'goals': 'sum',
            'assists': 'sum'
        }).reset_index().rename(columns={'goals': 'skater_goals', 'assists': 'skater_assists'})
    
    # 4. Standardize the team-game level dataframe
    # game_teams_stats already has 2 rows per game (one for each team)
    team_game_df = game_teams_stats[['game_id', 'team_id', 'HoA', 'goals', 'shots', 'hits']].copy()
    
    # Merge with goalie stats
    team_game_df = pd.merge(team_game_df, goalie_agg, on=['game_id', 'team_id'], how='left')
    
    # Merge with skater stats if available
    if has_skater_stats:
        team_game_df = pd.merge(team_game_df, skater_agg, on=['game_id', 'team_id'], how='left')
        
    # Fill any NaNs with 0
    team_game_df = team_game_df.fillna(0)
    
    # 5. Merge date from game.csv for chronological sorting
    game_dates = game[['game_id', 'date_time_GMT']].copy()
    team_game_df = pd.merge(team_game_df, game_dates, on='game_id', how='left')
    
    # Convert date to datetime
    team_game_df['date_time_GMT'] = pd.to_datetime(team_game_df['date_time_GMT'])
    
    # 6. Compute rolling averages per team (shifting by 1 to prevent data leakage)
    logger.info("Computing 5-game rolling averages for each team...")
    team_game_df = team_game_df.sort_values(['team_id', 'date_time_GMT'])
    
    # Define metrics to average
    metrics = ['goals', 'shots', 'hits', 'goalie_saves', 'goalie_shots_faced']
    if has_skater_stats:
        metrics.extend(['skater_goals', 'skater_assists'])
        
    # Calculate rolling metrics
    for metric in metrics:
        # shift(1) ensures we only look at games prior to the current game
        team_game_df[f'{metric}_rolling_5'] = (
            team_game_df.groupby('team_id')[metric]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
        )
        
    # Fill any initial rolling values (first game of a team) with the global average of that metric
    for metric in metrics:
        global_avg = team_game_df[metric].mean()
        team_game_df[f'{metric}_rolling_5'] = team_game_df[f'{metric}_rolling_5'].fillna(global_avg)
        
    # 7. Pivot the data back to game-level rows
    # We want one row per game containing home team rolling averages and away team rolling averages
    logger.info("Structuring features to the game level (Home vs Away)...")
    
    rolling_cols = [f'{m}_rolling_5' for m in metrics]
    
    home_df = team_game_df[team_game_df['HoA'] == 'home'][['game_id', 'team_id'] + rolling_cols].copy()
    home_df.rename(columns={col: f'home_{col}' for col in rolling_cols}, inplace=True)
    home_df.rename(columns={'team_id': 'home_team_id'}, inplace=True)
    
    away_df = team_game_df[team_game_df['HoA'] == 'away'][['game_id', 'team_id'] + rolling_cols].copy()
    away_df.rename(columns={col: f'away_{col}' for col in rolling_cols}, inplace=True)
    away_df.rename(columns={'team_id': 'away_team_id'}, inplace=True)
    
    # Merge home and away rolling averages back with the game dataframe
    game_features = game[['game_id', 'home_team_id', 'away_team_id', 'outcome']].copy()
    game_features = pd.merge(game_features, home_df, on=['game_id', 'home_team_id'], how='inner')
    game_features = pd.merge(game_features, away_df, on=['game_id', 'away_team_id'], how='inner')
    
    # 8. Define target variable y
    # 1 for home win, 0 for away win
    y = game_features['outcome'].map({
        'home win REG': 1,
        'home win OT': 1,
        'away win REG': 0,
        'away win OT': 0
    })
    # Fill any other outcomes (e.g. ties or unfinished) with 0
    y = y.fillna(0)
    
    # Define features matrix X
    feature_cols = [f'home_{col}' for col in rolling_cols] + [f'away_{col}' for col in rolling_cols]
    X = game_features[feature_cols]
    
    logger.info(f"Preprocessing completed. Shape of X: {X.shape}, Shape of y: {y.shape}")
    return X, y

if __name__ == '__main__':
    # Test data loader
    X, y = load_and_preprocess_data()
    print("Features preview:")
    print(X.head())
    print("Target distribution:")
    print(y.value_counts(normalize=True))
