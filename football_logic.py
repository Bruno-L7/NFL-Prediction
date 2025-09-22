import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, HuberRegressor, PoissonRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, log_loss, accuracy_score, mean_squared_error
from scipy.stats import poisson
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from statsmodels.discrete.discrete_model import NegativeBinomial
import statsmodels.api as sm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import xgboost as xgb
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Input, Concatenate, Dropout, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import BinaryCrossentropy, MeanSquaredError
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.base import clone
from sklearn.model_selection import KFold


warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- CONFIGURATION ---
HISTORICAL_YEARS = range(2018, 2025)
CURRENT_SEASON = 2024
ROLLING_WINDOW = 5
EWMA_ALPHA = 0.3
OPP_ADJUSTMENT_LAMBDA = 1.0
AR1_RHO = 0.9

def create_ensemble(models, weights=None, voting='soft'):
    """Create an ensemble of models"""
    if voting == 'soft' and all(hasattr(m, 'predict_proba') for m in models.values()):
        return VotingClassifier(
            estimators=[(name, model) for name, model in models.items()],
            voting=voting,
            weights=weights
        )
    else:
        return VotingRegressor(
            estimators=[(name, model) for name, model in models.items()],
            weights=weights
        )

def generate_oof_predictions(X, y, base_models, n_splits=5):
    """Generate out-of-fold predictions for stacking"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros((X.shape[0], len(base_models)))
    
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for j, (name, model) in enumerate(base_models.items()):
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            
            if hasattr(model_clone, 'predict_proba'):
                oof_preds[val_idx, j] = model_clone.predict_proba(X_val)[:, 1]
            else:
                oof_preds[val_idx, j] = model_clone.predict(X_val)
    
    return oof_preds

def train_stacked_model(X, y, base_models, meta_model=None):
    """Train a stacked model with meta-learner"""
    # Generate out-of-fold predictions
    oof_preds = generate_oof_predictions(X, y, base_models)
    
    # Train meta-model on OOF predictions
    if meta_model is None:
        meta_model = LogisticRegression(max_iter=1000, random_state=42)
    
    meta_model.fit(oof_preds, y)
    
    # Refit base models on full data
    for name, model in base_models.items():
        model.fit(X, y)
    
    return meta_model, base_models

def predict_stacked_model(X, base_models, meta_model):
    """Make predictions using stacked model"""
    base_preds = np.zeros((X.shape[0], len(base_models)))
    
    for j, (name, model) in enumerate(base_models.items()):
        if hasattr(model, 'predict_proba'):
            base_preds[:, j] = model.predict_proba(X)[:, 1]
        else:
            base_preds[:, j] = model.predict(X)
    
    return meta_model.predict_proba(base_preds)[:, 1] if hasattr(meta_model, 'predict_proba') else meta_model.predict(base_preds)

def train_ensemble_win_prob_model(X_train, y_train, X_val, y_val):
    """Train an ensemble of models for win probability"""
    models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgboost': xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            max_depth=4,
            eta=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_lambda=1,
            reg_alpha=0,
            n_estimators=1000,
            random_state=42
        )
    }
    
    # Train individual models
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    # Create ensemble
    ensemble = create_ensemble(models, voting='soft')
    ensemble.fit(X_train, y_train)
    
    return ensemble, models

def calibrate_model_probas(model, X_train, y_train, X_val, y_val, method='sigmoid'):
    """Calibrate model probabilities using Platt scaling or isotonic regression"""
    if hasattr(model, 'predict_proba'):
        # Calibrate using validation data
        calibrated = CalibratedClassifierCV(model, method=method, cv='prefit')
        calibrated.fit(X_val, y_val)
        return calibrated
    return model

def get_pbp_data(years):
    """Fetches play-by-play NFL data for a given range of years."""
    print("Fetching historical play-by-play data...")
    try:
        df = nfl.import_pbp_data(years, downcast=True, cache=False)

        # ensure lowercase alias exists when upstream code expects 'epa' or 'success'
        if 'EPA' in df.columns and 'epa' not in df.columns:
            df['epa'] = df['EPA']
        if 'Success' in df.columns and 'success' not in df.columns:
            df['success'] = df['Success']

        print("DEBUG: raw dataframe loaded from DB (PBP Data)")
        print("  shape:", df.shape)
        print("  columns:", df.columns.tolist())
        print("  sample rows:")
        print(df.head(5).to_string(index=False))
        # check the specific columns that later become sequence features
        # guarantee the variable exists even if aggregation fails
        team_def_stats = pd.DataFrame(columns=['season','week','team','opp_def_epa','opp_def_success'])
        try:
            # only attempt aggregation if defteam + season + week exist
            if all(col in df.columns for col in ['defteam','season','week']):
                agg = {}
                if 'epa' in df.columns:
                    agg['epa'] = 'mean'
                if 'success' in df.columns:
                    agg['success'] = 'mean'

                if agg:
                    team_def_stats = (
                        df.dropna(subset=['defteam','season','week'])
                          .groupby(['season','week','defteam'])
                          .agg(agg)
                          .reset_index()
                          .rename(columns={'defteam': 'team',
                                           'epa': 'opp_def_epa',
                                           'success': 'opp_def_success'})
                    )
                    # normalize team codes (uppercase) so merges are more reliable
                    team_def_stats['team'] = team_def_stats['team'].astype(str).str.upper()
            else:
                print("WARN: PBP missing 'defteam' or season/week -> team_def_stats left empty.")
        except Exception as e:
            print("WARN building team_def_stats:", e)

        print("✅ Data fetched successfully.")
        return df, team_def_stats
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def get_weekly_data(years, team_def_stats):
    """Fetches weekly player data for a given range of years."""
    print("Fetching weekly player data...")
    try:
        df = nfl.import_weekly_data(years)

        print("DEBUG: raw dataframe loaded from DB (Weekly Data)")
        print("  shape:", df.shape)
        print("  columns:", df.columns.tolist())
        print("  sample rows:")
        print(df.head(5).to_string(index=False))
        # check the specific columns that later become sequence features
        for c in ["opp_def_epa", "opp_def_success", "rolling_yds", "rolling_attempts"]:
            if c in df.columns:
                nna = df[c].isna().sum()
                print(f"  {c} NaNs: {nna} / {len(df)}")
            else:
                print(f"  {c} NOT FOUND in df")
        # if large, dump the rows where key features are NaN to file for inspection
        problem_mask = df[["opp_def_epa","opp_def_success"]].isna().all(axis=1) if set(["opp_def_epa","opp_def_success"]).issubset(df.columns) else None
        if problem_mask is not None and problem_mask.any():
            df.loc[problem_mask, :].head(50).to_csv("debug_missing_def_stats.csv", index=False)
            print("  -> wrote debug_missing_def_stats.csv with sample problematic rows")
            # The weekly dataframe is named 'df' in this function's scope.
            if df is not None and not team_def_stats.empty:
                df = df.merge(
                    team_def_stats,
                    left_on=['season', 'week', 'opponent_team'],
                    right_on=['season', 'week', 'team'],
                    how='left'
                )
                # drop helper 'team' column from the merge
                if 'team' in df.columns:
                    df = df.drop(columns=['team'])
            else:
                print("WARNING: team_def_stats empty or weekly data missing; opp_def_epa not merged.")

        print("✅ Weekly player data fetched successfully.")
        return df
    except Exception as e:
        print(f"Error fetching weekly data: {e}")
        return pd.DataFrame()

def calculate_custom_losses(y_true, y_pred, loss_type='mse'):
    """Calculate various loss functions for evaluation."""
    if loss_type == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif loss_type == 'huber':
        delta = 1.0
        residual = np.abs(y_true - y_pred)
        huber_loss = np.where(residual <= delta,
                              0.5 * residual ** 2,
                              delta * (residual - 0.5 * delta))
        return np.mean(huber_loss)
    elif loss_type == 'poisson':
        y_pred = np.maximum(y_pred, 1e-10)
        return -np.mean(poisson.logpmf(y_true.astype(int), y_pred))
    else:
        return mean_squared_error(y_true, y_pred)

def calculate_ewma(series, alpha=EWMA_ALPHA):
    """Calculate Exponentially Weighted Moving Average."""
    return series.ewm(alpha=alpha, adjust=False).mean()

def calculate_ar1_strength(series, rho=0.9):
    """Calculate AR(1) forecast for team strength"""
    if len(series) < 2:
        return np.nan
    # Fit AR(1) model
    model = sm.tsa.ARIMA(series.dropna(), order=(1, 0, 0))
    try:
        results = model.fit()
        rho_est = results.params[0]  # AR coefficient
        # Forecast next value
        forecast = rho_est * series.iloc[-1]
    except:
        # Fallback to simple persistence if AR fails
        forecast = series.iloc[-1] * rho
    return forecast

def create_team_embeddings(team_stats_df, embedding_dim=10):
    """Create team embedding layer and encoder"""
    teams = sorted(team_stats_df['team'].unique())
    team_encoder = LabelEncoder()
    team_encoder.fit(teams)
    
    num_teams = len(teams) + 1  # +1 for unknown teams
    team_embedding = Embedding(input_dim=num_teams, output_dim=embedding_dim, name='team_embedding')
    
    return team_encoder, team_embedding, num_teams

def create_player_embeddings(player_data, embedding_dim=15):
    """Create player embedding layer and encoder"""
    players = sorted(player_data['player_id'].unique())
    player_encoder = LabelEncoder()
    player_encoder.fit(players)
    
    num_players = len(players) + 1  # +1 for unknown players
    player_embedding = Embedding(input_dim=num_players, output_dim=embedding_dim, name='player_embedding')
    
    return player_encoder, player_embedding, num_players

def prepare_team_sequences(team_stats, sequence_length=5, target='win'):
    """Prepare sequential data for team RNN models with data validation"""
    sequences = []
    targets = []
    team_ids = []
    
    for team in team_stats['team'].unique():
        team_data = team_stats[team_stats['team'] == team].sort_values(['season', 'week'])
        
        for i in range(len(team_data) - sequence_length):
            seq = team_data.iloc[i:i+sequence_length]
            target_val = team_data.iloc[i+sequence_length][target] if target in team_data.columns else None
            
            # Extract features and check for NaN values
            feature_values = seq.drop(['team', 'game_id', 'season', 'week'], axis=1).values
            
            # Skip sequences with NaN values
            if np.isnan(feature_values).any():
                continue
                
            sequences.append(feature_values)
            if target_val is not None:
                targets.append(target_val)
            team_ids.append(team)
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    # Normalize the sequences
    sequences_mean = np.mean(sequences, axis=(0, 1))
    sequences_std = np.std(sequences, axis=(0, 1)) + 1e-8  # Add small value to avoid division by zero
    sequences = (sequences - sequences_mean) / sequences_std
    
    return sequences, targets, team_ids

def prepare_player_sequences(player_data, sequence_length=4, target_col='passing_yards'):
    """Prepare sequential data for player RNN models with data validation"""
    sequences = []
    targets = []
    player_ids = []
    
    # Calculate normalization parameters on the entire dataset first
    all_features = []
    for player in player_data['player_id'].unique():
        player_games = player_data[player_data['player_id'] == player].sort_values(['season', 'week'])
        
        for i in range(len(player_games) - sequence_length):
            seq = player_games.iloc[i:i+sequence_length]
            features = seq[['rolling_yds', 'rolling_attempts', 'opp_def_epa', 'opp_def_success']].values
            all_features.append(features)
    
    all_features = np.concatenate(all_features, axis=0)
    feature_means = np.mean(all_features, axis=0)
    feature_stds = np.std(all_features, axis=0) + 1e-8
    
    # Now process each sequence with consistent normalization
    for player in player_data['player_id'].unique():
        player_games = player_data[player_data['player_id'] == player].sort_values(['season', 'week'])
        
        for i in range(len(player_games) - sequence_length):
            seq = player_games.iloc[i:i+sequence_length]
            target_val = player_games.iloc[i+sequence_length][target_col]
            
            features = seq[['rolling_yds', 'rolling_attempts', 'opp_def_epa', 'opp_def_success']].values
            
            # Skip sequences with NaN values
            if np.isnan(features).any() or np.isnan(target_val):
                continue
                
            # Normalize features consistently
            features = (features - feature_means) / feature_stds
                
            sequences.append(features)
            targets.append(target_val)
            player_ids.append(player)
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    return sequences, targets, player_ids

def create_team_rnn_model(input_shape, num_teams, embedding_dim=10, rnn_units=32):
    """Create RNN model with team embeddings for sequence prediction"""
    # Team input
    team_input = Input(shape=(1,), name='team_input')
    team_embedding = Embedding(input_dim=num_teams, output_dim=embedding_dim)(team_input)
    team_flatten = Flatten()(team_embedding)
    
    # Sequence input
    sequence_input = Input(shape=input_shape, name='sequence_input')
    
    # Add BatchNormalization to stabilize training
    normalized_sequence = BatchNormalization()(sequence_input)
    
    lstm_layer = LSTM(rnn_units, return_sequences=False)(normalized_sequence)
    
    # Concatenate
    concatenated = Concatenate()([team_flatten, lstm_layer])
    
    # Dense layers with regularization
    dense1 = Dense(16, activation='relu')(concatenated)
    dropout = Dropout(0.3)(dense1)
    output = Dense(1, activation='sigmoid')(dropout)  # For binary classification
    
    model = Model(inputs=[team_input, sequence_input], outputs=output)
    
    # Use a lower learning rate to prevent exploding gradients
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])
    
    return model

def create_player_rnn_model(input_shape, num_players, embedding_dim=15, rnn_units=32):
    """Create RNN model with player embeddings for yardage prediction"""
    # Player input
    player_input = Input(shape=(1,), name='player_input')
    player_embedding = Embedding(input_dim=num_players, output_dim=embedding_dim)(player_input)
    player_flatten = Flatten()(player_embedding)
    
    # Sequence input
    sequence_input = Input(shape=input_shape, name='sequence_input')
    
    # Process sequence with multiple layers
    x = LSTM(64, return_sequences=True)(sequence_input)
    x = Dropout(0.3)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    
    # Concatenate with player embedding
    concatenated = Concatenate()([player_flatten, x])
    
    # Dense layers with regularization
    x = Dense(32, activation='relu')(concatenated)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(1, activation='linear')(x)  # For regression
    
    model = Model(inputs=[player_input, sequence_input], outputs=output)
    
    # Use gradient clipping to prevent explosion
    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)  # Added clipvalue
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def train_neural_win_prob_model(team_stats):
    """Train neural network with embeddings and sequences for win probability"""
    print("\n--- Training Neural Win Probability Model with Embeddings & Sequences ---")
    
    # Prepare sequences
    sequences, targets, team_ids = prepare_team_sequences(team_stats, sequence_length=5, target='win')
    
    # Encode teams
    team_encoder = LabelEncoder()
    team_encoder.fit(team_stats['team'].unique())
    encoded_teams = team_encoder.transform(team_ids)
    
    # Split data
    X_team_train, X_team_test, X_seq_train, X_seq_test, y_train, y_test = train_test_split(
        encoded_teams, sequences, targets, test_size=0.2, random_state=42, stratify=targets
    )
    
    # Create and train model
    num_teams = len(team_encoder.classes_) + 1
    model = create_team_rnn_model((X_seq_train.shape[1], X_seq_train.shape[2]), num_teams)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print(f"DEBUG: Any NaN in NN win prob arrays? X_seq_train: {np.isnan(X_seq_train).any()}, X_seq_test: {np.isnan(X_seq_test).any()}")

    history = model.fit(
        [X_team_train, X_seq_train], y_train,
        validation_data=([X_team_test, X_seq_test], y_test),
        epochs=2, batch_size=32, callbacks=[early_stop], verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate([X_team_test, X_seq_test], y_test, verbose=0)
    print(f"✅ Neural Win Probability Model trained.")
    print(f"   Loss: {loss:.4f}, Accuracy: {accuracy:.3f}")
    
    return model, team_encoder, history

def train_neural_player_model(player_data, position, target_col):
    """Train neural network with embeddings and sequences for player prediction"""
    print(f"\n--- Training Neural {position} Model with Embeddings & Sequences ---")
    
    impute_cols = ['opp_def_epa','opp_def_success','rolling_yds','rolling_attempts']
    for c in impute_cols:
        if c in player_data.columns:
            # fill with the mean for the player's opponent, then the global mean, then 0.
            if 'opponent_team' in player_data.columns:
                player_data[c] = player_data.groupby('opponent_team')[c].transform(lambda s: s.fillna(s.mean()))
            player_data[c] = player_data[c].fillna(player_data[c].mean())
            player_data[c] = player_data[c].fillna(0.0)

    # Prepare sequences
    sequences, targets, player_ids = prepare_player_sequences(player_data, sequence_length=4, target_col=target_col)
    
    if len(sequences) == 0:
        print(f"❌ No valid sequences for {position} position")
        return None, None, None
    
    # Encode players
    player_encoder = LabelEncoder()
    player_encoder.fit(player_data['player_id'].unique())
    encoded_players = player_encoder.transform(player_ids)
    
    # Split data
    X_player_train, X_player_test, X_seq_train, X_seq_test, y_train, y_test = train_test_split(
        encoded_players, sequences, targets, test_size=0.2, random_state=42
    )
    
    # Scale targets to improve numerical stability
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # Data quality check
    print(f"Data summary for {position}:")
    print(f"  Number of training samples: {len(y_train)}")
    print(f"  Training target mean: {np.mean(y_train):.2f}")
    print(f"  Training target std: {np.std(y_train):.2f}")
    print(f"  Training target min/max: {np.min(y_train):.2f}/{np.max(y_train):.2f}")
    print(f"  Number of test samples: {len(y_test)}")
    print(f"  Test target mean: {np.mean(y_test):.2f}")

    # Check if the mean is reasonable
    if position == 'QB' and np.mean(y_train) < 100:
        print("⚠️  Warning: QB yardage data seems too low - check data processing")
    elif position == 'RB' and np.mean(y_train) < 30:
        print("⚠️  Warning: RB yardage data seems too low - check data processing")
    elif position in ['WR', 'TE'] and np.mean(y_train) < 40:
        print(f"⚠️  Warning: {position} yardage data seems too low - check data processing")
    
    print(f"  NaN in X_seq_train: {np.isnan(X_seq_train).any()}")
    print(f"  NaN in y_train_scaled: {np.isnan(y_train_scaled).any()}")
    print(f"  NaN in X_seq_test: {np.isnan(X_seq_test).any()}")
    print(f"  NaN in y_test_scaled: {np.isnan(y_test_scaled).any()}")

    print(f"  X_seq_train range: {X_seq_train.min():.4f} to {X_seq_train.max():.4f}")
    print(f"  y_train_scaled range: {y_train_scaled.min():.4f} to {y_train_scaled.max():.4f}")

    print("Checking for NaN in feature columns:")
    feature_columns = ['rolling_yds', 'rolling_attempts', 'opp_def_epa', 'opp_def_success']
    for i, col in enumerate(feature_columns):
        nan_count = np.isnan(X_seq_train[:, :, i]).sum()
        print(f"  {col}: {nan_count} NaN values in training sequences")
        
        nan_count_test = np.isnan(X_seq_test[:, :, i]).sum()
        print(f"  {col}: {nan_count_test} NaN values in test sequences")
    
    # Check a few specific sequences to see what's happening
    print("\nSample sequence values (first sequence):")
    sample_sequence = X_seq_train[0]
    for i, col in enumerate(feature_columns):
        print(f"  {col}: {sample_sequence[:, i]}")

    # Create and train model
    num_players = len(player_encoder.classes_) + 1
    model = create_player_rnn_model((X_seq_train.shape[1], X_seq_train.shape[2]), num_players)
    
    # Add callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
    
    print(f"Training on {X_seq_train.shape[0]} sequences")
    
    print(f"DEBUG: Any NaN in NN player arrays? X_seq_train: {np.isnan(X_seq_train).any()}, X_seq_test: {np.isnan(X_seq_test).any()}")

    history = model.fit(
        [X_player_train, X_seq_train], y_train_scaled,
        validation_data=([X_player_test, X_seq_test], y_test_scaled),
        epochs=2,
        batch_size=32, 
        callbacks=[early_stop, reduce_lr], 
        verbose=1
    )
    
    # Evaluate
    loss, mae = model.evaluate([X_player_test, X_seq_test], y_test_scaled, verbose=0)
    print(f"✅ Neural {position} Model trained.")
    print(f"   Loss: {loss:.4f}, MAE: {mae:.3f}")
    
    # Make some sample predictions to verify they're reasonable
    sample_preds_scaled = model.predict([X_player_test[:5], X_seq_test[:5]])
    sample_preds = target_scaler.inverse_transform(sample_preds_scaled).flatten()
    print(f"   Sample predictions: {sample_preds}")
    print(f"   Actual values: {y_test[:5]}")
    
    return model, player_encoder, player_data, target_scaler  # Return the scaler too

def build_markov_chain_model(pbp_df):
    """Build a Markov chain model for football game states"""
    print("\n--- Building Markov Chain Expected Points Model ---")
    
    # Filter to relevant plays and create a copy to avoid fragmentation
    df = pbp_df[pbp_df['play_type'].isin(['pass', 'run', 'qb_kneel', 'qb_spike'])].copy()
    df = df[df['down'].notna() & df['ydstogo'].notna() & df['yardline_100'].notna()]
    
    # Precompute all state columns at once to avoid fragmentation
    state_data = pd.DataFrame({
        'state_yardline': (df['yardline_100'] // 5) * 5,  # 5-yard bins
        'state_down': df['down'],
        'state_ydstogo': (df['ydstogo'] // 2) * 2  # 2-yard bins
    })
    
    # Create state identifier
    state_data['state_id'] = (
        state_data['state_yardline'].astype(str) + '_' + 
        state_data['state_down'].astype(str) + '_' + 
        state_data['state_ydstogo'].astype(str)
    )
    
    # Add the state_id to the main DataFrame
    df['state_id'] = state_data['state_id']
    
    # Calculate next state and points scored
    df['next_state_id'] = df.groupby('game_id')['state_id'].shift(-1)
    
    # Calculate points scored in a vectorized way
    df['points_scored'] = (
        df['touchdown'] * 6 + 
        df['field_goal_attempt'] * df['field_goal_result'].eq('made') * 3
    )
    
    # Build transition matrix
    states = sorted(df['state_id'].unique())
    n_states = len(states)
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    
    # Initialize transition matrix and reward vector
    transition_counts = lil_matrix((n_states, n_states))
    reward_sums = lil_matrix((n_states, n_states))
    
    # Count transitions and rewards
    valid_transitions = df[df['next_state_id'].notna() & df['next_state_id'].isin(state_to_idx)]
    
    for _, row in valid_transitions.iterrows():
        i = state_to_idx[row['state_id']]
        j = state_to_idx[row['next_state_id']]
        transition_counts[i, j] += 1
        reward_sums[i, j] += row['points_scored']
    
    # Convert to probabilities and expected rewards
    transition_matrix = lil_matrix((n_states, n_states))
    expected_rewards = lil_matrix((n_states, n_states))
    
    for i in range(n_states):
        total = transition_counts[i].sum()
        if total > 0:
            for j in range(n_states):
                if transition_counts[i, j] > 0:
                    transition_matrix[i, j] = transition_counts[i, j] / total
                    expected_rewards[i, j] = reward_sums[i, j] / transition_counts[i, j]
    
    # Solve for expected points using Bellman equation
    # EP(s) = Σ P(s→s')[R(s→s') + EP(s')]
    identity = lil_matrix(np.eye(n_states))
    A = identity - transition_matrix
    b = transition_matrix.multiply(expected_rewards).sum(axis=1)
    
    # Solve the linear system
    try:
        EP = spsolve(csr_matrix(A), b)
    except:
        # Fallback if the matrix is singular
        print("Warning: Singular matrix encountered, using fallback method")
        EP = np.zeros(n_states)
    
    # Create EP lookup dictionary
    ep_lookup = {state: ep_value for state, ep_value in zip(states, EP)}
    
    print(f"✅ Markov chain model built with {n_states} states")
    return ep_lookup

def calculate_epa(pbp_df, ep_lookup):
    """Calculate Expected Points Added for each play"""
    df = pbp_df.copy()
    
    # Precompute all state columns at once
    state_data = pd.DataFrame({
        'state_yardline': (df['yardline_100'] // 5) * 5,
        'state_down': df['down'],
        'state_ydstogo': (df['ydstogo'] // 2) * 2
    })
    
    # Create state identifier
    state_data['state_id'] = (
        state_data['state_yardline'].astype(str) + '_' + 
        state_data['state_down'].astype(str) + '_' + 
        state_data['state_ydstogo'].astype(str)
    )
    
    # Add to main DataFrame
    df['state_id'] = state_data['state_id']
    
    # Get next state
    df['next_state_id'] = df.groupby('game_id')['state_id'].shift(-1)
    
    # Calculate EP for current and next state
    df['EP'] = df['state_id'].map(ep_lookup).fillna(0)
    df['next_EP'] = df['next_state_id'].map(ep_lookup).fillna(0)
    
    # Calculate EPA
    df['points_scored'] = (
        df['touchdown'] * 6 + 
        df['field_goal_attempt'] * df['field_goal_result'].eq('made') * 3
    )
    df['epa'] = df['next_EP'] - df['EP'] + df['points_scored']
    
    return df

def add_epa_features(pbp_df):
    """Add EPA-based features to the play-by-play data"""
    ep_lookup = build_markov_chain_model(pbp_df)
    pbp_df_with_epa = calculate_epa(pbp_df, ep_lookup)
    return pbp_df_with_epa

class HierarchicalPlayerModel(BaseEstimator, RegressorMixin):
    """Hierarchical model for player effects with random intercepts"""
    
    def __init__(self, formula, group_var='player_id'):
        self.formula = formula
        self.group_var = group_var
        self.model = None
        self.fitted_models = {}
        self.fallback_features = None # NEW: Add attribute to store feature names
        
    def fit(self, X, y, groups=None):
        df = X.copy()
        target_var = self.formula.split('~')[0].strip()
        df[target_var] = y
        
        if groups is not None:
            df[self.group_var] = groups
            
        try:
            self.model = smf.mixedlm(self.formula, df, groups=df[self.group_var])
            self.fitted_model = self.model.fit(reml=True)
            return self
        except Exception as e:
            print(f"Error fitting hierarchical model: {e}")
            # Fall back to regular regression
            X_with_const = sm.add_constant(X)
            
            # Check for duplicate columns and make them unique
            if X_with_const.columns.duplicated().any():
                new_columns = []
                for i, col in enumerate(X_with_const.columns):
                    if list(X_with_const.columns).count(col) > 1:
                        new_columns.append(f"{col}_{i}")
                    else:
                        new_columns.append(col)
                X_with_const.columns = new_columns
                
            self.fallback_features = X_with_const.columns
            self.fallback_model = sm.OLS(y, X_with_const).fit()
            return self
    
    def predict(self, X, groups=None):
        if hasattr(self, 'fitted_model'):
            df = X.copy()
            target_variable = self.formula.split('~')[0].strip()
            df[target_variable] = 0
            
            if groups is not None:
                df[self.group_var] = groups
                
            try:
                return self.fitted_model.predict(df)
            except Exception as e:
                print(f"DEBUG: Hierarchical prediction failed inside model: {e}")
                return np.full(X.shape[0], np.mean(self.fitted_model.fittedvalues))

        elif hasattr(self, 'fallback_model'):
            # This section is now much more robust
            X_with_const = sm.add_constant(X, has_constant='add')
            
            # Check for and handle duplicate columns
            if X_with_const.columns.duplicated().any():
                # Make column names unique by appending suffixes
                new_columns = []
                for i, col in enumerate(X_with_const.columns):
                    if list(X_with_const.columns).count(col) > 1:
                        new_columns.append(f"{col}_{i}")
                    else:
                        new_columns.append(col)
                X_with_const.columns = new_columns
            
            # Ensure fallback_features doesn't have duplicates
            if self.fallback_features.duplicated().any():
                self.fallback_features = pd.Index([f"{col}_{i}" if list(self.fallback_features).count(col) > 1 else col 
                                                for i, col in enumerate(self.fallback_features)])
            
            print(f"DEBUG: X_with_const columns: {X_with_const.columns.tolist()}")
            print(f"DEBUG: fallback_features: {self.fallback_features.tolist()}")
            print(f"DEBUG: Duplicates in X_with_const: {X_with_const.columns.duplicated().any()}")
            print(f"DEBUG: Duplicates in fallback_features: {self.fallback_features.duplicated().any()}")

            # NEW: Realign the prediction data to match the training data's columns perfectly
            X_aligned = X_with_const.reindex(columns=self.fallback_features, fill_value=0)
            return self.fallback_model.predict(X_aligned)
        
def adjust_for_opponent(stats_df, metric_col, team_col='team', opp_col='opponent', lambda_reg=OPP_ADJUSTMENT_LAMBDA):
    """
    Adjust metrics for opponent strength using ridge regression.
    Returns a DataFrame with opponent-adjusted metrics.
    """
    df = stats_df.copy()
    df = df[df[metric_col].notna()]
    
    teams = sorted(df[team_col].unique())
    opponents = sorted(df[opp_col].unique())
    
    X_team = pd.get_dummies(df[team_col], prefix='team')
    X_opp = pd.get_dummies(df[opp_col], prefix='opp')
    
    for team in teams:
        if f'team_{team}' not in X_team.columns:
            X_team[f'team_{team}'] = 0
    for opp in opponents:
        if f'opp_{opp}' not in X_opp.columns:
            X_opp[f'opp_{opp}'] = 0
    
    X = pd.concat([X_team, X_opp], axis=1)
    y = df[metric_col]
    
    model = Ridge(alpha=lambda_reg, fit_intercept=True)
    model.fit(X, y)
    
    team_coefs = {}
    for team in teams:
        col_name = f'team_{team}'
        if col_name in X.columns:
            team_coefs[team] = model.coef_[X.columns.get_loc(col_name)]
        else:
            team_coefs[team] = 0
    
    opp_effects = {}
    for opp in opponents:
        col_name = f'opp_{opp}'
        if col_name in X.columns:
            opp_effects[opp] = model.coef_[X.columns.get_loc(col_name)]
        else:
            opp_effects[opp] = 0
    
    avg_opp_effect = np.mean(list(opp_effects.values()))
    
    df[f'adj_{metric_col}'] = df.apply(
        lambda row: row[metric_col] - opp_effects.get(row[opp_col], 0) + avg_opp_effect,
        axis=1
    )
    
    return df, team_coefs, opp_effects

def train_enhanced_win_prob_model(pbp_df, use_neural=False, use_stacking=True, calibrate=True):
    """Enhanced version with EWMA and opponent adjustment."""
    print("\n--- Training Enhanced Binary Game Outcome Model ---")
    
    team_game_stats = pbp_df.groupby(['game_id', 'posteam', 'defteam', 'season', 'week']).agg(
        offensive_epa=('epa', 'mean'),
        offensive_success_rate=('success', 'mean'),
        expected_points_added=('epa', 'mean'),
        yards_gained=('yards_gained', 'sum'),
        turnovers=('interception', 'sum')
    ).reset_index()
    
    team_game_stats['turnover_ratio'] = team_game_stats['turnovers'] / (team_game_stats['yards_gained'] + 1)
    team_game_stats['opponent'] = team_game_stats['defteam']
    
    metrics_to_adjust = ['offensive_epa', 'offensive_success_rate', 'turnover_ratio', 'expected_points_added']
    for metric in metrics_to_adjust:
        team_game_stats, _, _ = adjust_for_opponent(team_game_stats, metric, 'posteam', 'opponent')
    
    off_stats = team_game_stats.rename(columns={
        'posteam': 'team',
        'offensive_epa': 'off_epa',
        'adj_offensive_epa': 'adj_off_epa',
        'offensive_success_rate': 'off_success',
        'adj_offensive_success_rate': 'adj_off_success',
        'yards_gained': 'total_yards',
        'turnover_ratio': 'turnover_rate',
        'adj_turnover_ratio': 'adj_turnover_rate'
    })
    
    def_stats = team_game_stats.rename(columns={
        'defteam': 'team',
        'offensive_epa': 'def_epa',
        'adj_offensive_epa': 'adj_def_epa',
        'offensive_success_rate': 'def_success',
        'adj_offensive_success_rate': 'adj_def_success'
    })
    
    full_team_stats = pd.merge(
        off_stats[['game_id', 'team', 'season', 'week', 'off_epa', 'adj_off_epa', 
                  'off_success', 'adj_off_success', 'total_yards', 'turnover_rate', 'adj_turnover_rate']],
        def_stats[['game_id', 'team', 'def_epa', 'adj_def_epa', 'def_success', 'adj_def_success']],
        on=['game_id', 'team']
    )
    
    full_team_stats = full_team_stats.sort_values(by=['team', 'season', 'week'])
    stats_to_avg = ['off_epa', 'adj_off_epa', 'off_success', 'adj_off_success', 
                   'def_epa', 'adj_def_epa', 'def_success', 'adj_def_success',
                   'total_yards', 'turnover_rate', 'adj_turnover_rate']
    
    game_details_for_win = pbp_df[['game_id', 'home_team', 'away_team', 'home_score', 'away_score']].drop_duplicates()
    game_details_for_win['winner'] = np.where(game_details_for_win['home_score'] > game_details_for_win['away_score'],
                                              game_details_for_win['home_team'],
                                              game_details_for_win['away_team'])
    
    full_team_stats = pd.merge(full_team_stats, game_details_for_win[['game_id', 'winner']], on='game_id', how='left')
    full_team_stats['win'] = (full_team_stats['team'] == full_team_stats['winner']).astype(int)
    full_team_stats = full_team_stats.drop(columns=['winner'])

    for stat in stats_to_avg:
        full_team_stats = full_team_stats.reset_index(drop=True)
        full_team_stats[f'ar1_{stat}'] = full_team_stats.groupby('team')[stat].transform(
            lambda x: x.shift(1) * AR1_RHO
        )

    # After calculating all AR(1) features, drop any rows with NaN values
    full_team_stats = full_team_stats.dropna(subset=[f'ar1_{stat}' for stat in stats_to_avg])
    full_team_stats = full_team_stats.reset_index(drop=True)

    if use_neural:
        result = train_neural_win_prob_model(full_team_stats)
        if result is not None:
            model, team_encoder, history = result
            return model, None, full_team_stats, 0.0
        else:
            return None, None, None, None
    else:
        game_details = pbp_df[['game_id', 'home_team', 'away_team', 'home_score', 'away_score']].drop_duplicates()
        game_details['home_win'] = (game_details['home_score'] > game_details['away_score']).astype(int)
        
        # Create home and away stats with proper column naming
        home_stats = full_team_stats.copy()
        home_stats = home_stats.rename(
            columns={col: f"{col}_home" for col in full_team_stats.columns if col != 'game_id'}
        )

        away_stats = full_team_stats.copy()
        away_stats = away_stats.rename(
            columns={col: f"{col}_away" for col in full_team_stats.columns if col != 'game_id'}
        )
        
        # Merge to create model dataframe
        try:
            model_df = pd.merge(
                game_details,
                home_stats,
                left_on=['game_id', 'home_team'],
                right_on=['game_id', 'team_home']
            )
            
            model_df = pd.merge(
                model_df,
                away_stats,
                left_on=['game_id', 'away_team'],
                right_on=['game_id', 'team_away']
            )
            
            model_df = model_df.dropna()
            
        except Exception as e:
            print(f"Error in merge operations: {e}")
            return None, None, None, None
        
        # Create differential features
        for stat in stats_to_avg:
            model_df[f'diff_ar1_{stat}'] = model_df[f'ar1_{stat}_home'] - model_df[f'ar1_{stat}_away']

        # Create interaction features
        interaction_terms = [
            ('ar1_off_epa_home', 'ar1_def_epa_away'),
            ('ar1_adj_off_epa_home', 'ar1_adj_def_epa_away'),
            ('ar1_off_success_home', 'ar1_def_success_away'),
            ('ar1_adj_off_success_home', 'ar1_adj_def_success_away')
        ]
        
        for term1, term2 in interaction_terms:
            if term1 in model_df.columns and term2 in model_df.columns:
                model_df[f'{term1}_x_{term2}'] = model_df[term1] * model_df[term2]
        
        # Define features as column names (not indices!)
        base_features = [f'diff_ar1_{stat}' for stat in stats_to_avg]
        interaction_features = [f'{term1}_x_{term2}' for term1, term2 in interaction_terms 
                               if f'{term1}_x_{term2}' in model_df.columns]
        features = base_features + interaction_features
        
        # Define target variable
        y = model_df['home_win']

        # FIXED: Simply select by column names - no complex logic needed
        try:
            X = model_df[features]  # This should work if features are column names
        except KeyError as e:
            print(f"Missing columns in model_df: {e}")
            print(f"Available columns: {model_df.columns.tolist()}")
            print(f"Requested features: {features}")
            return None, None, None, None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Split training data for early stopping
        X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train)

        if use_stacking and not use_neural:
            # Define base models
            base_models = {
                'logistic': LogisticRegression(max_iter=1000, random_state=42),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    use_label_encoder=False,
                    max_depth=4,
                    eta=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=1,
                    reg_lambda=1,
                    reg_alpha=0,
                    n_estimators=1000,
                    random_state=42
                )
            }
            # Convert numpy arrays to DataFrames for stacking compatibility
            X_train_df = pd.DataFrame(X_train_scaled, columns=features)
            X_test_df = pd.DataFrame(X_test_scaled, columns=features)

            # Train stacked model
            meta_model, base_models = train_stacked_model(
                X_train_df, y_train, base_models
            )
            
            # Make predictions
            y_pred_proba = predict_stacked_model(X_test_df, base_models, meta_model)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Calibrate if requested
            if calibrate:
                # Generate OOF predictions for calibration (training-level stacked features)
                oof_preds = generate_oof_predictions(X_train_df, y_train, base_models)
                # Build the base-model predictions for the validation/test set (same shape as oof_preds)
                val_base_preds = np.column_stack([
                    m.predict_proba(X_test_df)[:, 1] if hasattr(m, 'predict_proba') else m.predict(X_test_df)
                    for m in base_models.values()
                ])
                # Calibrate the meta-model using the base-model preds (NOT the original features)
                meta_model = calibrate_model_probas(meta_model, oof_preds, y_train, val_base_preds, y_test)
                y_pred_proba = predict_stacked_model(X_test_df, base_models, meta_model)
                y_pred = (y_pred_proba >= 0.5).astype(int)

            
            # Calculate metrics
            logloss = log_loss(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"✅ Stacked Win Probability Model trained.")
            print(f"   Log Loss: {logloss:.4f}")
            print(f"   Accuracy: {accuracy:.3f}")
            
            return (meta_model, base_models), scaler, full_team_stats, logloss
        
        else:
            # Use Gradient Boosting (XGBoost) for classification with logistic loss
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                max_depth=4,
                eta=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1,
                reg_lambda=1,
                reg_alpha=0,
                n_estimators=1000,
                random_state=42,
                early_stopping_rounds=50
            )
            print(f"DEBUG: Any NaN in win prob arrays? X_train_part: {np.isnan(X_train_part).any()}, X_val: {np.isnan(X_val).any()}")


            model.fit(
                X_train_part, y_train_part,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            if calibrate and not use_neural and hasattr(model, 'predict_proba'):
                model = calibrate_model_probas(model, X_train_scaled, y_train, X_val, y_val)
            
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
            
            logloss = log_loss(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"✅ Gradient Boosting Win Probability Model trained.")
            print(f"   Log Loss: {logloss:.4f}")
            print(f"   Accuracy: {accuracy:.3f}")
            
            return model, scaler, full_team_stats, logloss

def train_spread_model_huber(pbp_df):
    """Enhanced version with EWMA and opponent adjustment for spread prediction."""
    print("\n--- Training Enhanced Spread Prediction Model (Huber Loss) ---")
    
    team_game_stats = pbp_df.groupby(['game_id', 'posteam', 'defteam', 'season', 'week']).agg(
        offensive_epa=('epa', 'mean'),
        offensive_success_rate=('success', 'mean'),
        expected_points_added=('epa', 'mean'),
        third_down_conv=('third_down_converted', 'mean'),
        red_zone_efficiency=('touchdown', lambda x: x[pbp_df.loc[x.index, 'yardline_100'] <= 20].mean() if len(x[pbp_df.loc[x.index, 'yardline_100'] <= 20]) > 0 else 0),
        yards_gained=('yards_gained', 'sum'),
        turnovers=('interception', 'sum')
    ).reset_index()
    
    team_game_stats['turnover_ratio'] = team_game_stats['turnovers'] / (team_game_stats['yards_gained'] + 1)
    team_game_stats['opponent'] = team_game_stats['defteam']
    
    metrics_to_adjust = ['offensive_epa', 'offensive_success_rate', 'third_down_conv', 
                         'red_zone_efficiency', 'turnover_ratio', 'expected_points_added']
    for metric in metrics_to_adjust:
        team_game_stats, _, _ = adjust_for_opponent(team_game_stats, metric, 'posteam', 'opponent')
    
    off_stats = team_game_stats.rename(columns={
        'posteam': 'team',
        'offensive_epa': 'off_epa',
        'adj_offensive_epa': 'adj_off_epa',
        'offensive_success_rate': 'off_success',
        'adj_offensive_success_rate': 'adj_off_success',
        'third_down_conv': 'third_down_rate',
        'adj_third_down_conv': 'adj_third_down_rate',
        'red_zone_efficiency': 'rz_efficiency',
        'adj_red_zone_efficiency': 'adj_rz_efficiency',
        'turnover_ratio': 'turnover_rate',
        'adj_turnover_ratio': 'adj_turnover_rate'
    })
    
    def_stats = team_game_stats.rename(columns={
        'defteam': 'team',
        'offensive_epa': 'def_epa',
        'adj_offensive_epa': 'adj_def_epa',
        'offensive_success_rate': 'def_success',
        'adj_offensive_success_rate': 'adj_def_success'
    })
    
    full_team_stats = pd.merge(
        off_stats[['game_id', 'team', 'season', 'week', 'off_epa', 'adj_off_epa', 
                  'off_success', 'adj_off_success', 'third_down_rate', 'adj_third_down_rate',
                  'rz_efficiency', 'adj_rz_efficiency', 'turnover_rate', 'adj_turnover_rate']],
        def_stats[['game_id', 'team', 'def_epa', 'adj_def_epa', 'def_success', 'adj_def_success']],
        on=['game_id', 'team']
    )
    
    full_team_stats = full_team_stats.sort_values(by=['team', 'season', 'week'])
    stats_to_avg = ['off_epa', 'adj_off_epa', 'off_success', 'adj_off_success', 
                   'def_epa', 'adj_def_epa', 'def_success', 'adj_def_success',
                   'third_down_rate', 'adj_third_down_rate', 'rz_efficiency', 
                   'adj_rz_efficiency', 'turnover_rate', 'adj_turnover_rate']
    
    for stat in stats_to_avg:
            full_team_stats[f'ar1_{stat}'] = full_team_stats.groupby('team')[stat].transform(
                lambda x: x.shift(1) * AR1_RHO
            )
    
    game_details = pbp_df[['game_id', 'home_team', 'away_team', 'home_score', 'away_score']].drop_duplicates()
    
    home_stats = full_team_stats.rename(columns={col: f'{col}_home' for col in full_team_stats.columns if col != 'game_id'})
    away_stats = full_team_stats.rename(columns={col: f'{col}_away' for col in full_team_stats.columns if col != 'game_id'})
    
    model_df = pd.merge(
        game_details,
        home_stats,
        left_on=['game_id', 'home_team'],
        right_on=['game_id', 'team_home']
    )
    
    model_df = pd.merge(
        model_df,
        away_stats,
        left_on=['game_id', 'away_team'],
        right_on=['game_id', 'team_away']
    )
    
    model_df = model_df.dropna()
    
    for stat in stats_to_avg:
        model_df[f'diff_ar1_{stat}'] = model_df[f'ar1_{stat}_home'] - model_df[f'ar1_{stat}_away']
    
    model_df['spread_actual'] = model_df['home_score'] - model_df['away_score']
    
    features = [f'diff_ar1_{stat}' for stat in stats_to_avg]
    X = model_df[features]
    y = model_df['spread_actual']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)
    
    # A. Use Gradient Boosting (XGBoost) for regression
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='mae',
        max_depth=4,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
        n_estimators=1000,
        random_state=42,
        early_stopping_rounds=50  # <-- MOVED HERE
    )

    print(f"DEBUG: Any NaN in spread arrays? X_train_part: {np.isnan(X_train_part).any()}, X_val: {np.isnan(X_val).any()}")

    model.fit(
        X_train_part, y_train_part,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"✅ Gradient Boosting Spread Model trained.")
    print(f"   MAE: {mae:.2f} points")
    print(f"   MSE: {mse:.2f}")
    print(f"   R² Score: {r2:.3f}")
    
    return model, scaler, full_team_stats, mae # Return MAE as the primary loss metric

def train_enhanced_points_model(pbp_df):
    """Enhanced points model with Negative Binomial regression for overdispersion"""
    print("\n--- Training Enhanced Points Model (Negative Binomial) ---")
    
    team_game_stats = pbp_df.groupby(['game_id', 'posteam', 'defteam', 'season', 'week']).agg(
        offensive_epa=('epa', 'mean'),
        explosive_plays=('yards_gained', lambda x: (x > 20).sum()),
        expected_points_added=('epa', 'mean'),
        scoring_opps=('touchdown', 'sum'),
        field_goals=('field_goal_attempt', 'sum'),
        avg_field_position=('yardline_100', 'mean')
    ).reset_index()
    
    team_game_stats['opponent'] = team_game_stats['defteam']
    
    metrics_to_adjust = ['offensive_epa', 'explosive_plays', 'scoring_opps', 'field_goals', 'avg_field_position', 'expected_points_added']
    for metric in metrics_to_adjust:
        team_game_stats, _, _ = adjust_for_opponent(team_game_stats, metric, 'posteam', 'opponent')
    
    team_game_stats = team_game_stats.rename(columns={'posteam': 'team'})
    
    team_game_stats = team_game_stats.sort_values(by=['team', 'season', 'week'])
    stats_to_avg = ['offensive_epa', 'adj_offensive_epa', 'explosive_plays', 'adj_explosive_plays',
                   'scoring_opps', 'adj_scoring_opps', 'field_goals', 'adj_field_goals',
                   'avg_field_position', 'adj_avg_field_position']
    
    for stat in stats_to_avg:
        team_game_stats[f'ewma_{stat}'] = team_game_stats.groupby('team')[stat].transform(
            lambda x: calculate_ewma(x.shift(1))
        )
    
    game_details = pbp_df[['game_id', 'home_team', 'away_team', 'home_score', 'away_score']].drop_duplicates()
    game_details['total_points'] = game_details['home_score'] + game_details['away_score']
    
    home_stats = team_game_stats.rename(columns={col: f'{col}_home' for col in team_game_stats.columns if col != 'game_id'})
    away_stats = team_game_stats.rename(columns={col: f'{col}_away' for col in team_game_stats.columns if col != 'game_id'})
    
    model_df = pd.merge(
        game_details,
        home_stats,
        left_on=['game_id', 'home_team'],
        right_on=['game_id', 'team_home']
    )
    
    model_df = pd.merge(
        model_df,
        away_stats,
        left_on=['game_id', 'away_team'],
        right_on=['game_id', 'team_away']
    )
    
    model_df = model_df.dropna()
    
    for stat in stats_to_avg:
        model_df[f'combined_ewma_{stat}'] = model_df[f'ewma_{stat}_home'] + model_df[f'ewma_{stat}_away']
    
    features = [f'combined_ewma_{stat}' for stat in stats_to_avg]
    X = model_df[features]
    y = model_df['total_points']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Split training data for early stopping
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)
    
    # A. Use Gradient Boosting (XGBoost) for counts with Poisson loss
    model = xgb.XGBRegressor(
        objective='count:poisson',
        eval_metric='poisson-nloglik',
        max_depth=4,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        n_estimators=1000,
        random_state=42,
        early_stopping_rounds=50  # <-- MOVED HERE
    )
    print(f"DEBUG: Any NaN in total points arrays? X_train_part: {np.isnan(X_train_part).any()}, X_val: {np.isnan(X_val).any()}")

    model.fit(
        X_train_part, y_train_part,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    predictions = model.predict(X_test_scaled)
    # Ensure predictions are non-negative
    predictions[predictions < 0] = 0
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    poisson_loss = calculate_custom_losses(y_test, predictions, 'poisson')
    r2 = r2_score(y_test, predictions)
    
    print(f"✅ Gradient Boosting Points Model (Poisson) trained.")
    print(f"   MAE: {mae:.2f} points")
    print(f"   MSE: {mse:.2f}")
    print(f"   Poisson Loss: {poisson_loss:.4f}")
    print(f"   R² Score: {r2:.3f}")
    
    return model, scaler, team_game_stats, poisson_loss

def train_player_yardage_models_enhanced(weekly_df, team_stats, use_neural=False):
    """Enhanced player models with proper loss functions."""
    print("\n--- Training Enhanced Player Yardage Models ---")
    
    qb_data = weekly_df[weekly_df['position'] == 'QB'].copy()
    wr_data = weekly_df[weekly_df['position'] == 'WR'].copy()
    te_data = weekly_df[weekly_df['position'] == 'TE'].copy()
    rb_data = weekly_df[weekly_df['position'] == 'RB'].copy()
    
    team_def_stats = team_stats[['team', 'season', 'week', 'ar1_def_epa', 'ar1_def_success']].copy()
    team_def_stats = team_def_stats.rename(columns={
        'team': 'opponent_team',
        'ar1_def_epa': 'opp_def_epa',
        'ar1_def_success': 'opp_def_success'
    })
    
    def prepare_player_data(player_data, yardage_column, attempts_column, additional_features=[]):
        """Enhanced version with proper NaN handling"""
        player_data = player_data.sort_values(by=['player_id', 'season', 'week'])
        
        # Ensure this entire block is indented correctly
        print(f"DEBUG: before sequence building for {yardage_column}:")
        print("  available columns:", player_data.columns.tolist())
        # inspect value distributions for columns used in rolling
        for col in [yardage_column, attempts_column, "opp_def_epa","opp_def_success"]:
            if col in player_data.columns:
                print(f"   {col} -> min/max/mean/nulls: ",
                        player_data[col].min(), player_data[col].max(), player_data[col].mean(), player_data[col].isna().sum())
        # if you use groupby+rolling, ensure there are enough rows per group
        if "player_id" in player_data.columns:
            counts = player_data.groupby("player_id").size()
            print("  player row counts: min/median/max =", counts.min(), counts.median(), counts.max())

        # Calculate rolling statistics with proper NaN handling
        player_data['rolling_yds'] = player_data.groupby('player_id')[yardage_column].transform(
            lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
        player_data['rolling_yds'] = player_data['rolling_yds'].fillna(0) # ADD THIS LINE
        player_data['rolling_attempts'] = player_data.groupby('player_id')[attempts_column].transform(
            lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
        player_data['rolling_attempts'] = player_data['rolling_attempts'].fillna(0) # ADD THIS LINE
        
        # KEEP THESE CRITICAL LINES - filter out invalid data first
        # These lines should no longer have a red underline
        player_data = player_data[player_data[yardage_column].notna()]
        player_data = player_data[player_data['rolling_yds'].notna()]
        
        # Now calculate derived features only on valid data
        player_data['yards_per_attempt'] = np.where(
            player_data['rolling_attempts'] > 0,
            player_data['rolling_yds'] / player_data['rolling_attempts'],
            0  # Only use 0 when we have attempts data but it's zero
        )
        
        # Add trend feature
        player_data['recent_yds'] = player_data.groupby('player_id')[yardage_column].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        player_data['recent_yds'] = player_data['recent_yds'].fillna(0) # ADD THIS LINE
        player_data['trend'] = player_data['recent_yds'] - player_data['rolling_yds']
        
        # Fill any remaining NaN values in features (not the target!)
        feature_columns = ['rolling_yds', 'rolling_attempts', 'yards_per_attempt', 'recent_yds', 'trend']
        for col in feature_columns:
            if col in player_data.columns:
                player_data[col] = player_data[col].fillna(0)
        
        # Merge with opponent stats
        player_data = pd.merge(
            player_data,
            team_def_stats,
            on=['opponent_team', 'season', 'week'],
            how='left'
        )
        
        # Fill opponent stats with league averages if missing
        opponent_stats_cols = ['opp_def_epa', 'opp_def_success']
        for col in opponent_stats_cols:
            if col in player_data.columns:
                league_avg = player_data[col].mean()
                player_data[col] = player_data[col].fillna(league_avg)
        
        return player_data
    
    qb_data = prepare_player_data(qb_data, 'passing_yards', 'attempts', ['passing_tds', 'interceptions'])
    wr_data = prepare_player_data(wr_data, 'receiving_yards', 'targets', ['receptions', 'receiving_tds'])
    te_data = prepare_player_data(te_data, 'receiving_yards', 'targets', ['receptions'])
    rb_data = prepare_player_data(rb_data, 'rushing_yards', 'carries', ['rushing_tds', 'receptions'])

    if use_neural:
        neural_models = {}
        for position, data in [('QB', qb_data), ('WR', wr_data), ('TE', te_data), ('RB', rb_data)]:
            if data.empty:
                continue

            target_col = 'passing_yards' if position == 'QB' else 'receiving_yards' if position in ['WR', 'TE'] else 'rushing_yards'
            model, encoder, history = train_neural_player_model(data, position, target_col)
            neural_models[position] = (model, encoder, data)

        return neural_models
    else:
        player_models = {}

        print("Training QB model...")
        features_qb = ['rolling_yds', 'rolling_attempts', 'consistency_score', 'opp_def_epa', 'opp_def_success']
        if 'rolling_passing_tds' in qb_data.columns:
            features_qb.append('rolling_passing_tds')
        
        X_qb = qb_data[features_qb].fillna(0)
        y_qb = qb_data['passing_yards']
        
        X_train_qb, X_test_qb, y_train_qb, y_test_qb = train_test_split(X_qb, y_qb, test_size=0.2, random_state=42)
        
        scaler_qb = StandardScaler()
        X_train_qb_scaled = scaler_qb.fit_transform(X_train_qb)
        X_test_qb_scaled = scaler_qb.transform(X_test_qb)
        
        model_qb = HuberRegressor(epsilon=1.5, max_iter=1000)
        model_qb.fit(X_train_qb_scaled, y_train_qb)
        
        predictions_qb = model_qb.predict(X_test_qb_scaled)
        mae_qb = mean_absolute_error(y_test_qb, predictions_qb)
        huber_loss_qb = calculate_custom_losses(y_test_qb, predictions_qb, 'huber')
        r2_qb = r2_score(y_test_qb, predictions_qb)
        
        print(f"✅ QB Model: MAE={mae_qb:.2f}, Huber Loss={huber_loss_qb:.4f}, R²={r2_qb:.3f}")
        player_models['QB'] = (model_qb, scaler_qb, qb_data, features_qb)
        
        for position, data in [('WR', wr_data), ('TE', te_data)]:
            print(f"Training {position} model...")
            features = ['rolling_yds', 'rolling_attempts', 'consistency_score', 'opp_def_epa', 'opp_def_success']
            if 'rolling_receptions' in data.columns:
                features.append('rolling_receptions')
            
            X = data[features].fillna(0)
            y = data['receiving_yards']
            
            y = np.maximum(y, 0)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = PoissonRegressor(alpha=0.01, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            predictions = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, predictions)
            poisson_loss = calculate_custom_losses(y_test, predictions, 'poisson')
            r2 = r2_score(y_test, predictions)
            
            print(f"✅ {position} Model: MAE={mae:.2f}, Poisson Loss={poisson_loss:.4f}, R²={r2:.3f}")
            player_models[position] = (model, scaler, data, features)
        
        if not rb_data.empty:
            print("Training RB model...")
            features_rb = ['rolling_yds', 'rolling_attempts', 'consistency_score', 'opp_def_epa', 'opp_def_success']
            if 'rolling_rushing_tds' in rb_data.columns:
                features_rb.append('rolling_rushing_tds')
            
            X_rb = rb_data[features_rb].fillna(0)
            y_rb = rb_data['rushing_yards']
            
            X_train_rb, X_test_rb, y_train_rb, y_test_rb = train_test_split(X_rb, y_rb, test_size=0.2, random_state=42)
            
            scaler_rb = StandardScaler()
            X_train_rb_scaled = scaler_rb.fit_transform(X_train_rb)
            X_test_rb_scaled = scaler_rb.transform(X_test_rb)
            
            model_rb = HuberRegressor(epsilon=1.35, max_iter=1000)
            model_rb.fit(X_train_rb_scaled, y_train_rb)
            
            predictions_rb = model_rb.predict(X_test_rb_scaled)
            mae_rb = mean_absolute_error(y_test_rb, predictions_rb)
            huber_loss_rb = calculate_custom_losses(y_test_rb, predictions_rb, 'huber')
            r2_rb = r2_score(y_test_rb, predictions_rb)
            
            print(f"✅ RB Model: MAE={mae_rb:.2f}, Huber Loss={huber_loss_rb:.4f}, R²={r2_rb:.3f}")
            player_models['RB'] = (model_rb, scaler_rb, rb_data, features_rb)
        
        return player_models

def prepare_player_data_hierarchical(player_data, team_def_stats, yardage_column, attempts_column):
    """Prepare player data for hierarchical modeling, including opponent stats."""
    player_data = player_data.sort_values(by=['player_id', 'season', 'week'])
    
    # Filter out rows without target values
    if yardage_column in player_data.columns:
        player_data = player_data[player_data[yardage_column].notna()]
    
    # Calculate rolling statistics and fill any NaNs immediately
    if yardage_column in player_data.columns:
        player_data['rolling_yds'] = player_data.groupby('player_id')[yardage_column].transform(
            lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        ).fillna(0)
    else:
        player_data['rolling_yds'] = 0

    if attempts_column in player_data.columns:
        player_data['rolling_attempts'] = player_data.groupby('player_id')[attempts_column].transform(
            lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        ).fillna(0)
    else:
        player_data['rolling_attempts'] = 0
    
    # Merge with opponent stats
    player_data = pd.merge(
        player_data,
        team_def_stats,
        on=['opponent_team', 'season', 'week'],
        how='left'
    )
    
    # Fill missing opponent stats with the league average
    if 'opp_def_epa' in player_data.columns:
        player_data['opp_def_epa'] = player_data['opp_def_epa'].fillna(player_data['opp_def_epa'].mean())
    if 'opp_def_success' in player_data.columns:
        player_data['opp_def_success'] = player_data['opp_def_success'].fillna(player_data['opp_def_success'].mean())

    # Ensure feature columns exist and are filled to prevent KeyErrors
    for col in ['rolling_yds', 'rolling_attempts', 'opp_def_epa', 'opp_def_success']:
        if col not in player_data.columns:
            player_data[col] = 0
        else:
            player_data[col] = player_data[col].fillna(0)
            
    return player_data

def train_player_yardage_models_hierarchical(weekly_df, team_stats):
    """Train hierarchical models for player yardage prediction"""
    print("\n--- Training Hierarchical Player Yardage Models ---")
    
    # Prepare data for each position (same as before)
    qb_data = weekly_df[weekly_df['position'] == 'QB'].copy()
    wr_data = weekly_df[weekly_df['position'] == 'WR'].copy()
    te_data = weekly_df[weekly_df['position'] == 'TE'].copy()
    rb_data = weekly_df[weekly_df['position'] == 'RB'].copy()
    
    team_def_stats = team_stats[['team', 'season', 'week', 'ar1_def_epa', 'ar1_def_success']].copy()
    team_def_stats = team_def_stats.rename(columns={
        'team': 'opponent_team',
        'ar1_def_epa': 'opp_def_epa',
        'ar1_def_success': 'opp_def_success'
    })
    
    # Prepare data for each position using the new helper that handles merging
    qb_data = prepare_player_data_hierarchical(qb_data, team_def_stats, 'passing_yards', 'attempts')
    wr_data = prepare_player_data_hierarchical(wr_data, team_def_stats, 'receiving_yards', 'targets')
    te_data = prepare_player_data_hierarchical(te_data, team_def_stats, 'receiving_yards', 'targets')
    rb_data = prepare_player_data_hierarchical(rb_data, team_def_stats, 'rushing_yards', 'carries')
    
    player_models = {}
    
    # Define hierarchical model formulas for each position
    model_formulas = {
        'QB': 'passing_yards ~ rolling_yds + rolling_attempts + opp_def_epa + opp_def_success',
        'WR': 'receiving_yards ~ rolling_yds + rolling_attempts + opp_def_epa + opp_def_success',
        'TE': 'receiving_yards ~ rolling_yds + rolling_attempts + opp_def_epa + opp_def_success',
        'RB': 'rushing_yards ~ rolling_yds + rolling_attempts + opp_def_epa + opp_def_success'
    }
    
    # Train models for each position
    for position, data in [('QB', qb_data), ('WR', wr_data), ('TE', te_data), ('RB', rb_data)]:
        if data.empty:
            continue
            
        print(f"Training hierarchical model for {position}...")
        
        # Prepare features and target
        features = ['rolling_yds', 'rolling_attempts', 'opp_def_epa', 'opp_def_success']
        target_col = 'passing_yards' if position == 'QB' else 'receiving_yards' if position in ['WR', 'TE'] else 'rushing_yards'
        
        X = data[features].fillna(0)
        y = data[target_col]
        
        # Get player groups
        groups = data['player_id']
        
        # Split data
        X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
            X, y, groups, test_size=0.2, random_state=42
        )
        
        # Train hierarchical model
        formula = model_formulas[position]
        hierarchical_model = HierarchicalPlayerModel(formula=formula)
        print(f"DEBUG: NaN counts in X_train for {position} hierarchical model:\n{X_train.isna().sum()[X_train.isna().sum() > 0]}")
        hierarchical_model.fit(X_train, y_train, groups=groups_train)
        
        # Make predictions
        predictions = hierarchical_model.predict(X_test, groups=groups_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"✅ {position} Hierarchical Model: MAE={mae:.2f}, MSE={mse:.2f}, R²={r2:.3f}")
        
        # Store model in a compatible format for the prediction interface
        # Using None for the scaler as this model doesn't use one
        player_models[position] = (hierarchical_model, None, data, features)
    
    return player_models

def run_enhanced_prediction_interface(models_dict):
    """Enhanced interface for all prediction types with better error handling"""
    print("\n--- Enhanced NFL Prediction Tool ---")
    print("Available models:")
    print("1. Binary Game Outcome (Win/Loss)")
    print("2. Point Spread (Huber Loss)")
    print("3. Total Points (Poisson/NB)")
    print("4. Player Yardage")
    print("5. Neural Network Predictions")
    print("6. Exit")
    
    choice = input("\nSelect prediction type (1-5): ")
        
    if choice == '1' and 'binary' in models_dict:
        model_info, scaler, team_stats, _ = models_dict['binary']
        
        # Check if it's a stacked model (tuple with meta_model and base_models)
        if isinstance(model_info, tuple) and len(model_info) == 2:
            meta_model, base_models = model_info
            is_stacked = True
        else:
            model = model_info
            is_stacked = False
        
        home_team = input("Enter Home Team abbreviation: ").upper().strip()
        away_team = input("Enter Away Team abbreviation: ").upper().strip()
        
        try:
            available_teams = team_stats['team'].unique()
            if home_team not in available_teams:
                print(f"❌ Team '{home_team}' not found. Available teams: {sorted(available_teams)}")
                return True
            if away_team not in available_teams:
                print(f"❌ Team '{away_team}' not found. Available teams: {sorted(available_teams)}")
                return True
            
            home_stats = team_stats[team_stats['team'] == home_team].iloc[-1]
            away_stats = team_stats[team_stats['team'] == away_team].iloc[-1]
            
            # Base features (differential stats)
            base_features = []
            stats_to_check = ['off_epa', 'adj_off_epa', 'off_success', 'adj_off_success', 
                            'def_epa', 'adj_def_epa', 'def_success', 'adj_def_success',
                            'total_yards', 'turnover_rate', 'adj_turnover_rate']
            
            for stat in stats_to_check:
                ar1_stat = f'ar1_{stat}'
                if ar1_stat in home_stats.index and ar1_stat in away_stats.index:
                    diff = home_stats[ar1_stat] - away_stats[ar1_stat]
                    base_features.append(diff)
                else:
                    base_features.append(0)
            
            # Interaction features (the model expects these with _home and _away suffixes)
            interaction_terms = [
                ('off_epa', 'def_epa'),
                ('adj_off_epa', 'adj_def_epa'),
                ('off_success', 'def_success'),
                ('adj_off_success', 'adj_def_success')
            ]
            
            interaction_features = []
            for term1, term2 in interaction_terms:
                home_term = f'ar1_{term1}'
                away_term = f'ar1_{term2}'
                if home_term in home_stats.index and away_term in away_stats.index:
                    interaction = home_stats[home_term] * away_stats[away_term]
                    interaction_features.append(interaction)
                else:
                    interaction_features.append(0)
            
            # Combine all features (base + interaction)
            features = base_features + interaction_features
            features = np.array([features])
            
            # Scale the features before predicting
            features_scaled = scaler.transform(features)
            
            # Handle both stacked and regular models
            if is_stacked:
                # Convert to DataFrame for stacking compatibility
                # Use the exact same feature names as during training
                feature_names = [f'diff_ar1_{stat}' for stat in stats_to_check]
                # Match the training format for interaction features
                interaction_names = [
                    'ar1_off_epa_home_x_ar1_def_epa_away',
                    'ar1_adj_off_epa_home_x_ar1_adj_def_epa_away', 
                    'ar1_off_success_home_x_ar1_def_success_away',
                    'ar1_adj_off_success_home_x_ar1_adj_def_success_away'
                ]
                all_feature_names = feature_names + interaction_names
                features_df = pd.DataFrame(features_scaled, columns=all_feature_names)
                
                # Use stacked model prediction
                win_prob = predict_stacked_model(features_df, base_models, meta_model)
                
                # Ensure win_prob is a scalar
                if isinstance(win_prob, np.ndarray):
                    win_prob = win_prob[0] if win_prob.size == 1 else win_prob.item()
                
                model_type = "Stacked Ensemble"
            else:
                # Use regular model prediction (preserve existing behavior)
                win_prob = model.predict_proba(features_scaled)[0, 1]
                model_type = "Single Model"
            
            # Ensure win_prob is a float for formatting
            win_prob = float(win_prob)
            
            print(f"\n🎯 {model_type} Prediction:")
            print(f"   {home_team} win probability: {win_prob:.1%}")
            print(f"   {away_team} win probability: {(1-win_prob):.1%}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Available teams:", sorted(team_stats['team'].unique()))
    
    elif choice == '2' and 'spread' in models_dict:
        model, scaler, team_stats, _ = models_dict['spread']
        home_team = input("Enter Home Team abbreviation: ").upper().strip()
        away_team = input("Enter Away Team abbreviation: ").upper().strip()
        
        try:
            available_teams = team_stats['team'].unique()
            if home_team not in available_teams:
                print(f"❌ Team '{home_team}' not found. Available teams: {sorted(available_teams)}")
                return True
            if away_team not in available_teams:
                print(f"❌ Team '{away_team}' not found. Available teams: {sorted(available_teams)}")
                return True
            
            home_stats = team_stats[team_stats['team'] == home_team].iloc[-1]
            away_stats = team_stats[team_stats['team'] == away_team].iloc[-1]
            
            features = []
            stats_to_check = ['off_epa', 'adj_off_epa', 'off_success', 'adj_off_success', 
                            'def_epa', 'adj_def_epa', 'def_success', 'adj_def_success',
                            'third_down_rate', 'adj_third_down_rate', 'rz_efficiency', 
                            'adj_rz_efficiency', 'turnover_rate', 'adj_turnover_rate']
            
            for stat in stats_to_check:
                ar1_stat = f'ar1_{stat}'
                if ar1_stat in home_stats.index and ar1_stat in away_stats.index:
                    diff = home_stats[ar1_stat] - away_stats[ar1_stat]
                    features.append(diff)
                else:
                    features.append(0)
            
            features = np.array([features])
            features_scaled = scaler.transform(features)
            
            spread = model.predict(features_scaled)[0]
            if spread > 0:
                print(f"\n📊 Predicted: {home_team} by {spread:.1f} points")
            else:
                print(f"\n📊 Predicted: {away_team} by {-spread:.1f} points")
                
        except Exception as e:
            print(f"Error: {e}")
            print("Available teams:", sorted(team_stats['team'].unique()))
    
    elif choice == '3' and 'total' in models_dict:
        model, scaler, team_stats, _ = models_dict['total']
        home_team = input("Enter Home Team abbreviation: ").upper().strip()
        away_team = input("Enter Away Team abbreviation: ").upper().strip()
        
        try:
            available_teams = team_stats['team'].unique()
            if home_team not in available_teams:
                print(f"❌ Team '{home_team}' not found. Available teams: {sorted(available_teams)}")
                return True
            if away_team not in available_teams:
                print(f"❌ Team '{away_team}' not found. Available teams: {sorted(available_teams)}")
                return True
            
            home_stats = team_stats[team_stats['team'] == home_team].iloc[-1]
            away_stats = team_stats[team_stats['team'] == away_team].iloc[-1]
            
            features = []
            stats_to_check = ['offensive_epa', 'adj_offensive_epa', 'explosive_plays', 'adj_explosive_plays',
                            'scoring_opps', 'adj_scoring_opps', 'field_goals', 'adj_field_goals',
                            'avg_field_position', 'adj_avg_field_position']
            
            for stat in stats_to_check:
                ar1_stat = f'ar1_{stat}'
                if ar1_stat in home_stats.index and ar1_stat in away_stats.index:
                    combined = home_stats[ar1_stat] + away_stats[ar1_stat]
                    features.append(combined)
                else:
                    features.append(0)
            
            features = np.array([features])
            features_scaled = scaler.transform(features)
            
            if hasattr(model, 'predict'):
                total_points = model.predict(features_scaled)[0]
            else:
                features_with_const = sm.add_constant(features_scaled)
                total_points = model.predict(features_with_const)[0]
            
            print(f"\n🎯 Predicted Total Points: {total_points:.1f}")
            print(f"   Over/Under suggestion: {total_points:.1f}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Available teams:", sorted(team_stats['team'].unique()))
    
    elif choice == '4' and 'players' in models_dict:
        player_models = models_dict['players']
        position = input("Enter position (QB, WR, TE, RB): ").upper()
        
        if position not in player_models:
            print(f"No model available for position: {position}")
            return True

        player_name = input("Enter player name: ")
        opponent = input("Enter opponent team abbreviation: ").upper()
        
        # This correctly unpacks the model information from the tuple
        model, _, player_data, feature_names = player_models[position]
        
        try:
            # Find player data
            player_history = player_data[
                player_data['player_display_name'].str.contains(player_name, case=False, na=False)
            ]
            
            if player_history.empty:
                print(f"Could not find data for player: {player_name}")
                return True
            
            # Get latest data for the player
            latest_data = player_history.iloc[-1]
            player_id = latest_data['player_id']
            
            # Get opponent defensive stats
            opp_data = player_data[player_data['opponent_team'] == opponent]
            if opp_data.empty:
                print(f"Warning: Limited data for opponent {opponent}, using league averages")
                opp_def_epa = player_data['opp_def_epa'].mean()
                opp_def_success = player_data['opp_def_success'].mean()
            else:
                opp_def_epa = opp_data['opp_def_epa'].mean()
                opp_def_success = opp_data['opp_def_success'].mean()
            
            # Prepare features for prediction
            features = []
            for feat in feature_names:
                if feat == 'opp_def_epa':
                    features.append(opp_def_epa)
                elif feat == 'opp_def_success':
                    features.append(opp_def_success)
                elif feat in latest_data.index:
                    features.append(latest_data[feat])
                else:
                    features.append(0)

            # Add intercept term (constant 1) as the first feature
            features_with_const = [1] + features
            X_pred = pd.DataFrame([features_with_const], columns=['const'] + feature_names)
            
            # Make prediction using hierarchical model
            prediction = model.predict(X_pred, groups=[player_id])[0]
            
            # Display results
            print(f"\n--- Prediction for {player_name} ({position}) vs {opponent} ---")
            if position == 'QB':
                print(f"🏈 Predicted passing yards: {prediction:.1f}")
            elif position == 'RB':
                print(f"🏈 Predicted rushing yards: {prediction:.1f}")
            else:
                print(f"🏈 Predicted receiving yards: {prediction:.1f}")
            
            # Show recent performance
            recent_games = min(3, len(player_history))
            print(f"\nRecent performance (last {recent_games} games):")
            for i in range(1, recent_games + 1):
                game = player_history.iloc[-i]
                yard_type = "yards"
                yards = 0
                if position == 'QB' and 'passing_yards' in game:
                    yards = game['passing_yards']
                    yard_type = "passing"
                elif position == 'RB' and 'rushing_yards' in game:
                    yards = game['rushing_yards']
                    yard_type = "rushing"
                elif position in ['WR', 'TE'] and 'receiving_yards' in game:
                    yards = game['receiving_yards']
                    yard_type = "receiving"
                print(f"  Week {int(game['week'])}: {yards:.0f} {yard_type} yards vs {game['opponent_team']}")
                
        except Exception as e:
            print(f"Error making prediction: {e}")
    
    elif choice == '5' and 'neural' in models_dict:
        print("\nNeural Network Prediction Options:")
        print("a. Team Win Probability")
        print("b. Player Yardage Prediction")
        neural_choice = input("Select neural network option (a/b): ").lower()
        
        if neural_choice == 'a' and 'neural_binary' in models_dict['neural']:
            model, team_encoder, team_stats = models_dict['neural']['neural_binary']
            home_team = input("Enter Home Team abbreviation: ").upper().strip()
            away_team = input("Enter Away Team abbreviation: ").upper().strip()
            
            try:
                # Check if teams exist
                available_teams = team_stats['team'].unique()
                if home_team not in available_teams:
                    print(f"❌ Team '{home_team}' not found. Available teams: {sorted(available_teams)}")
                    return True
                if away_team not in available_teams:
                    print(f"❌ Team '{away_team}' not found. Available teams: {sorted(available_teams)}")
                    return True
                
                # Get team sequences
                home_sequences, _, _ = prepare_team_sequences(
                    team_stats[team_stats['team'] == home_team], 
                    sequence_length=5
                )
                away_sequences, _, _ = prepare_team_sequences(
                    team_stats[team_stats['team'] == away_team], 
                    sequence_length=5
                )
                
                if len(home_sequences) == 0 or len(away_sequences) == 0:
                    print("❌ Not enough historical data for these teams")
                    return True
                
                # Use the most recent sequence
                home_sequence = home_sequences[-1:]
                away_sequence = away_sequences[-1:]
                
                # Encode teams
                home_encoded = team_encoder.transform([home_team])[0]
                away_encoded = team_encoder.transform([away_team])[0]
                
                # Make predictions
                home_win_prob = model.predict([np.array([home_encoded]), home_sequence])[0][0]
                away_win_prob = model.predict([np.array([away_encoded]), away_sequence])[0][0]
                
                # Normalize probabilities (they might not sum to 1)
                total = home_win_prob + away_win_prob
                home_win_prob /= total
                away_win_prob /= total
                
                print(f"\n🧠 Neural Network Prediction:")
                print(f"   {home_team} win probability: {home_win_prob:.1%}")
                print(f"   {away_team} win probability: {away_win_prob:.1%}")
                
            except Exception as e:
                print(f"Error making neural network prediction: {e}")
                
        elif neural_choice == 'b' and 'neural_players' in models_dict['neural']:
            player_models = models_dict['neural']['neural_players']
            position = input("Enter position (QB, WR, TE, RB): ").upper()
            
            if position not in player_models:
                print(f"No neural model available for position: {position}")
                return True

            model, player_encoder, player_data, target_scaler = player_models[position]  # Now 4 values
            player_name = input("Enter player name: ")
            opponent = input("Enter opponent team abbreviation: ").upper()
            
            try:
                # Find player data
                player_history = player_data[
                    player_data['player_display_name'].str.contains(player_name, case=False, na=False)
                ]
                
                if player_history.empty:
                    print(f"Could not find data for player: {player_name}")
                    return True
                
                # Get player ID
                player_id = player_history.iloc[-1]['player_id']
                
                # Prepare player sequence
                player_sequences, _, _ = prepare_player_sequences(
                    player_history, 
                    sequence_length=4,
                    target_col='passing_yards' if position == 'QB' else 
                            'receiving_yards' if position in ['WR', 'TE'] else 
                            'rushing_yards'
                )
                
                if len(player_sequences) == 0:
                    print("❌ Not enough historical data for this player")
                    return True
                
                # Use the most recent sequence
                player_sequence = player_sequences[-1:]
                
                # Encode player
                player_encoded = player_encoder.transform([player_id])[0]
                
                # Make prediction and scale back to original units
                prediction_scaled = model.predict([np.array([player_encoded]), player_sequence])[0][0]
                prediction = target_scaler.inverse_transform([[prediction_scaled]])[0][0]
                
                # Display results
                print(f"\n--- Neural Network Prediction for {player_name} ({position}) ---")
                if position == 'QB':
                    print(f"🏈 Predicted passing yards: {prediction:.1f}")
                elif position == 'RB':
                    print(f"🏈 Predicted rushing yards: {prediction:.1f}")
                else:
                    print(f"🏈 Predicted receiving yards: {prediction:.1f}")
                
                # Show recent performance
                recent_games = min(3, len(player_history))
                print(f"\nRecent performance (last {recent_games} games):")
                for i in range(1, recent_games + 1):
                    game = player_history.iloc[-i]
                    yard_type = "yards"
                    yards = 0
                    if position == 'QB' and 'passing_yards' in game:
                        yards = game['passing_yards']
                        yard_type = "passing"
                    elif position == 'RB' and 'rushing_yards' in game:
                        yards = game['rushing_yards']
                        yard_type = "rushing"
                    elif position in ['WR', 'TE'] and 'receiving_yards' in game:
                        yards = game['receiving_yards']
                        yard_type = "receiving"
                    print(f"  Week {int(game['week'])}: {yards:.0f} {yard_type} yards vs {game['opponent_team']}")
                    
            except Exception as e:
                print(f"Error making neural network prediction: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Invalid neural network option or model not available.")

    elif choice == '6':
        print("Goodbye!")
        return False
    else:
        print("Invalid choice or model not available.")
    
    return True

def display_model_performance_summary(models_dict):
    """Display a summary of all trained models' performance."""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    if 'binary' in models_dict:
        _, _, _, logloss = models_dict['binary']
        print(f"\n📊 Binary Classification (Home Win/Loss)")
        print(f"   Loss Function: Logistic (Cross-Entropy)")
        print(f"   Log Loss: {logloss:.4f}")
    
    if 'spread' in models_dict:
        _, _, _, huber_loss = models_dict['spread']
        print(f"\n📊 Point Spread Prediction")
        print(f"   Loss Function: Huber (Robust)")
        print(f"   Huber Loss: {huber_loss:.4f}")
    
    if 'total' in models_dict:
        _, _, _, poisson_loss = models_dict['total']
        print(f"\n📊 Total Points Prediction")
        print(f"   Loss Function: Poisson")
        print(f"   Poisson Loss: {poisson_loss:.4f}")
    
    print("\n" + "="*60)

def add_rolling_features(df, window=4, on='season_week_order'):
    # example rolling_yds (4-game rolling sum of yards)
    # ensure we have season & week ordering columns; if not, create one
    if 'season' in df.columns and 'week' in df.columns:
        df = df.sort_values(['player_id','season','week']).copy()
    else:
        df = df.sort_values(['player_id']).copy()

    # choose a yards column that exists (try rushing_yards -> passing_yards -> receiving_yards)
    if 'rushing_yards' in df.columns:
        yards_col = 'rushing_yards'
    elif 'passing_yards' in df.columns:
        yards_col = 'passing_yards'
    elif 'receiving_yards' in df.columns:
        yards_col = 'receiving_yards'
    else:
        # If no yards column, create a dummy one to avoid crashing
        print("WARN: No base yards column found to compute rolling_yds. Defaulting to 0.")
        yards_col = 'rushing_yards' # Set to a name
        if yards_col not in df.columns:
            df[yards_col] = 0

    # rolling attempts: try attempts or carries + targets
    if 'attempts' in df.columns:
        att_col = 'attempts'
    else:
        # fallback: define attempts as carries + targets if present
        att_col = None
        if 'carries' in df.columns and 'targets' in df.columns:
            df['attempts'] = df['carries'].fillna(0) + df['targets'].fillna(0)
            att_col = 'attempts'
    
    if att_col is None:
        df['rolling_attempts'] = 0
    else:
        # Use .shift(1) to prevent data leakage from the current week
        df['rolling_attempts'] = df.groupby('player_id')[att_col].shift(1).rolling(window=window, min_periods=1).sum().reset_index(level=0, drop=True)

    # Use .shift(1) here as well
    df['rolling_yds'] = df.groupby('player_id')[yards_col].shift(1).rolling(window=window, min_periods=1).sum().reset_index(level=0, drop=True)
    
    # Fill NaNs created by rolling/shifting
    df['rolling_yds'].fillna(0, inplace=True)
    df['rolling_attempts'].fillna(0, inplace=True)
    
    return df

def main():
    """Main function to run the enhanced NFL prediction system."""
    print("="*60)
    print("ENHANCED NFL PREDICTION SYSTEM")
    print("With Neural Networks, Embeddings, and Sequence Models")
    print("="*60)

    def show_col_diagnostics(df, name="df"):
        print(f"--- Columns in {name}: {len(df.columns)} ---")
        print(", ".join(list(df.columns)[:80]) + (" ..." if len(df.columns)>80 else ""))
        required = ['opp_def_epa','opp_def_success','rolling_yds','rolling_attempts']
        for c in required:
            present = c in df.columns
            # Ensure the required columns exist before trying to access them for nan count
            if present:
                print(f"{c:20} present: {present}  |  nan_count: {df[c].isna().sum()}")
            else:
                 print(f"{c:20} present: {present}  |  nan_count: N/A")
        # Sample only from existing columns to prevent KeyErrors
        if not df.empty:
            sample_cols = min(10, len(df.columns))
            print(df.head(2).T.sample(sample_cols))  # sample a few fields for quick spot-check


    print("\n📥 Loading NFL data...")
    pbp_historical_df, team_def_stats_from_pbp = get_pbp_data(HISTORICAL_YEARS)
    weekly_data = get_weekly_data(HISTORICAL_YEARS, team_def_stats_from_pbp)
    print(f"Initial ID of weekly_data: {id(weekly_data)}")
    # Normalize pbp_historical_df column names
# --- MINIMAL PATCH FOR DATA PREPARATION ---
    print("\n🔧 Applying minimal patch for data normalization and feature creation...")

    # NORMALIZE COLS
    for df_name, df in [('pbp_historical_df', pbp_historical_df), ('weekly_data', weekly_data)]:
        if not df.empty:
            df.columns = df.columns.map(lambda s: s.strip() if isinstance(s, str) else s)
            df.columns = df.columns.str.replace(" ", "_").str.lower()
    
    # BUILD team-week defensive proxy FROM PBP if not available
    if 'opp_def_epa' not in weekly_data.columns or 'opp_def_success' not in weekly_data.columns:
        print("   'opp_def_epa' or 'opp_def_success' not found. Building from PBP data...")
        def_by_team_week = (pbp_historical_df.loc[~pbp_historical_df['defteam'].isna(), ['season','week','defteam','epa']]
            .groupby(['season','week','defteam'], as_index=False)
            .agg(opp_def_epa=('epa','mean'),
                 opp_def_success=('epa', lambda s: (s>0).mean()))
            .rename(columns={'defteam':'opponent_team'}))
        weekly_data = weekly_data.merge(def_by_team_week, on=['season','week','opponent_team'], how='left')
        
        # Fill NaNs after merge
        weekly_data['opp_def_epa'] = pd.to_numeric(weekly_data['opp_def_epa'], errors='coerce').fillna(weekly_data['opp_def_epa'].mean())
        weekly_data['opp_def_success'] = pd.to_numeric(weekly_data['opp_def_success'], errors='coerce').fillna(weekly_data['opp_def_success'].mean())

    # BUILD rolling features if missing
    if 'rolling_yds' not in weekly_data.columns or 'rolling_attempts' not in weekly_data.columns:
        print("   'rolling_yds' or 'rolling_attempts' not found. Building now...")
        weekly_data = add_rolling_features(weekly_data, window=4)
        
    print("✅ Patch applied successfully.")
    print(f"ID of weekly_data after patch: {id(weekly_data)}")
    

    weekly_data = add_rolling_features(weekly_data, window=4)

    # 3) Final safety: ensure these columns exist and contain finite numbers
    for c in ["opp_def_epa", "opp_def_success", "rolling_yds", "rolling_attempts"]:
        if c not in weekly_data.columns:
            weekly_data[c] = 0.0
        weekly_data[c] = weekly_data[c].replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    models_dict = {}
    
    # Train traditional models
    print("\n🔧 Training traditional models...")
    
    # 1. Enhanced win probability model
    try:
        win_model, win_scaler, win_stats, win_loss = train_enhanced_win_prob_model(
            pbp_historical_df, 
            use_neural=False, 
            use_stacking=True,  # Enable stacking
            calibrate=True      # Enable calibration
        )
        if win_model is not None:
            models_dict['binary'] = (win_model, win_scaler, win_stats, win_loss)
        else:
            print("❌ Win probability model training failed")
    except Exception as e:
        print(f"❌ Failed to train enhanced win probability model: {e}")
    
    # Only proceed with neural models if binary model was successfully trained
    if 'binary' in models_dict:
        team_stats_for_neural = models_dict['binary'][2]  # Reuse stats from win prob model
        
        print("\n🔧 Preparing player data for neural network models...")
        # ... rest of the neural model preparation code ...
    else:
        print("❌ Skipping neural model training due to binary model failure")
    
    # 2. Enhanced spread model
    try:
        spread_model, spread_scaler, spread_stats, spread_loss = train_spread_model_huber(pbp_historical_df)
        models_dict['spread'] = (spread_model, spread_scaler, spread_stats, spread_loss)
    except Exception as e:
        print(f"❌ Failed to train enhanced spread model: {e}")
    
    # 3. Enhanced points model
    try:
        points_model, points_scaler, points_stats, points_loss = train_enhanced_points_model(pbp_historical_df)
        models_dict['total'] = (points_model, points_scaler, points_stats, points_loss)
    except Exception as e:
        print(f"❌ Failed to train enhanced points model: {e}")
    
    ROLL_WINDOW = 4  # adjust if you use a different window
    # find weekly df again (same logic as above)
    weekly_name = next((n for n in ['weekly_data'] if n in locals()), None)
    
    if weekly_name is None:
        print("WARN: cannot compute rolling features because weekly dataframe not found.")
    else:
        df_weekly = locals()[weekly_name].copy()
        df_weekly = df_weekly.sort_values(['player_id','season','week'])

        # rolling for receiving (only add if base columns exist)
        if 'receiving_yards' in df_weekly.columns:
            # Using .transform for potential performance benefits and correct alignment
            df_weekly['rolling_yds'] = (
                df_weekly.groupby('player_id')['receiving_yards']
                         .transform(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
            )

        # rolling for targets/attempts
        if 'targets' in df_weekly.columns:
            df_weekly['rolling_attempts'] = (
                df_weekly.groupby('player_id')['targets']
                         .transform(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
            )

        # optional: add pass/rush rolling if present
        if 'passing_yards' in df_weekly.columns:
            df_weekly['rolling_pass_yds'] = (
                df_weekly.groupby('player_id')['passing_yards']
                         .transform(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
            )
        if 'attempts' in df_weekly.columns:
            df_weekly['rolling_pass_attempts'] = (
                df_weekly.groupby('player_id')['attempts']
                         .transform(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
            )
        
        # Re-assign the modified dataframe back to its original variable name
        if weekly_name in locals():
            locals()[weekly_name] = df_weekly
        # Note: In Python, direct modification of locals() is not guaranteed to work.
        # It's safer to re-assign directly.
        weekly_data = df_weekly

    # --- SECOND DIAGNOSTIC CHECK ---
    print("\n\n*** DIAGNOSTIC CHECK 2: BEFORE PLAYER MODELS ***")
    show_col_diagnostics(weekly_data, name="weekly_data")
    print("*** END DIAGNOSTIC CHECK 2 ***\n\n")

    # 4. Player models
    # In the main function, replace the player model training with:
    try:
        team_stats_for_players = None
        if 'spread' in models_dict:
            team_stats_for_players = models_dict['spread'][2]
        elif 'binary' in models_dict:
            team_stats_for_players = models_dict['binary'][2]
        
        if team_stats_for_players is not None:
            # Use hierarchical models instead of enhanced models
            player_models = train_player_yardage_models_hierarchical(weekly_data, team_stats_for_players)
            models_dict['players'] = player_models
    except Exception as e:
        print(f"❌ Failed to train player models: {e}")

    # Initialize a dictionary to hold all neural models.
    neural_models = {}

    # Neural models depend on the data processed by the binary model.
    # Check if it was trained successfully before proceeding.
    if 'binary' in models_dict:
        print("\n🧠 Training neural network models with embeddings and sequences...")
        team_stats_for_neural = models_dict['binary'][2]

        # 1. Train Neural Win Probability Model
        try:
            win_model, win_encoder, win_history = train_neural_win_prob_model(team_stats_for_neural)
            # Store the necessary components for prediction
            neural_models['neural_binary'] = (win_model, win_encoder, team_stats_for_neural)
        except Exception as e:
            print(f"❌ Failed to train neural win probability model: {e}")

        # --- Sanity-check weekly_data BEFORE training neural player models ---
        required = ['opp_def_epa','opp_def_success','rolling_yds','rolling_attempts']
        print("\n\n🔬 Sanity-checking required features in 'weekly_data' before neural player model training...")
        print("   Data types of required features:\n", weekly_data[required].dtypes)

        missing = [c for c in required if c not in weekly_data.columns]

        if missing:
            raise RuntimeError(f"❌ Missing required player features: {missing}. Did you run add_rolling_features / merge team defense?")

        # Also check for NaNs
        nan_counts = weekly_data[required].isna().sum().to_dict()
        print("   NaN counts on required features:", nan_counts)

        if any(v > 0 for v in nan_counts.values()):
            print("   Rows with NaN values found. Filling with the median of each column.")
            # Optionally fill NaNs with the column median
            weekly_data[required] = weekly_data[required].fillna(weekly_data[required].median())
            print("   NaNs filled. New counts:", weekly_data[required].isna().sum().to_dict())

        print("✅ Sanity check passed.\n\n")
        
        # 2. Prepare Data and Train Neural Player Models
        try:
            print("\n🔧 Preparing player data for neural network models...")            
            # First, check if the required columns already exist in weekly_data
            # since they were added during the patch phase
            player_data_dict = {
                'QB': weekly_data[weekly_data['position'] == 'QB'].copy(),
                'WR': weekly_data[weekly_data['position'] == 'WR'].copy(),
                'TE': weekly_data[weekly_data['position'] == 'TE'].copy(),
                'RB': weekly_data[weekly_data['position'] == 'RB'].copy()
            }

            # The opp_def_epa and opp_def_success should already be in weekly_data from the patch
            # But we'll prepare additional features if needed
            def prepare_data(p_data, y_col, a_col):
                # Check if these columns already exist (they should from the patch)
                if 'opp_def_epa' not in p_data.columns or 'opp_def_success' not in p_data.columns:
                    # If somehow they're missing, try to get them from team_stats
                    team_def_stats_for_neural = team_stats_for_neural[['team', 'season', 'week', 'ar1_def_epa', 'ar1_def_success']].copy()
                    team_def_stats_for_neural = team_def_stats_for_neural.rename(columns={
                        'team': 'opponent_team',
                        'ar1_def_epa': 'opp_def_epa_new',
                        'ar1_def_success': 'opp_def_success_new'
                    })
                    p_data = pd.merge(p_data, team_def_stats_for_neural, on=['opponent_team', 'season', 'week'], how='left')
                    # Use the new columns if the original ones don't exist
                    if 'opp_def_epa' not in p_data.columns:
                        p_data['opp_def_epa'] = p_data['opp_def_epa_new']
                    if 'opp_def_success' not in p_data.columns:
                        p_data['opp_def_success'] = p_data['opp_def_success_new']
                    # Drop temporary columns
                    p_data = p_data.drop(columns=['opp_def_epa_new', 'opp_def_success_new'], errors='ignore')
                
                # Check if rolling features already exist (they should from the patch)
                if 'rolling_yds' not in p_data.columns:
                    p_data = p_data.sort_values(by=['player_id', 'season', 'week'])
                    p_data['rolling_yds'] = p_data.groupby('player_id')[y_col].transform(
                        lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).mean()
                    )
                if 'rolling_attempts' not in p_data.columns:
                    p_data['rolling_attempts'] = p_data.groupby('player_id')[a_col].transform(
                        lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).mean()
                    )
                
                # Fill NaNs in all required columns
                for col in ['rolling_yds', 'rolling_attempts', 'opp_def_epa', 'opp_def_success']:
                    if col in p_data.columns:
                        if col in ['opp_def_epa', 'opp_def_success']:
                            # Fill with mean for opponent stats
                            p_data[col] = p_data[col].fillna(p_data[col].mean() if not p_data[col].isna().all() else 0)
                        else:
                            # Fill with 0 for rolling stats
                            p_data[col] = p_data[col].fillna(0)
                
                return p_data

            player_data_dict['QB'] = prepare_data(player_data_dict['QB'], 'passing_yards', 'attempts')
            player_data_dict['WR'] = prepare_data(player_data_dict['WR'], 'receiving_yards', 'targets')
            player_data_dict['TE'] = prepare_data(player_data_dict['TE'], 'receiving_yards', 'targets')
            player_data_dict['RB'] = prepare_data(player_data_dict['RB'], 'rushing_yards', 'carries')

            neural_player_models = {}
            for position in ['QB', 'WR', 'TE', 'RB']:
                if not player_data_dict[position].empty:
                    target_col = 'passing_yards' if position == 'QB' else 'rushing_yards' if position == 'RB' else 'receiving_yards'
                    # The function now returns 4 items, including the scaler
                    model, encoder, player_data, target_scaler = train_neural_player_model(
                        player_data_dict[position], position, target_col
                    )
                    neural_player_models[position] = (model, encoder, player_data, target_scaler)

            neural_models['neural_players'] = neural_player_models

        except Exception as e:
            print(f"❌ Failed to train neural player models: {e}")

    else:
        print("\n❌ Binary model not available, skipping all neural model training.")

    # Store neural models in the main models dictionary
    models_dict['neural'] = neural_models

    display_model_performance_summary(models_dict)
    
    print("\n🎮 Starting Interactive Prediction Interface...")
    continue_loop = True
    while continue_loop:
        continue_loop = run_enhanced_prediction_interface(models_dict)
        if continue_loop:
            input("\nPress Enter to make another prediction...")

if __name__ == '__main__':
    main()