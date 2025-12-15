import pandas as pd
import numpy as np
import sys

sys.path.append('../kaggle_prediction_library/')
import preprocess

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score

import warnings
warnings.filterwarnings('ignore')


def load_regular_season_games():
    regular_season_results = pd.read_csv('../data/MRegularSeasonDetailedResults.csv')
    detailed_tourney_results = pd.read_csv('../data/MNCAATourneyDetailedResults.csv')
    regular_season_results_w = pd.read_csv('../data/WRegularSeasonDetailedResults.csv')
    detailed_tourney_results_w = pd.read_csv('../data/WNCAATourneyDetailedResults.csv')
    mteams = pd.read_csv('../data/MTeams.csv')
    sub_df = pd.read_csv('SampleSubmission2024.csv')

    _, _, regular_season_games, _ = preprocess.full_setup(
        detailed_tourney_results,
        regular_season_results,
        detailed_tourney_results_w,
        regular_season_results_w,
        sub_df,
        mteams,
    )

    return regular_season_games


def build_schedule_features(games_df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """Build schedule features (rest, recent density, running margin) in a
    way that is robust to the winner/loser duplication used in PreProcess.

    PreProcess produces two rows per physical game: one with Team1=winner,
    Outcome=1, and one with Team1=loser, Outcome=0. For schedule logic we
    should treat each physical game only once. Here we collapse duplicated
    games to a single canonical row before constructing per-team histories.
    """

    # Work on a copy and create a canonical game key that is invariant to
    # winner/loser orientation: (Season, DayNum, sorted(Team1,Team2), abs(margin)).
    base = games_df.copy()

    base['team_min'] = base[['Team1', 'Team2']].min(axis=1)
    base['team_max'] = base[['Team1', 'Team2']].max(axis=1)
    base['abs_margin'] = base['margin'].abs()

    # Deterministically collapse the two PreProcess rows per physical game into
    # a single canonical row with:
    #   Team1 = team_min, Team2 = team_max,
    #   margin = Team1_score - Team2_score (signed w.r.t canonical Team1),
    #   Outcome = 1 if canonical Team1 actually won, else 0.
    group_cols = ['Season', 'DayNum', 'team_min', 'team_max', 'abs_margin']

    def canonicalize_game(g: pd.DataFrame) -> pd.DataFrame:
        season = g['Season'].iloc[0]
        day = g['DayNum'].iloc[0]
        team_min = g['team_min'].iloc[0]
        team_max = g['team_max'].iloc[0]
        abs_margin = g['abs_margin'].iloc[0]

        # Find the winner-oriented row from PreProcess (Outcome==1, Team1=winner)
        winners = g[g['Outcome'] == 1]
        if len(winners) == 0:
            # Fallback: infer winner from margin sign in the first row
            row0 = g.iloc[0]
            if row0['margin'] > 0:
                winner_id = row0['Team1']
            else:
                winner_id = row0['Team2']
        else:
            winner_id = winners['Team1'].iloc[0]

        # Canonical orientation
        if winner_id == team_min:
            margin = abs_margin
            outcome = 1
        else:
            margin = -abs_margin
            outcome = 0

        return pd.DataFrame({
            'Season': [season],
            'DayNum': [day],
            'Team1': [team_min],
            'Team2': [team_max],
            'margin': [margin],
            'Outcome': [outcome],
        })

    canonical_games = (
        base
        .groupby(group_cols, as_index=False, group_keys=False)
        .apply(canonicalize_game)
        .reset_index(drop=True)
    )

    # Use the canonical game-level table for schedule computation.
    df = canonical_games.sort_values(['Season', 'DayNum']).reset_index(drop=True).copy()
    df['game_id'] = df.index

    # Long form: one row per team-game, margin from that team's POV
    t1 = df[['Season', 'DayNum', 'game_id', 'Team1', 'margin']].rename(
        columns={'Team1': 'TeamID', 'margin': 'margin_for'}
    )
    t2 = df[['Season', 'DayNum', 'game_id', 'Team2', 'margin']].rename(
        columns={'Team2': 'TeamID', 'margin': 'margin_for'}
    )
    t2['margin_for'] = -t2['margin_for']

    long = pd.concat([t1, t2], ignore_index=True)
    long = long.sort_values(['Season', 'TeamID', 'DayNum', 'game_id']).reset_index(drop=True)

    def per_team_schedule(group: pd.DataFrame) -> pd.DataFrame:
        days = group['DayNum'].values
        margins = group['margin_for'].values
        n = len(group)

        # rest_days: gap from previous game (0 for first)
        rest = np.zeros(n, dtype=float)
        if n > 1:
            rest[1:] = days[1:] - days[:-1]

        # games_last_window: count games in [d - window_days, d)
        games_last = np.zeros(n, dtype=int)
        for i in range(n):
            d = days[i]
            games_last[i] = ((days[:i] >= d - window_days) & (days[:i] < d)).sum()

        # sos_margin_pt: mean margin_for of previous games
        sos = np.zeros(n, dtype=float)
        if n > 1:
            csum = np.cumsum(margins)
            counts = np.arange(1, n + 1)
            prev_mean = csum[:-1] / counts[:-1]
            sos[1:] = prev_mean

        out = group.copy()
        out['rest_days'] = rest
        out['games_last_window'] = games_last
        out['sos_margin_pt'] = sos
        return out

    long = long.groupby(['Season', 'TeamID'], group_keys=False).apply(per_team_schedule)

    t1_feats = long[['Season', 'DayNum', 'game_id', 'TeamID',
                     'rest_days', 'games_last_window', 'sos_margin_pt']].rename(
        columns={
            'TeamID': 'Team1',
            'rest_days': 't1_rest_days',
            'games_last_window': 't1_games_last_window',
            'sos_margin_pt': 't1_sos_margin_pt',
        }
    )
    t2_feats = long[['Season', 'DayNum', 'game_id', 'TeamID',
                     'rest_days', 'games_last_window', 'sos_margin_pt']].rename(
        columns={
            'TeamID': 'Team2',
            'rest_days': 't2_rest_days',
            'games_last_window': 't2_games_last_window',
            'sos_margin_pt': 't2_sos_margin_pt',
        }
    )

    df = df.merge(t1_feats, on=['Season', 'DayNum', 'game_id', 'Team1'], how='left')
    df = df.merge(t2_feats, on=['Season', 'DayNum', 'game_id', 'Team2'], how='left')

    df['rest_days_diff'] = df['t1_rest_days'] - df['t2_rest_days']
    df['games_last_7_diff'] = df['t1_games_last_window'] - df['t2_games_last_window']
    df['sos_margin_pt_diff'] = df['t1_sos_margin_pt'] - df['t2_sos_margin_pt']

    return df


def run_schedule_sanity():
    print("Loading regular season games...")
    games = load_regular_season_games()
    print(f"Regular season games: {games.shape[0]:,}")

    print("\nBuilding schedule-only features (rest, density, SoS margin)...")
    games_with_sched = build_schedule_features(games, window_days=7)
    print(f"games_with_sched shape: {games_with_sched.shape}")

    sched_feature_cols = ['rest_days_diff', 'games_last_7_diff', 'sos_margin_pt_diff']

    model_sched = games_with_sched.copy()
    train_mask = model_sched['Season'] < 2020
    test_mask = model_sched['Season'] >= 2020

    X_train = model_sched[train_mask][sched_feature_cols]
    y_train = model_sched[train_mask]['Outcome']

    X_test = model_sched[test_mask][sched_feature_cols]
    y_test = model_sched[test_mask]['Outcome']

    print(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', lr),
    ])

    param_grid = {
        'lr__C': [0.01, 0.1, 1.0, 10.0, 100.0]
    }

    cv = TimeSeriesSplit(n_splits=5)

    lr_grid = GridSearchCV(
        estimator=pipe_lr,
        param_grid=param_grid,
        scoring='neg_brier_score',
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )

    # With random per-game dedupe it's possible (though unlikely) that the
    # training set ends up with only one class. In that case, skip LR and
    # focus on GBM diagnostics.
    if y_train.nunique() < 2:
        print("\nSkipping Logistic Regression: training labels contain only one class (", y_train.iloc[0], ")")
        y_proba_lr = np.full_like(y_test, y_test.mean(), dtype=float)
    else:
        lr_grid.fit(X_train, y_train)
        print("Best LR params (schedule-only):", lr_grid.best_params_)

        y_proba_lr = lr_grid.predict_proba(X_test)[:, 1]
        y_pred_lr = (y_proba_lr >= 0.5).astype(int)

        print("\nLogistic Regression (schedule-only) test performance:")
        print(f"  brier_score: {brier_score_loss(y_test, y_proba_lr):.6f}")
        print(f"  log_loss:    {log_loss(y_test, y_proba_lr):.6f}")
        print(f"  roc_auc:     {roc_auc_score(y_test, y_proba_lr):.6f}")
        print(f"  accuracy:    {accuracy_score(y_test, y_pred_lr):.6f}")

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
    )

    gb.fit(X_train, y_train)
    y_proba_gb = gb.predict_proba(X_test)[:, 1]
    y_pred_gb = (y_proba_gb >= 0.5).astype(int)

    print("\nGradient Boosting (schedule-only) test performance:")
    print(f"  brier_score: {brier_score_loss(y_test, y_proba_gb):.6f}")
    print(f"  log_loss:    {log_loss(y_test, y_proba_gb):.6f}")
    print(f"  roc_auc:     {roc_auc_score(y_test, y_proba_gb):.6f}")
    print(f"  accuracy:    {accuracy_score(y_test, y_pred_gb):.6f}")

    # === DayNum tranche analysis for GBM (schedule-only) ===
    print("\n=== GBM (schedule-only) performance by DayNum tranche (test set) ===")

    test_df = model_sched[test_mask].copy()
    test_df = test_df.reset_index(drop=True)
    test_df['gb_proba_sched'] = y_proba_gb

    # Define tranches as (label, lower_inclusive, upper_exclusive)
    tranches = [
        ("[  1,  50)", 1, 50),
        ("[ 50, 100)", 50, 100),
        ("[100, 150)", 100, 150),
        ("[150, 200)", 150, 200),
        ("[200, 999]", 200, 1000),
    ]

    for label, lo, hi in tranches:
        mask = (test_df['DayNum'] >= lo) & (test_df['DayNum'] < hi)
        if not mask.any():
            print(f"Tranche {label}: no games")
            continue

        y_true_t = test_df.loc[mask, 'Outcome']
        y_prob_t = test_df.loc[mask, 'gb_proba_sched']

        brier_t = brier_score_loss(y_true_t, y_prob_t)
        logloss_t = log_loss(y_true_t, y_prob_t)
        auc_t = roc_auc_score(y_true_t, y_prob_t)

        print(f"Tranche {label}: n={len(y_true_t):5d}  brier={brier_t:.6f}  log_loss={logloss_t:.6f}  auc={auc_t:.6f}")

    # === Early-season calibration by probability bin (DayNum < 50) ===
    print("\n=== Early-season (DayNum < 50) calibration by GBM probability bin ===")
    early_mask = test_df['DayNum'] < 50
    early_df = test_df.loc[early_mask].copy()

    if early_df.empty:
        print("No early-season games in test set.")
    else:
        # Define bins: [0.0,0.1), ..., [0.9,1.0]
        bin_edges = np.linspace(0.0, 1.0, 11)
        bin_labels = [f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})" for i in range(9)] + ["[0.9, 1.0]"]

        # For the last bin, include prob == 1.0
        bins = pd.cut(
            early_df['gb_proba_sched'].clip(0.0, 1.0 - 1e-8),
            bins=bin_edges,
            right=False,
            labels=bin_labels,
        )
        early_df['prob_bin'] = bins

        calib = (
            early_df.groupby('prob_bin')['Outcome']
            .agg(['count', 'mean'])
            .rename(columns={'count': 'n', 'mean': 'win_rate'})
        )

        print(calib.to_string())

    # === Export test set with schedule features and predictions for manual inspection ===
    print("\nWriting test set with schedule features and predictions to CSV...")

    # Attach predictions back to the full model_sched frame for test rows
    test_df = model_sched[test_mask].copy()
    test_df = test_df.reset_index(drop=True)

    # Align predictions positionally with test_df
    test_df['lr_proba_sched'] = y_proba_lr
    test_df['gb_proba_sched'] = y_proba_gb

    export_cols = [
        'Season', 'DayNum', 'Team1', 'Team2', 'Outcome',
        'rest_days_diff', 'games_last_7_diff', 'sos_margin_pt_diff',
        'lr_proba_sched', 'gb_proba_sched',
    ]

    export_path = 'schedule_feature_sanity_test_predictions.csv'
    test_df[export_cols].to_csv(export_path, index=False)
    print(f"Saved test predictions to {export_path}")

    # SANITY CHECK: GBM on ONLY rest + density (no SoS)
    print("\n" + "="*60)
    print("SANITY: GBM on rest_days_diff + games_last_7_diff ONLY")
    print("="*60)
    
    rest_density_cols = ['rest_days_diff', 'games_last_7_diff']
    X_train_rd = model_sched[train_mask][rest_density_cols]
    X_test_rd = model_sched[test_mask][rest_density_cols]
    
    gb_rd = GradientBoostingClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
    )
    
    gb_rd.fit(X_train_rd, y_train)
    y_proba_gb_rd = gb_rd.predict_proba(X_test_rd)[:, 1]
    
    print("\nGBM on rest + density (no SoS) test performance:")
    print(f"  brier_score: {brier_score_loss(y_test, y_proba_gb_rd):.6f}")
    print(f"  log_loss:    {log_loss(y_test, y_proba_gb_rd):.6f}")
    print(f"  roc_auc:     {roc_auc_score(y_test, y_proba_gb_rd):.6f}")
    
    # SANITY CHECK: GBM on ONLY sos_margin_pt_diff
    print("\n" + "="*60)
    print("SANITY: GBM on sos_margin_pt_diff ONLY")
    print("="*60)
    
    sos_only_cols = ['sos_margin_pt_diff']
    X_train_sos = model_sched[train_mask][sos_only_cols]
    X_test_sos = model_sched[test_mask][sos_only_cols]
    
    gb_sos = GradientBoostingClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
    )
    
    gb_sos.fit(X_train_sos, y_train)
    y_proba_gb_sos = gb_sos.predict_proba(X_test_sos)[:, 1]
    y_pred_gb_sos = (y_proba_gb_sos >= 0.5).astype(int)
    
    print("\nGBM on sos_margin_pt_diff ONLY test performance:")
    print(f"  brier_score: {brier_score_loss(y_test, y_proba_gb_sos):.6f}")
    print(f"  log_loss:    {log_loss(y_test, y_proba_gb_sos):.6f}")
    print(f"  accuracy:    {accuracy_score(y_test, y_pred_gb_sos):.6f}")


if __name__ == "__main__":
    run_schedule_sanity()
