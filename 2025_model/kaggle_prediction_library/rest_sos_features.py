import pandas as pd
import numpy as np


def add_rest_sos_features(games_df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """
    Add point-in-time rest, schedule density, and SoS features.

    Designed to work with PreProcess regular_season_games + point-in-time WLS,
    i.e. a DataFrame that already has:

        Season, DayNum, Team1, Team2, Outcome, margin,
        t1_adj_margin, t2_adj_margin

    Strategy:
      1. Build a canonical (1 row per physical game) internal table for computing
         schedule histories WITHOUT leakage.
      2. Compute per-team schedule features (rest_days, games_last_window, sos_pt)
         on that canonical table.
      3. Merge the schedule features back to the ORIGINAL input DataFrame by
         (Season, DayNum, Team1, Team2), preserving all original columns.
    """

    orig = games_df.copy()

    # ---------------------------------------------------------------------------
    # Step 1: Build canonical 1-row-per-game table for schedule computation
    # ---------------------------------------------------------------------------
    # Canonical key: (Season, DayNum, minTeam, maxTeam)
    orig["_team_min"] = orig[["Team1", "Team2"]].min(axis=1)
    orig["_team_max"] = orig[["Team1", "Team2"]].max(axis=1)

    # Deduplicate to 1 row per physical game (keep first occurrence)
    canonical = (
        orig
        .drop_duplicates(subset=["Season", "DayNum", "_team_min", "_team_max"], keep="first")
        .copy()
    )

    # For schedule features we need: Season, DayNum, the two teams, and their adj_margins
    # Remap to canonical orientation: Team1 = _team_min, Team2 = _team_max
    # Also remap t1/t2_adj_margin accordingly
    canonical["_orig_t1"] = canonical["Team1"]
    canonical["_need_swap"] = canonical["Team1"] != canonical["_team_min"]

    # Swap adj margins where needed
    canonical["_can_t1_adj"] = np.where(
        canonical["_need_swap"],
        canonical["t2_adj_margin"],
        canonical["t1_adj_margin"]
    )
    canonical["_can_t2_adj"] = np.where(
        canonical["_need_swap"],
        canonical["t1_adj_margin"],
        canonical["t2_adj_margin"]
    )

    # Now set canonical Team1/Team2
    canonical["Team1"] = canonical["_team_min"]
    canonical["Team2"] = canonical["_team_max"]
    canonical["t1_adj_margin"] = canonical["_can_t1_adj"]
    canonical["t2_adj_margin"] = canonical["_can_t2_adj"]

    # Sort for schedule computation
    canonical = canonical.sort_values(["Season", "DayNum"]).reset_index(drop=True)
    canonical["_game_idx"] = canonical.index

    # ---------------------------------------------------------------------------
    # Step 2: Compute per-team schedule features on canonical table
    # ---------------------------------------------------------------------------
    # Long form: one row per team-game
    t1_long = canonical[["Season", "DayNum", "_game_idx", "Team1", "t2_adj_margin"]].rename(
        columns={"Team1": "TeamID", "t2_adj_margin": "opp_adj_margin"}
    )
    t2_long = canonical[["Season", "DayNum", "_game_idx", "Team2", "t1_adj_margin"]].rename(
        columns={"Team2": "TeamID", "t1_adj_margin": "opp_adj_margin"}
    )

    long = pd.concat([t1_long, t2_long], ignore_index=True)
    long = long.sort_values(["Season", "TeamID", "DayNum", "_game_idx"]).reset_index(drop=True)

    # Rest days: gap from previous game
    long["rest_days"] = long.groupby(["Season", "TeamID"])["DayNum"].diff()

    # SoS to date: mean past opponent adj_margin (point-in-time safe)
    def past_cummean(series: pd.Series) -> pd.Series:
        return series.shift(1).expanding(min_periods=1).mean()

    long["sos_pt"] = long.groupby(["Season", "TeamID"])["opp_adj_margin"].transform(past_cummean)

    # Schedule density: number of games in [d - window_days, d)
    def games_last_k_days_fast(group: pd.DataFrame) -> pd.Series:
        daynums = group["DayNum"].values
        result = np.zeros(len(daynums), dtype=int)
        for i in range(len(daynums)):
            d = daynums[i]
            result[i] = ((daynums[:i] >= d - window_days) & (daynums[:i] < d)).sum()
        return pd.Series(result, index=group.index)

    long["games_last_window"] = long.groupby(["Season", "TeamID"], group_keys=False).apply(
        games_last_k_days_fast
    )

    # ---------------------------------------------------------------------------
    # Step 3: Merge schedule features back to ORIGINAL input DataFrame
    # ---------------------------------------------------------------------------
    # We need to join on the actual Team1/Team2 in the original, not the canonical ones.
    # So we join per-team features by (Season, DayNum, TeamID).

    team_feats = long[["Season", "DayNum", "TeamID", "rest_days", "games_last_window", "sos_pt"]].copy()

    # Merge for original Team1
    orig = orig.merge(
        team_feats.rename(columns={
            "TeamID": "Team1",
            "rest_days": "t1_rest_days",
            "games_last_window": "t1_games_last_window",
            "sos_pt": "t1_sos_pt",
        }),
        on=["Season", "DayNum", "Team1"],
        how="left"
    )

    # Merge for original Team2
    orig = orig.merge(
        team_feats.rename(columns={
            "TeamID": "Team2",
            "rest_days": "t2_rest_days",
            "games_last_window": "t2_games_last_window",
            "sos_pt": "t2_sos_pt",
        }),
        on=["Season", "DayNum", "Team2"],
        how="left"
    )

    # Fill early-season NaNs
    for col in ["t1_rest_days", "t2_rest_days", "t1_sos_pt", "t2_sos_pt"]:
        orig[col] = orig[col].fillna(0.0)
    for col in ["t1_games_last_window", "t2_games_last_window"]:
        orig[col] = orig[col].fillna(0).astype(int)

    # Compute diffs (in original Team1/Team2 orientation)
    orig["rest_days_diff"] = orig["t1_rest_days"] - orig["t2_rest_days"]
    orig["games_last_window_diff"] = orig["t1_games_last_window"] - orig["t2_games_last_window"]
    orig["sos_pt_diff"] = orig["t1_sos_pt"] - orig["t2_sos_pt"]

    # Drop internal helper columns
    orig = orig.drop(columns=["_team_min", "_team_max"], errors="ignore")

    return orig