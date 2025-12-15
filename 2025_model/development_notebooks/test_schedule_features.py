import pandas as pd
import numpy as np

from schedule_feature_sanity import build_schedule_features


def make_simple_schedule():
    """Construct a tiny, fully-transparent toy schedule to test schedule features.

    Scenario:
    - Season 2020
    - Three teams: 1, 2, 3
    - Games:
      Day 10: Team1 vs Team2, Team1 wins by 5 (margin=+5)
      Day 13: Team1 vs Team3, Team1 wins by 7 (margin=+7)
      Day 15: Team2 vs Team3, Team2 wins by 3 (margin=+3)
    """
    rows = [
        # Season, DayNum, Team1, Team2, Outcome, margin
        (2020, 10, 1, 2, 1,  +5),
        (2020, 13, 1, 3, 1,  +7),
        (2020, 15, 2, 3, 1,  +3),
    ]
    cols = ["Season", "DayNum", "Team1", "Team2", "Outcome", "margin"]
    return pd.DataFrame(rows, columns=cols)


def print_team_game_view(df):
    """Print per-team schedule features as produced by build_schedule_features."""
    print("\nPer-team game view (Season, TeamID, DayNum, rest_days, games_last_window, sos_margin_pt):")

    # Use build_schedule_features to ensure we're inspecting the exact same logic
    enriched = build_schedule_features(df, window_days=7)

    t1_long = enriched[[
        "Season", "DayNum", "Team1",
        "t1_rest_days", "t1_games_last_window", "t1_sos_margin_pt",
    ]].rename(
        columns={
            "Team1": "TeamID",
            "t1_rest_days": "rest_days",
            "t1_games_last_window": "games_last_window",
            "t1_sos_margin_pt": "sos_margin_pt",
        }
    )

    t2_long = enriched[[
        "Season", "DayNum", "Team2",
        "t2_rest_days", "t2_games_last_window", "t2_sos_margin_pt",
    ]].rename(
        columns={
            "Team2": "TeamID",
            "t2_rest_days": "rest_days",
            "t2_games_last_window": "games_last_window",
            "t2_sos_margin_pt": "sos_margin_pt",
        }
    )

    long = pd.concat([t1_long, t2_long], ignore_index=True)
    long = long.sort_values(["Season", "TeamID", "DayNum"]).reset_index(drop=True)

    print(long[["Season", "TeamID", "DayNum", "rest_days", "games_last_window", "sos_margin_pt"]])


def run_manual_test():
    games = make_simple_schedule()
    print("Base games:")
    print(games)

    enriched = build_schedule_features(games, window_days=7)
    print("\nEnriched game-level schedule features:")
    print(enriched[[
        "Season", "DayNum", "Team1", "Team2", "Outcome", "margin",
        "t1_rest_days", "t2_rest_days",
        "t1_games_last_window", "t2_games_last_window",
        "t1_sos_margin_pt", "t2_sos_margin_pt",
        "rest_days_diff", "games_last_7_diff", "sos_margin_pt_diff",
    ]])

    # Also print per-team view for clarity
    print_team_game_view(games)


if __name__ == "__main__":
    run_manual_test()
