# KenPom WLS Implementation Guide

## Overview

A new `KenPomWLS` feature class has been added to `feature_engineering.py` that implements KenPom-style adjusted offensive and defensive efficiency calculations using Weighted Least Squares (WLS).

## Key Differences from Original Efficiency

| Aspect | Original Efficiency | KenPomWLS |
|--------|-------------------|-----------|
| **Method** | Rolling averages with opponent adjustments | Single WLS linear system |
| **Possessions** | `0.96 * (FGA + TO + 0.44*FTA - OR)` | `FGA - OR + TO + 0.475*FTA` (KenPom formula) |
| **Tempo** | Per-team calculation | Averaged between both teams |
| **Opponent Adjustment** | Rolling opponent efficiency | Simultaneous Off/Def estimation |
| **Home Court** | Multiplicative bonus/penalty | Additive effect in linear model |
| **Shrinkage** | None (early season noise) | Bayesian shrinkage toward league mean |
| **Weighting** | Equal weight per game | Weighted by possessions |

## Mathematical Approach

The KenPomWLS implementation:

1. **Computes possessions** using KenPom's formula: `FGA - OR + TO + 0.475*FTA`
2. **Averages possessions** between both teams (tempo equalization)
3. **Builds linear system**: For each game, creates two equations:
   - Team1: `Off_team1 - Def_team2 + Home_flag ≈ OffRtg_team1`
   - Team2: `Off_team2 - Def_team1 - Home_flag ≈ OffRtg_team2`
4. **Weights observations** by possessions (more possessions = more reliable data)
5. **Solves WLS** to get offensive and defensive ratings for all teams
6. **Anchors to league mean** (ensures interpretability)
7. **Applies shrinkage** toward league mean based on games played (reduces early-season noise)

## Usage

### Basic Usage

```python
from kaggle_prediction_library.feature_engineering import KenPomWLS

# Initialize with your game data
kenpom_wls = KenPomWLS(games_df, shrink_k=10)

# Generate ratings
ratings = kenpom_wls.process()
# Returns: DataFrame with ['Season', 'TeamID', 'AdjO', 'AdjD', 'AdjMargin', 'games_played']
```

### Get Additional Details

```python
# Get ratings plus home effect and league mean
ratings, details = kenpom_wls.process(return_detailed=True)

print(f"Home court advantage: {details['home_effect_per100']:.2f} pts/100 poss")
print(f"League mean OffRtg: {details['league_mean']:.2f}")
```

### Add to Prediction Pipeline

```python
# Add features to your base dataset (inherits from FeatureEng)
base_with_features = kenpom_wls.add(base_predictions)

# This automatically adds:
# - t1_AdjO, t1_AdjD, t1_AdjMargin, t1_games_played
# - t2_AdjO, t2_AdjD, t2_AdjMargin, t2_games_played
```

## Parameter Tuning

### shrink_k Parameter

Controls how much to shrink toward the league mean:

- **Lower values (e.g., 5)**: Less shrinkage, ratings closer to raw estimates
  - Good if you trust early-season data
  - More variance, especially early in the season
  
- **Higher values (e.g., 15-20)**: More shrinkage toward league mean
  - More conservative estimates
  - Better for teams with few games
  - Reduces overfitting

**Default**: `shrink_k=10` (balanced approach)

**Formula**: `shrink_weight = games_played / (games_played + shrink_k)`

## Expected Data Format

Your `games_df` should have these columns:

| Column | Type | Description |
|--------|------|-------------|
| Season | int | Season year |
| DayNum | int | Day number in season |
| Team1 | int | Team 1 ID |
| Team2 | int | Team 2 ID |
| Team1_score | int | Team 1 final score |
| Team2_score | int | Team 2 final score |
| Team1_FGA | int | Team 1 field goal attempts |
| Team2_FGA | int | Team 2 field goal attempts |
| Team1_TO | int | Team 1 turnovers |
| Team2_TO | int | Team 2 turnovers |
| Team1_FTA | int | Team 1 free throw attempts |
| Team2_FTA | int | Team 2 free throw attempts |
| Team1_OR | int | Team 1 offensive rebounds |
| Team2_OR | int | Team 2 offensive rebounds |
| Loc | str | Location: 'H' (Team1 home), 'A' (Team1 away), 'N' (neutral) |

## Testing & Validation

### Run the Test Script

```bash
cd /Users/skylerdale/workspace/cbb/kaggle-ncaam/2025_model
python test_kenpom_wls.py
```

### Compare with Original Efficiency

The test script includes a comparison function to see how the new method differs:

```python
from test_kenpom_wls import compare_with_original_efficiency

comparison = compare_with_original_efficiency(games_df)
```

This will show:
- Correlation between methods
- Side-by-side comparison of ratings
- Which teams are rated differently

## Integration into Your Pipeline

### Option 1: Replace Original Efficiency

```python
# Replace this:
# efficiency = Efficiency(games, away_bonus=0.014)

# With this:
efficiency = KenPomWLS(games, shrink_k=10)

# Rest of pipeline stays the same
features = efficiency.process()
base_with_features = efficiency.add(base_predictions)
```

### Option 2: Use Both (A/B Testing)

```python
# Keep both features
original_eff = Efficiency(games, away_bonus=0.014)
kenpom_wls = KenPomWLS(games, shrink_k=10)

# Add both to your dataset with different prefixes
base = original_eff.add(base)  # Adds t1_adj_oe, t1_adj_de, etc.
base = kenpom_wls.add(base)    # Adds t1_AdjO, t1_AdjD, etc.

# Train models with both and compare performance
```

## Advantages

1. **Theoretically Sound**: Based on KenPom's proven methodology
2. **Handles Dependencies**: Simultaneously estimates offense and defense
3. **Better Early Season**: Shrinkage prevents overfitting with limited data
4. **Possession Weighting**: More reliable games get more weight
5. **Tempo Equalization**: Properly accounts for pace differences
6. **Home Court**: Explicitly modeled as separate parameter

## Next Steps

1. ✅ Implementation added to `feature_engineering.py`
2. ✅ Test script created
3. ⏳ Load your historical game data
4. ⏳ Run test script to validate
5. ⏳ Compare with original Efficiency class
6. ⏳ Train models with new features
7. ⏳ Evaluate model performance improvements
8. ⏳ Tune `shrink_k` parameter if needed
9. ⏳ Integrate into daily prediction pipeline (when box scores available)

## Questions?

- How do ratings compare to actual KenPom? (Should be very close)
- Which shrink_k value works best for your models?
- Does this improve prediction accuracy vs original Efficiency?
- Should you use both features or just one?
