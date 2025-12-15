# Home Advantage Testing - Implementation Summary

## Changes Made to `KenPomWLS`

### 1. New Parameter: `home_advantage`

The `KenPomWLS` class now accepts an optional `home_advantage` parameter:

```python
KenPomWLS(games_df, shrink_k=10, home_advantage=None)
```

- **`home_advantage=None`** (default): Estimates home court advantage from the data (original behavior)
- **`home_advantage=0`**: Assumes no home court advantage
- **`home_advantage=1`**: Assumes 1 point per 100 possessions home advantage
- **`home_advantage=2`**: Assumes 2 points per 100 possessions home advantage
- **`home_advantage=3`**: Assumes 3 points per 100 possessions home advantage
- **`home_advantage=4`**: Assumes 4 points per 100 possessions home advantage

### 2. Updated Column Naming

Column names now include both shrink factor AND home advantage:

| Old Format | New Format (estimated) | New Format (fixed) |
|------------|----------------------|-------------------|
| `AdjO_k10` | `AdjO_k10_hEst` | `AdjO_k10_h3` |
| `AdjD_k10` | `AdjD_k10_hEst` | `AdjD_k10_h3` |
| `AdjMargin_k10` | `AdjMargin_k10_hEst` | `AdjMargin_k10_h3` |
| `games_played_k10` | `games_played_k10_hEst` | `games_played_k10_h3` |

Examples:
- `t1_AdjMargin_k10_h0` = Team1's adjusted margin with shrink_k=10, home_advantage=0
- `t1_AdjMargin_k10_h3` = Team1's adjusted margin with shrink_k=10, home_advantage=3
- `t1_AdjMargin_k10_hEst` = Team1's adjusted margin with shrink_k=10, estimated home advantage

### 3. How It Works

When `home_advantage` is specified:
- The WLS solver does NOT estimate a home effect parameter
- Instead, it subtracts the fixed home advantage from observed ratings before solving
- This constrains the solution to assume that specific home court value

When `home_advantage=None`:
- The WLS solver estimates the home effect as an additional parameter (original behavior)
- The estimated value is returned in the detailed output

## Usage in data_pipeline.ipynb

Add this code after loading `regular_season_games` to generate features for all home advantage values:

```python
# Generate KenPomWLS features for different home advantage values
print("Generating KenPomWLS features with different home advantages...")

for home_adv in [0, 1, 2, 3, 4]:
    print(f"  Processing home_advantage={home_adv}...")
    to_predict_mens = feature_engineering.KenPomWLS(
        regular_season_games, 
        shrink_k=10, 
        home_advantage=home_adv
    ).add(to_predict_mens)

print("Done!")
```

This will create columns like:
- `t1_AdjMargin_k10_h0`, `t2_AdjMargin_k10_h0`
- `t1_AdjMargin_k10_h1`, `t2_AdjMargin_k10_h1`
- `t1_AdjMargin_k10_h2`, `t2_AdjMargin_k10_h2`
- `t1_AdjMargin_k10_h3`, `t2_AdjMargin_k10_h3`
- `t1_AdjMargin_k10_h4`, `t2_AdjMargin_k10_h4`

## Testing in assess_new_efficiencies.ipynb

After running the updated `data_pipeline.ipynb` to generate the dataset, copy the cells from `home_advantage_test_cells.md` into your assessment notebook.

The test cells will:

1. **Setup**: Define the model and parameter grid
2. **Test each value**: Run evaluation for home_advantage = 0, 1, 2, 3, 4
3. **Compare**: Create a comparison table and visualization showing which home advantage assumption performs best

### Expected Output

You'll get:
- Individual evaluation results for each home advantage value
- A comparison table sorted by performance
- Visualization plots showing how model performance varies with home advantage assumption
- Identification of the optimal home advantage value for your prediction task

## Workflow

1. **Edit `data_pipeline.ipynb`**: Add the code to generate features for home_advantage values 0, 1, 2, 3, 4
2. **Run `data_pipeline.ipynb`**: This will create the updated `to_predict_mens.csv` with all the new columns
3. **Open `assess_new_efficiencies.ipynb`**: Add the test cells from `home_advantage_test_cells.md`
4. **Run the test cells**: Compare performance across different home advantage assumptions

## Example Results Interpretation

If the comparison shows:
- **home_advantage=3 has lowest Brier score**: Use 3 points/100 poss as your home advantage assumption
- **home_advantage=0 performs best**: Home court may not matter much in tournament games (all neutral sites)
- **All values perform similarly**: The model is robust to this assumption; stick with estimated value

## Backward Compatibility

The default behavior (`home_advantage=None`) maintains backward compatibility:
- Old code still works without changes
- Columns will have `_hEst` suffix instead of no suffix (minor breaking change in column names)
- To get the old exact column names, you'd need to manually rename after processing

## Technical Details

The implementation modifies two methods:

1. **`build_matrix()`**: When `home_advantage` is fixed, it:
   - Removes the home effect column from the design matrix
   - Subtracts the fixed home advantage from the target variable `y`

2. **`solve_wls()`**: When `home_advantage` is fixed, it:
   - Doesn't extract a home effect from the solution
   - Uses the user-specified value directly

This ensures the ratings are optimized assuming that specific home court value.
