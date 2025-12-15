# Quick Start: Home Advantage Testing

## Step 1: Update data_pipeline.ipynb

Find the section where you call `KenPomWLS` and replace it with this loop:

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

## Step 2: Run data_pipeline.ipynb

Execute all cells to generate `to_predict_mens.csv` with the new columns:
- `t1_AdjMargin_k10_h0`, `t2_AdjMargin_k10_h0`
- `t1_AdjMargin_k10_h1`, `t2_AdjMargin_k10_h1`
- `t1_AdjMargin_k10_h2`, `t2_AdjMargin_k10_h2`
- `t1_AdjMargin_k10_h3`, `t2_AdjMargin_k10_h3`
- `t1_AdjMargin_k10_h4`, `t2_AdjMargin_k10_h4`

## Step 3: Copy cells into assess_new_efficiencies.ipynb

Open `home_advantage_test_cells.md` and copy cells 1-7 into your notebook.

## Step 4: Run the assessment cells

Execute each cell to:
1. Test home_advantage = 0, 1, 2, 3, 4
2. See individual results for each
3. View comparison table and plots

## Expected Runtime

- data_pipeline.ipynb: ~5-10 minutes (depending on data size)
- Assessment cells: ~2-3 minutes per home advantage value = ~10-15 minutes total

## What You'll Learn

The comparison will show you:
- Which home advantage assumption gives the best predictive performance
- How sensitive your model is to this parameter
- Whether home court matters for tournament predictions

## Column Naming Convention

Format: `{prefix}_{metric}_k{shrink}_h{home}`

Examples:
- `t1_AdjMargin_k10_h3` = Team1, AdjMargin, shrink_k=10, home_advantage=3
- `t2_AdjO_k10_h0` = Team2, AdjO, shrink_k=10, home_advantage=0
- `t1_games_played_k10_h2` = Team1, games_played, shrink_k=10, home_advantage=2
