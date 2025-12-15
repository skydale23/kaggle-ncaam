# Home Advantage Test Cells
## Add these cells to assess_new_efficiencies.ipynb after running data_pipeline.ipynb

---

## Cell 1: Setup

```python
# Define the classifier and parameter grid
model = LogisticRegression(C=0.05)
pipeline = make_pipeline(StandardScaler(), model)
param_grid = {
    'logisticregression__C': [.005, 0.001, .05, 0.01, 0.1],
}

to_predict_mens_recent = to_predict_mens[(to_predict_mens.Season >= 2009)
        # filter out first four games
        & (to_predict_mens.GameRound >= 1)
        ]

# Store results for comparison
home_advantage_results = []
```

---

## Cell 2: Test Home Advantage = 0

```python
print("Testing Home Advantage = 0...")
baseline_features = ['t1_AdjMargin_k10_h0', 't2_AdjMargin_k10_h0']
eval_df = validation.run_evaluation_framework(to_predict_mens_recent, pipeline, baseline_features, param_grid, cv_start=2013)
home_advantage_results.append({
    'home_advantage': 0,
    'best_params': eval_df['best_params'].iloc[0],
    'mean_cv': eval_df['mean_repeated_cv_score'].iloc[0],
    'rolling_cv': eval_df['rolling_season_cv'].iloc[0]
})
display(eval_df)
```

---

## Cell 3: Test Home Advantage = 1

```python
print("Testing Home Advantage = 1...")
baseline_features = ['t1_AdjMargin_k10_h1', 't2_AdjMargin_k10_h1']
eval_df = validation.run_evaluation_framework(to_predict_mens_recent, pipeline, baseline_features, param_grid, cv_start=2013)
home_advantage_results.append({
    'home_advantage': 1,
    'best_params': eval_df['best_params'].iloc[0],
    'mean_cv': eval_df['mean_repeated_cv_score'].iloc[0],
    'rolling_cv': eval_df['rolling_season_cv'].iloc[0]
})
display(eval_df)
```

---

## Cell 4: Test Home Advantage = 2

```python
print("Testing Home Advantage = 2...")
baseline_features = ['t1_AdjMargin_k10_h2', 't2_AdjMargin_k10_h2']
eval_df = validation.run_evaluation_framework(to_predict_mens_recent, pipeline, baseline_features, param_grid, cv_start=2013)
home_advantage_results.append({
    'home_advantage': 2,
    'best_params': eval_df['best_params'].iloc[0],
    'mean_cv': eval_df['mean_repeated_cv_score'].iloc[0],
    'rolling_cv': eval_df['rolling_season_cv'].iloc[0]
})
display(eval_df)
```

---

## Cell 5: Test Home Advantage = 3

```python
print("Testing Home Advantage = 3...")
baseline_features = ['t1_AdjMargin_k10_h3', 't2_AdjMargin_k10_h3']
eval_df = validation.run_evaluation_framework(to_predict_mens_recent, pipeline, baseline_features, param_grid, cv_start=2013)
home_advantage_results.append({
    'home_advantage': 3,
    'best_params': eval_df['best_params'].iloc[0],
    'mean_cv': eval_df['mean_repeated_cv_score'].iloc[0],
    'rolling_cv': eval_df['rolling_season_cv'].iloc[0]
})
display(eval_df)
```

---

## Cell 6: Test Home Advantage = 4

```python
print("Testing Home Advantage = 4...")
baseline_features = ['t1_AdjMargin_k10_h4', 't2_AdjMargin_k10_h4']
eval_df = validation.run_evaluation_framework(to_predict_mens_recent, pipeline, baseline_features, param_grid, cv_start=2013)
home_advantage_results.append({
    'home_advantage': 4,
    'best_params': eval_df['best_params'].iloc[0],
    'mean_cv': eval_df['mean_repeated_cv_score'].iloc[0],
    'rolling_cv': eval_df['rolling_season_cv'].iloc[0]
})
display(eval_df)
```

---

## Cell 7: Compare All Results

```python
# Create comparison DataFrame
comparison_df = pd.DataFrame(home_advantage_results)
comparison_df = comparison_df.sort_values('rolling_cv')

print("\n" + "="*60)
print("HOME ADVANTAGE COMPARISON (sorted by Rolling Season CV)")
print("="*60)
display(comparison_df)

print("\nBest Home Advantage:", comparison_df.iloc[0]['home_advantage'])
print("Best Rolling CV Score:", comparison_df.iloc[0]['rolling_cv'])

# Visualize the results
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Rolling CV by Home Advantage
ax1.plot(comparison_df['home_advantage'], comparison_df['rolling_cv'], marker='o', linewidth=2, markersize=8)
ax1.set_xlabel('Home Advantage (points per 100 possessions)', fontsize=12)
ax1.set_ylabel('Rolling Season CV (Brier Score)', fontsize=12)
ax1.set_title('Model Performance vs Home Advantage', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=comparison_df['rolling_cv'].min(), color='r', linestyle='--', alpha=0.3, label='Best')

# Plot 2: Mean CV by Home Advantage
ax2.plot(comparison_df['home_advantage'], comparison_df['mean_cv'].abs(), marker='s', linewidth=2, markersize=8, color='orange')
ax2.set_xlabel('Home Advantage (points per 100 possessions)', fontsize=12)
ax2.set_ylabel('Mean Repeated CV (Brier Score)', fontsize=12)
ax2.set_title('Mean CV vs Home Advantage', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Instructions for data_pipeline.ipynb

Add these lines to generate the features for all home advantage values (after loading regular_season_games):

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
