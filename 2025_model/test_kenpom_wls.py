"""
Test script for KenPomWLS feature implementation.

This script demonstrates how to use the new KenPomWLS class for computing
KenPom-style adjusted offensive and defensive efficiencies.
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from kaggle_prediction_library.feature_engineering import KenPomWLS

def load_sample_data():
    """
    Load sample game data for testing.
    
    Replace this with your actual data loading logic.
    For Kaggle data, you'd typically load from:
    - MRegularSeasonDetailedResults.csv
    - or similar files with box score data
    """
    # Example: Load your actual game data
    # games = pd.read_csv('path/to/MRegularSeasonDetailedResults.csv')
    
    # For this example, we'll show what columns are expected
    print("Expected columns for KenPomWLS:")
    print("  - Season, DayNum, Team1, Team2")
    print("  - Team1_score, Team2_score")
    print("  - Team1_FGA, Team2_FGA")
    print("  - Team1_TO, Team2_TO")
    print("  - Team1_FTA, Team2_FTA")
    print("  - Team1_OR, Team2_OR")
    print("  - Loc (H/A/N for home/away/neutral)")
    print()
    
    return None

def test_kenpom_wls(games_df, shrink_k=10):
    """
    Test the KenPomWLS implementation.
    
    Args:
        games_df: DataFrame with game data
        shrink_k: Shrinkage parameter (default 10)
    
    Returns:
        DataFrame with adjusted offensive/defensive ratings
    """
    print(f"Testing KenPomWLS with shrink_k={shrink_k}")
    print(f"Processing {len(games_df)} games...")
    
    # Initialize the KenPomWLS feature
    kenpom_wls = KenPomWLS(games_df, shrink_k=shrink_k)
    
    # Process and get results
    ratings_df, details = kenpom_wls.process(return_detailed=True)
    
    print(f"\n✓ Processed {len(ratings_df)} teams")
    print(f"✓ Home court advantage: {details['home_effect_per100']:.2f} points per 100 possessions")
    print(f"✓ League mean OffRtg: {details['league_mean']:.2f}")
    
    # Show top 10 teams by adjusted margin
    print("\nTop 10 teams by AdjMargin:")
    top_teams = ratings_df.sort_values('AdjMargin', ascending=False).head(10)
    print(top_teams[['TeamID', 'AdjO', 'AdjD', 'AdjMargin', 'games_played']].to_string(index=False))
    
    return ratings_df

def compare_with_original_efficiency(games_df):
    """
    Compare KenPomWLS with the original Efficiency implementation.
    
    This helps you see the differences and decide which to use in your models.
    """
    from kaggle_prediction_library.feature_engineering import Efficiency
    
    print("\n" + "="*60)
    print("Comparing KenPomWLS vs Original Efficiency")
    print("="*60)
    
    # Original efficiency
    print("\n1. Original Efficiency (rolling average approach)...")
    original = Efficiency(games_df, away_bonus=0.014)
    original_ratings = original.process()
    
    # KenPom WLS
    print("\n2. KenPomWLS (weighted least squares approach)...")
    kenpom_wls = KenPomWLS(games_df, shrink_k=10)
    wls_ratings = kenpom_wls.process()
    
    # Merge for comparison
    comparison = original_ratings.merge(
        wls_ratings[['TeamID', 'Season', 'AdjO', 'AdjD', 'AdjMargin']], 
        on=['TeamID', 'Season'],
        suffixes=('_orig', '_wls')
    )
    
    print(f"\n✓ Compared {len(comparison)} teams")
    print("\nCorrelations between methods:")
    print(f"  AdjO correlation:  {comparison[['adj_oe', 'AdjO']].corr().iloc[0,1]:.4f}")
    print(f"  AdjD correlation:  {comparison[['adj_de', 'AdjD']].corr().iloc[0,1]:.4f}")
    print(f"  Margin correlation: {comparison[['adj_margin', 'AdjMargin']].corr().iloc[0,1]:.4f}")
    
    print("\nSample comparison (top 5 teams by original margin):")
    sample = comparison.sort_values('adj_margin', ascending=False).head(5)
    print(sample[['TeamID', 'adj_oe', 'AdjO', 'adj_de', 'AdjD', 'adj_margin', 'AdjMargin']].to_string(index=False))
    
    return comparison

def usage_example():
    """Show how to use KenPomWLS in your prediction pipeline."""
    print("\n" + "="*60)
    print("Usage Example")
    print("="*60)
    
    example_code = '''
# In your model training/prediction pipeline:

from kaggle_prediction_library.feature_engineering import KenPomWLS

# 1. Load your game data
games = pd.read_csv('path/to/MRegularSeasonDetailedResults.csv')

# 2. Initialize KenPomWLS (you can tune shrink_k)
kenpom_wls = KenPomWLS(games, shrink_k=10)

# 3. Generate the features
ratings = kenpom_wls.process()
# Returns: DataFrame with ['Season', 'TeamID', 'AdjO', 'AdjD', 'AdjMargin', 'games_played']

# 4. Add to your base prediction dataset
base_with_features = kenpom_wls.add(base_predictions)
# This adds: t1_AdjO, t1_AdjD, t1_AdjMargin, t2_AdjO, t2_AdjD, t2_AdjMargin

# 5. Use in your model
model.fit(base_with_features[['t1_AdjO', 't1_AdjD', 't2_AdjO', 't2_AdjD', ...]], y)
'''
    print(example_code)

if __name__ == "__main__":
    print("="*60)
    print("KenPomWLS Feature Test Script")
    print("="*60)
    
    # Load your actual game data here
    games = load_sample_data()
    
    if games is None:
        print("\n⚠️  No game data loaded.")
        print("\nTo test with your actual data:")
        print("1. Load your game data (e.g., MRegularSeasonDetailedResults.csv)")
        print("2. Make sure it has the required columns (see above)")
        print("3. Run: ratings = test_kenpom_wls(games)")
        print("\nTo compare with original Efficiency:")
        print("4. Run: comparison = compare_with_original_efficiency(games)")
    else:
        # Run tests
        ratings = test_kenpom_wls(games)
        comparison = compare_with_original_efficiency(games)
        
        print("\n✓ All tests completed!")
        print("\nNext steps:")
        print("1. Integrate KenPomWLS into your feature pipeline")
        print("2. Train models with both old and new features")
        print("3. Compare model performance to decide which to use")
    
    # Show usage example
    usage_example()
    
    print("\n" + "="*60)
    print("Test script complete!")
    print("="*60)
