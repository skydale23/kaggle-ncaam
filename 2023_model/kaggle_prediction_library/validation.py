import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.feature_selection import chi2
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss


def calculate_confidence_interval(df, column_name):
    """
    Calculate the mean and 95% confidence interval for a column in a pandas DataFrame.

    Parameters:
    - df: pandas DataFrame
    - column_name: str, name of the column containing the scores

    Returns:
    - mean: float, mean of the scores
    - confidence_interval: tuple, (lower_bound, upper_bound)
    """
    # Extract the scores as a numpy array
    scores = df[column_name].values

    # Calculate the mean and standard deviation of the scores
    mean_score = np.mean(scores)
    std_dev = np.std(scores, ddof=1)  # ddof=1 for sample standard deviation

    # Calculate the standard error of the mean
    sem = std_dev / np.sqrt(len(scores))

    # Calculate the margin of error for a 95% confidence interval
    margin_of_error = 1.96 * sem  # 1.96 corresponds to the z-score for a 95% confidence interval

    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = mean_score - margin_of_error
    upper_bound = mean_score + margin_of_error

    return mean_score, (lower_bound, upper_bound)

def repeated_kfold(df, model, features, target, param_grid, n_splits=10, 
                   n_repeats=25, scoring='neg_brier_score'):

    # Define the cross-validation strategy
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

    # Create the GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv)

    # Fit the model with the grid search
    grid_search.fit(df[features], df[target])

    return grid_search

def get_confidence_interval_grid_search(grid_search, total_splits):
    rows = []
    for i in range(total_splits):
        row = grid_search.cv_results_[f'split{i}_test_score'][grid_search.best_index_]
        rows.append(row)

    tmp = pd.DataFrame(rows, columns = ["best_scores"])

    mean_score, confidence_interval = calculate_confidence_interval(tmp, "best_scores")

    return mean_score, confidence_interval

def rolling_season_cv(model, train_input, features, label='Outcome', cv_start=2007, return_preds=False):
    ''' returns estimate for model performance using shifted validation'''
    
    scores = []

    preds_dfs = []

    for n, season in enumerate(train_input.Season.unique()):
        
        if season >= cv_start:

            train = train_input[train_input.Season < season].copy()
            test = train_input[train_input.Season == season].copy()

            X_train = train[features].copy()
            X_test = test[features].copy()
            y_train = train[label].copy()
            y_test = test[label].copy()

            model.fit(X_train[features], y_train)
            y_prob = model.predict_proba(X_test[features])

            test["y_prob"] = y_prob[:,1].copy()
            test["y_true"] = y_test.copy()
            preds_dfs.append(test)

            loss = brier_score_loss(y_test, y_prob[:,1])
            scores.append((season, loss))

    validation_df = pd.DataFrame(scores, columns = ['season', 'score']).sort_values(by = 'score')
    avg_validation_score = validation_df.score.mean()

    all_preds = pd.concat(preds_dfs,axis =0)

    if return_preds:
        return avg_validation_score, all_preds
    else:
        return avg_validation_score

# This is the final function to use for cv / eval
def run_evaluation_framework(df, model, features, param_grid, target="Outcome",
                             repeated_kfold_n_splits=10, repeated_kfold_n_repeats=25):

    grid_search = repeated_kfold(df, model, features, target, param_grid, n_splits=repeated_kfold_n_splits, 
                    n_repeats=repeated_kfold_n_repeats, scoring='neg_brier_score')

    mean_score, confidence_interval = get_confidence_interval_grid_search(grid_search, 
                                                            total_splits=repeated_kfold_n_splits*repeated_kfold_n_repeats)

    model.set_params(**grid_search.best_params_)

    rolling_season_avg = rolling_season_cv(model, df, features, label=target, cv_start=2007)

    eval_df = pd.DataFrame(
                [[grid_search.best_params_, mean_score, confidence_interval, rolling_season_avg]],
                columns = ["best_params", 
                        "mean_repeated_cv_score", 
                        "repeated_cv_confidence_interval",
                        "rolling_season_cv"])
    return eval_df
