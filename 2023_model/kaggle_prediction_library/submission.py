import pandas as pd
import numpy as np

class SubmissionSetup:

    def __init__(self, sub_data, mens_tourney_games, womens_tourney_games, mens_teams):

        self.sub_data=sub_data
        self.mens_tourney_games=mens_tourney_games.copy()
        self.womens_tourney_games=womens_tourney_games.copy()
        self.mens_teams=mens_teams

    def prepare_submission_games(sub, mens_teams):

        sub[['Season', 'Team1', 'Team2']] = sub.ID.str.split('_', expand=True)
        sub['Team1'] = sub['Team1'].astype(int)
        sub['Team2'] = sub['Team2'].astype(int)
        mens_teams['Team1'] = mens_teams['TeamID'].astype(int)
        mens_teams['Men'] = 1
        sub = sub.merge(mens_teams[['Team1', 'Men']], how='left', on='Team1')
        sub['Gender'] = np.where(sub.Men == 1, 'M', 'W')
        sub['Outcome'] = None
        sub['margin'] = None
        sub['type'] = 'Prediction'

        return sub[['type', 'ID', 'Pred', 'Season', 'Team1', 'Team2', 'Outcome', 'Gender', 'margin']].copy()


    def prepare_historical_games(tourney_games, gender):

        tourney_games['ID'] = tourney_games['Season'].astype(str) + '_' + \
                                    tourney_games['Team1'].astype(str) + '_' + \
                                    tourney_games['Team2'].astype(str)
        tourney_games['Pred'] = None
        tourney_games['type'] = 'Historical'
        tourney_games['Gender'] = gender

        return tourney_games[['type', 'ID', 'Pred', 'Season', 'Team1', 'Team2', 'Outcome', 'Gender', 'margin']].copy()

    def setup(self):
        sub = SubmissionSetup.prepare_submission_games(self.sub_data, self.mens_teams)
        mens_historical_games = SubmissionSetup.prepare_historical_games(self.mens_tourney_games, 'M')
        womens_historical_games = SubmissionSetup.prepare_historical_games(self.womens_tourney_games, 'W')
        sub['Season'] = sub['Season'].astype(int)
        return pd.concat([mens_historical_games, womens_historical_games, sub], axis = 0)
