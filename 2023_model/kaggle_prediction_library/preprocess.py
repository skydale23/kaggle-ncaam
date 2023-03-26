import pandas as pd
import numpy as np

class PreProcess():

    ''' Unify and pre-process data from Kaggle regular reason and tournament datasets '''

    win_cols = ['TourneyGame','Season', 'DayNum', 'WTeamID','LTeamID', 'WScore', 'LScore', 'WLoc', 
    'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
    'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
    'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

    lose_cols = ['TourneyGame','Season', 'DayNum', 'LTeamID', 'WTeamID', 'LScore', 'WScore', 'WLoc',
    'NumOT', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3','LFTM', 'LFTA', 'LOR', 'LDR', 
    'LAst', 'LTO', 'LStl', 'LBlk', 'LPF','WFGM', 'WFGA', 'WFGM3', 'WFGA3', 
    'WFTM', 'WFTA', 'WOR', 'WDR','WAst', 'WTO', 'WStl', 'WBlk', 'WPF']

    new_cols = ['TourneyGame','Season', 'DayNum', 'Team1', 'Team2', 'Team1_score', 'Team2_score', 'WLoc',
    'NumOT', 'Team1_FGM', 'Team1_FGA', 'Team1_FGM3', 'Team1_FGA3', 'Team1_FTM', 'Team1_FTA', 'Team1_OR', 'Team1_DR',
    'Team1_Ast', 'Team1_TO', 'Team1_Stl', 'Team1_Blk', 'Team1_PF', 'Team2_FGM', 'Team2_FGA', 'Team2_FGM3', 'Team2_FGA3',
    'Team2_FTM', 'Team2_FTA', 'Team2_OR', 'Team2_DR', 'Team2_Ast', 'Team2_TO', 'Team2_Stl', 'Team2_Blk', 'Team2_PF']


    def __init__(self, tournament_data, regular_season_data):
        self.tournament_data = tournament_data
        self.regular_season_data = regular_season_data
    
    def process(self):

        self.tournament_data['TourneyGame'] = 1
        self.regular_season_data['TourneyGame'] = 0
        df = pd.concat([self.tournament_data, self.regular_season_data])
        
        #Remap column names
        df_winners = df[PreProcess.win_cols].rename(columns={i:j for (i,j) in zip(
                            PreProcess.win_cols, PreProcess.new_cols)}).copy()
        df_losers = df[PreProcess.lose_cols].rename(columns={i:j for (i,j) in zip(
                            PreProcess.lose_cols, PreProcess.new_cols)}).copy()
        
        #Add column to remember who won/lost
        df_winners['Outcome'] = 1
        df_losers['Outcome'] = 0
        
        #Encode location when Team1 is the winner
        df_winners['Loc'] = np.where(df_winners['WLoc'] == 'H', 'H', \
                                    np.where(df_winners['WLoc'] == 'A', 'A', 'N'))
        
        #Encode location when Team1 is the loser
        df_losers['Loc'] = np.where(df_losers['WLoc'] == 'H', 'A', \
                                    np.where(df_losers['WLoc'] == 'A', 'H', 'N'))
        
        #Create full df
        full = df_winners.append(df_losers)

        return full
            

    