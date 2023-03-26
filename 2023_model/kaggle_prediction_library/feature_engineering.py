import pandas as pd
import numpy as np
import re

import abc

class FeatureEng:

    def process(self):
        """Process the data to create feature(s)"""
        raise NotImplementedError
    
    def add(self, base):
        """Add features to another dataset"""

        features = self.process()

        cols = [col for col in features.columns if col not in ['Season', 'TeamID']]

        features_team1 = features.rename(columns = {f: 't1_' + f for f in cols})
        features_team1 = features_team1.rename(columns = {'TeamID': 'Team1'})
        features_team2 = features.rename(columns ={f: 't2_' + f for f in cols})
        features_team2 = features_team2.rename(columns = {'TeamID': 'Team2'})
        
        base = base.merge(features_team1, on = ['Team1', 'Season'], how = 'left')
        base = base.merge(features_team2, on = ['Team2', 'Season'], how = 'left')
    
        return base
    
class PreSeasonAPRankings(FeatureEng):
    
    def __init__(self, rankings_df):
        self.rankings_df = rankings_df

    def process(self):
        ap_rankings = self.rankings_df[(self.rankings_df.SystemName == 'AP')]
        first_day = ap_rankings.groupby('Season').agg({'RankingDayNum':'min'}).rename(
                                    columns = {'RankingDayNum': 'first_day'})

        ap_rankings = ap_rankings.join(first_day, on = 'Season')
        ap_rankings = ap_rankings[ap_rankings.RankingDayNum == ap_rankings.first_day]

        return ap_rankings[['Season', 'TeamID', 'OrdinalRank']]
    
    def add(self, base):
        base = super().add(base)
        base['t1_OrdinalRank'] = base['t1_OrdinalRank'].fillna(25) 
        base['t2_OrdinalRank'] = base['t2_OrdinalRank'].fillna(25) 
        return base
                
class TournamentSeed(FeatureEng):
    
    def __init__(self, tourney_seeds):
        self.tourney_seeds = tourney_seeds

    def process(self):
        df = self.tourney_seeds.copy()
        df['Seed'] = self.tourney_seeds['Seed'].apply(lambda x: re.sub('[^0-9]','', x)).apply(int)
        return df[['Season', 'TeamID', 'Seed']]
    
    def add(self, base):
        base = super().add(base)
        base['seed_diff'] = base['t1_Seed'] - base['t2_Seed'] 
        return base

    
class Efficiency(FeatureEng):

    def __init__(self, games, away_bonus):
        self.games = games
        self.away_bonus = away_bonus

    def get_ratings(df):
        #Get possessions
        df['Pos'] = df.apply(lambda row: 0.96*(row.Team1_FGA + row.Team1_TO + 0.44*row.Team1_FTA - row.Team1_OR), axis=1)
        #Offensive efficiency (OffRtg) = 100 x (Points / Possessions)
        df['OffRtg'] = df.apply(lambda row: 100 * (row.Team1_score / row.Pos), axis=1)
        #Defensive efficiency (DefRtg) = 100 x (Opponent points / Opponent possessions)
        df['DefRtg'] = df.apply(lambda row: 100 * (row.Team2_score / row.Pos), axis=1)
        df.drop('Pos', axis = 1)
        return df

    def location_adjustment(self, all_games):

        all_games['OffRtg'] = np.where(all_games['Loc'] == 'H', all_games['OffRtg'] * (1 - self.away_bonus),
            np.where(all_games['Loc'] == 'A', all_games['OffRtg'] * (1 + self.away_bonus),
                    all_games['OffRtg']))
        
        all_games['DefRtg'] = np.where(all_games['Loc'] == 'H', all_games['DefRtg'] * (1 + self.away_bonus),
            np.where(all_games['Loc'] == 'A', all_games['DefRtg'] * (1 - self.away_bonus),
                    all_games['DefRtg'])) 
        
        return all_games
    
    def shifted_expanding_mean(df, groupby_cols, agg_col):
        return df.groupby(groupby_cols)[agg_col].transform(lambda x: x.shift(1).expanding().mean())


    def process(self):

        all_games = self.games.copy()

        all_games = Efficiency.get_ratings(all_games)

        all_games = self.location_adjustment(all_games)

        #sort values for rolling
        all_games.sort_values(by = ['Season', 'Team1', 'DayNum'], inplace = True)
        all_games.reset_index(drop=True, inplace = True)

        all_games['avg_oe'] = Efficiency.shifted_expanding_mean(all_games, ['Season', 'Team1'], 'OffRtg')
        all_games['avg_de'] = Efficiency.shifted_expanding_mean(all_games, ['Season', 'Team1'], 'DefRtg')

        #get opponents rolling averages "at that point in the season"
        all_games2 = all_games.rename(columns = {'Team1': 'Team2', 'Team2': 'Team1',
                                                'avg_oe': 'opp_avg_oe', 'avg_de':'opp_avg_de'})
        join_key = ['Team2', 'Season', 'DayNum']
        all_games3 = all_games.merge(all_games2[join_key + ['opp_avg_oe', 'opp_avg_de']], on = join_key, how = 'left')

        #get league's rolling averages "at that point in the season"
        all_games3.sort_values(by = ['Season', 'DayNum'], inplace = True)

        all_games3['league_avg_oe'] = Efficiency.shifted_expanding_mean(all_games3, ['Season'], 'OffRtg')
        all_games3['league_avg_de'] = Efficiency.shifted_expanding_mean(all_games3, ['Season'], 'DefRtg')

        #adjust oe and de based on opponents 
        all_games3.sort_values(by = ['Season', 'Team1', 'DayNum'], inplace = True)
        all_games3['adj_oe'] = (1 - (all_games3['opp_avg_de']/all_games3['league_avg_de'] - 1) ) * all_games3['OffRtg']
        all_games3['adj_de'] = (1 - (all_games3['opp_avg_oe']/all_games3['league_avg_oe'] - 1) ) * all_games3['DefRtg']
        
        #aggregate to Season / Team Level
        final = all_games3.groupby(['Season', 'Team1']).agg(adj_oe=('adj_oe', np.mean), adj_de=('adj_de', np.mean)).reset_index()
        final.columns = ['Season', 'TeamID', 'adj_oe', 'adj_de']

        final['adj_margin'] = final['adj_oe'] - final['adj_de']

        return final

class FinalRanking(FeatureEng):

    def __init__(self, rankings_df, system):
        self.rankings_df = rankings_df
        self.system = system

    def process(self):

        last_day = self.rankings_df.groupby('Season').agg({'RankingDayNum':'max'}).rename(
                                    columns = {'RankingDayNum': 'last_day'})


        end_rankings = self.rankings_df.join(last_day, on = 'Season')
        end_rankings = end_rankings[ (end_rankings.RankingDayNum == end_rankings.last_day)
                                    & (end_rankings.SystemName ==self.system)]
        end_rankings = end_rankings.groupby(['TeamID', 'Season']).agg({'OrdinalRank':np.mean})
        end_rankings.reset_index(inplace = True)
        end_rankings.columns = ['TeamID', 'Season', 'avg_rank']
        end_rankings['final_rank'] = 100-4*np.log(end_rankings['avg_rank']+1)-end_rankings['avg_rank']/22
        
        return end_rankings[['TeamID', 'Season', 'final_rank']]
        
class Kenpom(FeatureEng):

    def __init__(self, kp_snapshot):
        self.kp_snapshot = kp_snapshot

    def process(self):

        self.kp_snapshot['TeamID'] = self.kp_snapshot['TeamID'].astype(int)
        self.kp_snapshot['Season'] = self.kp_snapshot['Season'].astype(int)

        # deriving these two myself because kenpom switched the definitions in 2017
            # after 2017, he used the simple diff between adjo and adjd
        self.kp_snapshot['adjem'] = self.kp_snapshot['adjo'] - self.kp_snapshot['adjd']
        self.kp_snapshot['sos_adjem'] = self.kp_snapshot['sos_opp_o'] - self.kp_snapshot['sos_opp_d'] 
        
        return self.kp_snapshot[['TeamID', 'Season', 'adjem', 'adjo',
            'adjo_rank', 'adjd', 'adjd_rank', 'adjt', 'adjt_rank', 'luck',
            'luck_rank', 'sos_adjem', 'sos_adjem_rank', 'sos_opp_o',
            'sos_opp_o_rank', 'sos_opp_d', 'sos_opp_d_rank', 
            'ncsos_adjem_rank',
            #'ncsos_adjem', --> purposefully exclude this one because of shift described above
            # and because KP doesn't share the underlying inputs for this one
            # note that we do keep the rank for this one, but should be fine since its normalized
            ]]


class FiveThirtyEight(FeatureEng):
    
    def __init__(self, fivethirtyeight_df):
        self.fivethirtyeight_df = fivethirtyeight_df

    def process(self):
        
        df = self.fivethirtyeight_df.copy()
        df.rename(columns={'team_rating': 'team_rating_538'}, inplace=True)
        features = ['team_rating_538', 'rd1_win', 'rd2_win', 'rd3_win', 'rd4_win', 'rd5_win', 'rd6_win', 'rd7_win']
        df['TeamID'] = df['TeamID'].astype(int)
        df['Season'] = df['Season'].astype(int)
        
        return df[['TeamID', 'Season'] + features]
    