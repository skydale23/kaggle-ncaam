import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
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


    def process(self, return_detailed=False):

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

        if return_detailed:
            return final, all_games3

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
    

class SeasonStats(FeatureEng):

    def __init__(self, games):
        self.games = games

    def process(self):
        
        df = self.games.copy()

        df["Team1_PointDiff"] = df["Team1_score"] - df["Team2_score"]

        boxscore_cols = [
                'Team1_FGM', 'Team1_FGA', 'Team1_FGM3', 'Team1_FGA3', 'Team1_OR', 'Team1_Ast', 'Team1_TO', 
                'Team1_Stl', 'Team1_PF', 'Team1_FTA', 'Team1_FTM',  
                'Team1_PointDiff']

        season_statistics = df.groupby(["Season", 'Team1'])[boxscore_cols].agg(np.mean).reset_index()
        season_statistics.columns = ["Season", 'TeamID'] + [i[6:] for i in boxscore_cols]

        return season_statistics

class TeamQuality(FeatureEng):

    def __init__(self, games, seeds):

        df = games.copy()
        df["Team1"] = df["Team1"].astype(str).copy()
        df["Team2"] = df["Team2"].astype(str).copy()

        march_madness = pd.merge(seeds[['Season','TeamID']],seeds[['Season','TeamID']], on='Season')
        march_madness.columns = ['Season', 'Team1', 'Team2']
        march_madness.Team1 = march_madness.Team1.astype(str)
        march_madness.Team2 = march_madness.Team2.astype(str)
        df = pd.merge(df, march_madness, on = ['Season','Team1','Team2'])

        self.games = df

    def get_team_quality(self, games, season):
        formula = 'Outcome~-1+Team1+Team2'
        glm = sm.GLM.from_formula(formula=formula, 
                                data=games.loc[games.Season==season,:], 
                                family=sm.families.Binomial()).fit()
        
        quality = pd.DataFrame(glm.params).reset_index()
        quality.columns = ['TeamID','quality']
        quality['Season'] = season
        quality = quality.loc[quality.TeamID.str.contains('Team1')].reset_index(drop=True)
        quality['TeamID'] = quality['TeamID'].astype(str).apply(lambda x: x[6:10]).astype(int)
        return quality
    
    def process(self):
        
        games = self.games.copy()
        team_quality_stats = pd.concat([self.get_team_quality(games, s) 
                                        for s in games.Season.unique()], axis=0)
        return team_quality_stats


class RoundNumber(FeatureEng):

    def __init__(self, seeds, seed_round):

        self.seeds = seeds.copy()
        self.seed_round = seed_round.copy()
       
    def process(self):
        """Process the data to create feature(s)"""
        
        tmp2 = self.seeds.merge(self.seed_round, how="left", on="Seed")

        rename_cols = ['Season', 'Seed', 'TeamID', 'GameRound', 'GameSlot', 'EarlyDayNum', 'LateDayNum']

        tmp3=tmp2.copy()
        tmp3.columns = ["Team1_"+col if col in rename_cols else col for col in tmp2.columns]

        tmp4=tmp2.copy()
        tmp4.columns = ["Team2_"+col if col in rename_cols else col for col in tmp2.columns]

        tmp5 = tmp3.merge(tmp4, how="left",
                        left_on = ['Team1_Season', 'Team1_GameSlot'],
                        right_on = ['Team2_Season', 'Team2_GameSlot'])

        # Sort the DataFrame by 'cola', 'colb', and 'colc'
        tmp5 = tmp5.sort_values(by=['Team1_TeamID', 'Team2_TeamID', 'Team1_GameRound'])

        # Create a row number column within each group
        tmp5['row_number'] = tmp5.groupby(['Team1_Season', 'Team1_TeamID', 'Team2_TeamID']).cumcount() + 1

        tmp6 = tmp5[tmp5.row_number == 1]

        final = tmp6[tmp6.Team1_TeamID != tmp6.Team2_TeamID].copy()

        final.rename(columns = {"Team1_TeamID":"Team1", 
                                "Team2_TeamID":"Team2", 
                                "Team1_Season":"Season", 
                                "Team1_GameRound": "GameRound"}, inplace=True)

        return final
    
    def add(self, base):
        """Add features to another dataset"""

        features = self.process()
        join_key = ["Team1", "Team2", "Season"]
        base = base.merge(features[["GameRound"] + join_key], how="left", on=join_key)
    
        return base

class FirstRoundOpponentQuality(FeatureEng):

    def __init__(self, first_round_df, other_rounds_df):

        self.first_round_df = first_round_df.copy()
        self.other_rounds_df = other_rounds_df.copy()
       
    def process(self):
        """Process the data to create feature(s)"""

        first_round_opp = self.first_round_df.rename(columns = {"t2_final_rank":"round1_opponent_rank"})
    
        tmp = self.other_rounds_df.merge(first_round_opp[["Season", "Team1", "round1_opponent_rank"]], how = "left", on = ["Season", "Team1"])

        #tmp2 = tmp[~tmp.round1_opponent_rank.isna()].copy()

        tmp["round1_opponent_quality"] = (tmp["round1_opponent_rank"] - tmp["round1_opponent_rank"].min()) / (tmp["round1_opponent_rank"].max() - tmp["round1_opponent_rank"].min())
        
        return tmp
    
    def add(self):
        # returns other round data with new col
        return self.process()

class TeamNames(FeatureEng):

    def __init__(self, team_names):
        self.team_names = team_names.copy()

    def process(self):
        return self.team_names
    
    def add(self, base):
        """Add features to another dataset"""

        features = self.process()

        cols = [col for col in features.columns if col not in ['Season', 'TeamID']]

        features_team1 = features.rename(columns = {f: 't1_' + f for f in cols})
        features_team1 = features_team1.rename(columns = {'TeamID': 'Team1'})
        features_team2 = features.rename(columns ={f: 't2_' + f for f in cols})
        features_team2 = features_team2.rename(columns = {'TeamID': 'Team2'})
        
        base = base.merge(features_team1, on = ['Team1'], how = 'left')
        base = base.merge(features_team2, on = ['Team2'], how = 'left')
    
        return base

class FirstRoundOdds(FeatureEng):

    def __init__(self, first_round_odds_data):
        self.first_round_odds_data = first_round_odds_data.copy()

    def process(self):

        odds_data = self.first_round_odds_data.copy()

        if "Season" not in odds_data.columns:
            odds_data["Date"] = pd.to_datetime(odds_data["Date"], format='%b %d, %Y')
            # Extract the year
            odds_data['Season'] = odds_data["Date"].dt.year
    
        return odds_data


    def add(self, base):

        odds_data = self.process() 

        first_round_odds_data1 = odds_data.rename(columns={"kaggle_team": "t1_TeamName",
                                                               "odds": "odds_team1"})
        
        cols = ["t1_TeamName", "Season", "odds_team1"]
        
        base = base.merge(first_round_odds_data1[cols], how="left", on=["Season", "t1_TeamName"])

        first_round_odds_data2 = odds_data.rename(columns={"kaggle_team": "t2_TeamName",
                                                               "odds": "odds_team2"})

        cols = ["t2_TeamName", "Season", "odds_team2"]

        base = base.merge(first_round_odds_data2[cols], how="left", on=["Season", "t2_TeamName"])
        
        base["final_odds"] = np.where(base.odds_team1.isna(),
                                        base.odds_team2 * -1, 
                                        base.odds_team1)
        
        base.drop(["odds_team1", "odds_team2"], axis=1, inplace=True)
        
        return base
        
    

