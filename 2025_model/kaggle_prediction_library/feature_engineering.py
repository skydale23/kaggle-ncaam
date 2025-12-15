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
        final = all_games3.groupby(['Season', 'Team1']).agg(adj_oe=('adj_oe', 'mean'), adj_de=('adj_de', 'mean')).reset_index()
        final.columns = ['Season', 'TeamID', 'adj_oe', 'adj_de']

        final['adj_margin'] = final['adj_oe'] - final['adj_de']

        if return_detailed:
            return final, all_games3

        return final


class KenPomWLS(FeatureEng):
    """
    Compute KenPom-style AdjO/AdjD for a season using Weighted Least Squares.
    
    This implementation:
    1. Uses KenPom's possessions formula: FGA - OR + TO + 0.475*FTA
    2. Averages possessions between both teams (tempo equalization)
    3. Builds a linear system: Off_team - Def_opponent + Home ≈ observed_offrtg
    4. Weights observations by possessions (more possessions = more information)
    5. Anchors solution to league mean and applies shrinkage by games played
    
    Args:
        games_df: DataFrame with game data including:
            ['Season','DayNum','Team1','Team2','Team1_score','Team2_score',
             'Team1_FGA','Team2_FGA','Team1_TO','Team2_TO','Team1_FTA','Team2_FTA',
             'Team1_OR','Team2_OR','Loc']
        shrink_k: Shrinkage parameter (default 10). Higher = more shrinkage toward league mean.
        home_advantage: Fixed home court advantage in points per 100 possessions (default None = estimate from data).
    """
    
    def __init__(self, games_df, shrink_k=10, home_advantage=None):
        self.games = games_df.copy()
        self.shrink_k = shrink_k
        self.home_advantage = home_advantage

    @staticmethod
    def kenpom_poss(df, prefix):
        """KenPom possessions per team from box score columns (vectorized)."""
        return (df[f'{prefix}_FGA'] - df[f'{prefix}_OR'] + df[f'{prefix}_TO']
                + 0.475 * df[f'{prefix}_FTA'])

    @staticmethod
    def updated_poss(df, prefix):
        """KenPom possessions per team from box score columns (vectorized)."""
        return (df[f'{prefix}_FGA'] - df[f'{prefix}_OR'] + df[f'{prefix}_TO']
                + 0.475 * df[f'{prefix}_FTA'])

    def compute_poss_and_raw(self):
        """Compute possessions and raw offensive ratings for each team."""
        g = self.games
        # per-box-score possession estimates
        g['pos_est1'] = KenPomWLS.kenpom_poss(g, 'Team1')
        g['pos_est2'] = KenPomWLS.kenpom_poss(g, 'Team2')
        # KenPom: average the two estimates and use that for both teams (tempo equalization)
        g['poss'] = 0.5 * (g['pos_est1'].fillna(0) + g['pos_est2'].fillna(0))
        # avoid zero poss
        g['poss'] = g['poss'].replace(0, np.nan)
        # observed offensive rating per 100 possessions
        g['OffRtg1'] = 100.0 * g['Team1_score'] / g['poss']
        g['OffRtg2'] = 100.0 * g['Team2_score'] / g['poss']
        self.games = g
        return g

    def build_matrix(self):
        """Build the design matrix for weighted least squares."""
        g = self.games
        teams = pd.Index(pd.concat([g['Team1'], g['Team2']]).unique())
        team_to_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)
        n_games = len(g)
        n_obs = 2 * n_games
        
        # If home_advantage is fixed, don't solve for it
        estimate_home = self.home_advantage is None
        n_params = 2 * n_teams + (1 if estimate_home else 0)  # Off_i, Def_i, optionally home_effect

        A = np.zeros((n_obs, n_params), dtype=float)
        y = np.zeros(n_obs, dtype=float)
        w = np.zeros(n_obs, dtype=float)

        row = 0
        for _, r in g.iterrows():
            i1 = team_to_idx[r['Team1']]
            i2 = team_to_idx[r['Team2']]

            # Team1 observation
            A[row, i1] = 1.0                 # Off_team1
            A[row, n_teams + i2] = -1.0      # -Def_team2
            home_flag_1 = 1.0 if r['Loc'] == 'H' else (-1.0 if r['Loc'] == 'A' else 0.0)
            
            if estimate_home:
                A[row, -1] = home_flag_1
                y[row] = r['OffRtg1']
            else:
                # Subtract fixed home advantage from target
                y[row] = r['OffRtg1'] - home_flag_1 * self.home_advantage
            
            w[row] = max(r['poss'] / 100.0, 1e-6)
            row += 1

            # Team2 observation
            A[row, i2] = 1.0
            A[row, n_teams + i1] = -1.0
            home_flag_2 = 1.0 if r['Loc'] == 'A' else (-1.0 if r['Loc'] == 'H' else 0.0)
            
            if estimate_home:
                A[row, -1] = home_flag_2
                y[row] = r['OffRtg2']
            else:
                # Subtract fixed home advantage from target
                y[row] = r['OffRtg2'] - home_flag_2 * self.home_advantage
            
            w[row] = max(r['poss'] / 100.0, 1e-6)
            row += 1

        self.design = {'A': A, 'y': y, 'w': w, 'teams': teams, 'estimate_home': estimate_home}
        return self.design

    def solve_wls(self):
        """Solve the weighted least squares system."""
        A = self.design['A']
        y = self.design['y']
        w = self.design['w']
        estimate_home = self.design['estimate_home']
        W_sqrt = np.sqrt(w)[:, None]
        Aw = W_sqrt * A
        yw = (W_sqrt[:, 0] * y)
        # Solve weighted least squares
        x, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        teams = self.design['teams']
        n_teams = len(teams)
        offs = x[:n_teams]
        defs = x[n_teams:2*n_teams]
        
        # Home effect: either estimated from data or user-specified
        if estimate_home:
            home_effect = x[-1]
        else:
            home_effect = self.home_advantage

        # Anchor to league mean
        g = self.games
        raw_offs = np.concatenate([g['OffRtg1'].dropna().to_numpy(), g['OffRtg2'].dropna().to_numpy()])
        league_mean = float(np.nanmean(raw_offs))
        
        # Shift offensive parameters so their mean equals observed league mean
        off_shift = league_mean - offs.mean()
        offs += off_shift
        
        # The defensive parameters from the model represent "defensive strength"
        # where higher values mean better defense (more points prevented).
        # But KenPom convention is "points allowed" where LOWER is better.
        # So we need to NEGATE the defensive parameters to convert them.
        # After negation, shift so mean equals league mean.
        defs = -defs  # Flip sign: now higher defs = worse defense (more points allowed)
        def_shift = league_mean - defs.mean()
        defs += def_shift

        # shrink toward league mean by games played
        teams_series = teams.to_series(index=teams)
        # games played per team
        gp = pd.Series(0, index=teams)
        gp = gp.add(g.groupby('Team1').size(), fill_value=0).add(g.groupby('Team2').size(), fill_value=0)

        k = self.shrink_k
        shrink_w = gp.values / (gp.values + k)
        offs_shrunk = shrink_w * offs + (1 - shrink_w) * league_mean
        defs_shrunk = shrink_w * defs + (1 - shrink_w) * league_mean

        final = pd.DataFrame({
            'TeamID': teams,
            'AdjO': offs_shrunk,
            'AdjD': defs_shrunk,
            'games_played': gp.values
        })
        final['AdjMargin'] = final['AdjO'] - final['AdjD']
        
        # Season will be added by process() method

        self.result = {'final': final, 'home_effect_per100': home_effect, 'league_mean': league_mean}
        return self.result

    def process(self, return_detailed=False):
        """
        Process the games and compute KenPom-style ratings.
        
        Returns:
            DataFrame with columns: ['Season', 'TeamID', 'AdjO_k{shrink}_h{home}', 'AdjD_k{shrink}_h{home}', 
                                     'AdjMargin_k{shrink}_h{home}', 'games_played_k{shrink}_h{home}']
            where {shrink} is the shrink_k value and {home} is either the home_advantage value or 'Est'
            If return_detailed=True, also returns dict with 'home_effect_per100' and 'league_mean'
        """
        # Check if we have multiple seasons - if so, process each separately
        if 'Season' in self.games.columns and self.games['Season'].nunique() > 1:
            all_results = []
            details_by_season = {}
            
            for season in sorted(self.games['Season'].unique()):
                # Create temporary instance for this season
                season_games = self.games[self.games['Season'] == season].copy()
                season_wls = KenPomWLS(season_games, shrink_k=self.shrink_k, home_advantage=self.home_advantage)
                
                # Process this season
                season_wls.compute_poss_and_raw()
                season_wls.build_matrix()
                season_result = season_wls.solve_wls()
                
                # Add season column
                season_result['final']['Season'] = season
                # Add shrink and home advantage suffix to column names
                home_suffix = f"h{self.home_advantage}" if self.home_advantage is not None else "hEst"
                suffix = f"k{self.shrink_k}_{home_suffix}"
                season_final = season_result['final'].copy()
                season_final = season_final.rename(
                    columns={
                        'AdjO': f'AdjO_{suffix}',
                        'AdjD': f'AdjD_{suffix}',
                        'AdjMargin': f'AdjMargin_{suffix}',
                        'games_played': f'games_played_{suffix}',
                    }
                )
                all_results.append(season_final)
                
                if return_detailed:
                    details_by_season[season] = {
                        'home_effect_per100': season_result['home_effect_per100'],
                        'league_mean': season_result['league_mean']
                    }
            
            final_df = pd.concat(all_results, ignore_index=True)
            
            if return_detailed:
                return final_df, details_by_season
            return final_df
        
        else:
            # Single season processing (original logic)
            self.compute_poss_and_raw()
            self.build_matrix()
            result = self.solve_wls()
            
            # Add Season column if present in data
            if 'Season' in self.games.columns:
                result['final']['Season'] = self.games['Season'].iloc[0]
            
            # Add shrink and home advantage suffix to column names
            home_suffix = f"h{self.home_advantage}" if self.home_advantage is not None else "hEst"
            suffix = f"k{self.shrink_k}_{home_suffix}"
            final_df = result['final'].copy().rename(
                columns={
                    'AdjO': f'AdjO_{suffix}',
                    'AdjD': f'AdjD_{suffix}',
                    'AdjMargin': f'AdjMargin_{suffix}',
                    'games_played': f'games_played_{suffix}',
                }
            )
            
            if return_detailed:
                return final_df, result
            
            return final_df


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
        end_rankings = end_rankings.groupby(['TeamID', 'Season']).agg({'OrdinalRank':'mean'})
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

        season_statistics = df.groupby(["Season", 'Team1'])[boxscore_cols].agg('mean').reset_index()
        season_statistics.columns = ["Season", 'TeamID'] + [i[6:] for i in boxscore_cols]

        return season_statistics


class AdvancedTeamStats(FeatureEng):

    def __init__(self, games):
        self.games = games

    def process(self):

        df = self.games.copy()

        # possessions using standard formula per team
        df["Team1_poss"] = (
            df["Team1_FGA"]
            + 0.475 * df["Team1_FTA"]
            - df["Team1_OR"]
            + df["Team1_TO"]
        )
        df["Team2_poss"] = (
            df["Team2_FGA"]
            + 0.475 * df["Team2_FTA"]
            - df["Team2_OR"]
            + df["Team2_TO"]
        )

        # offensive four-factor style stats for Team1 perspective
        df["Team1_ORB"] = df["Team1_OR"]
        df["Team1_TOV"] = df["Team1_TO"]
        df["Team1_FTr"] = df["Team1_FTA"] / df["Team1_FGA"].replace(0, np.nan)
        df["Team1_3PA_rate"] = df["Team1_FGA3"] / df["Team1_FGA"].replace(0, np.nan)

        # shooting percentages
        df["Team1_FG%"] = df["Team1_FGM"] / df["Team1_FGA"].replace(0, np.nan)
        df["Team1_3P%"] = df["Team1_FGM3"] / df["Team1_FGA3"].replace(0, np.nan)

        # pace proxy (possessions per game for Team1)
        df["Team1_pace"] = df["Team1_poss"]

        # aggregate to season / team level from Team1 perspective
        agg_cols = [
            "Team1_pace",
            "Team1_ORB",
            "Team1_TOV",
            "Team1_FTr",
            "Team1_3PA_rate",
            "Team1_FG%",
            "Team1_3P%",
        ]

        season_stats = (
            df.groupby(["Season", "Team1"])[agg_cols]
            .mean()
            .reset_index()
        )

        season_stats.rename(
            columns={
                "Team1": "TeamID",
                "Team1_pace": "pace",
                "Team1_ORB": "ORB",
                "Team1_TOV": "TOV",
                "Team1_FTr": "FTr",
                "Team1_3PA_rate": "3PA_rate",
                "Team1_FG%": "FG_pct",
                "Team1_3P%": "3P_pct",
            },
            inplace=True,
        )

        return season_stats

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
        

# Next step is to rewrite this so that the calculate_team_season_stats part only calculates the metrics I need:

# Work is in progress

# 't1_top8_BPM_weighted_mean', 't2_top8_BPM_weighted_mean',
#     't1_top8_TO_stdev', 't2_top8_TO_stdev',
#     't1_top5_PRPG!_median', 't2_top5_PRPG!_median',
#     't1_top3_DR_median', 't2_top3_DR_median',
#     't1_top5_STL_cv', 't2_top5_STL_cv',
#     't1_top3_Min%_median', 't2_top3_Min%_median',
#     't1_top8_TS_gini', 't2_top8_TS_gini',
#     't1_top3_USG_gini', 't2_top3_USG_gini',
#       
class AggregatedPlayerStats(FeatureEng):

    def __init__(self, data):
        self.data = data.copy()

    def gini_coefficient(self, x):
        x = np.sort(x)  # Sort values
        n = len(x)
        cum_x = np.cumsum(x)
        return (2 * np.sum(np.arange(1, n + 1) * x) - (n + 1) * cum_x[-1]) / (n * cum_x[-1])

    def calculate_team_season_stats(self, df, team_column):
        # Ensure numeric types for "Min%" and the required columns
        required_columns = ["TO", "PRPG!", "DR", "STL", "Min%", "TS", "USG"]
        df["Min%"] = df["Min%"].astype(float)
        for col in required_columns:
            df[col] = df[col].astype(float)
        
        grouped_stats = []
        
        # Process each team-season group
        for (team, season), group in df.groupby([team_column, "Season"]):
            stats = {team_column: team, "Season": season}
            # Sort the group by "Min%" in descending order once
            group = group.sort_values("Min%", ascending=False)
            
            # For column TO: top 8 players stdev
            top8 = group.head(8)
            values = top8["TO"].dropna()
            if len(values) > 0:
                stats["top8_TO_stdev"] = values.std(ddof=0)
            else:
                stats["top8_TO_stdev"] = np.nan
            
            # For column PRPG!: top 5 players median
            top5 = group.head(5)
            values = top5["PRPG!"].dropna()
            if len(values) > 0:
                stats["top5_PRPG!_median"] = values.median()
            else:
                stats["top5_PRPG!_median"] = np.nan
            
            # For column DR: top 3 players median
            top3 = group.head(3)
            values = top3["DR"].dropna()
            if len(values) > 0:
                stats["top3_DR_median"] = values.median()
            else:
                stats["top3_DR_median"] = np.nan
            
            # For column STL: top 5 players coefficient of variation (cv)
            top5 = group.head(5)
            values = top5["STL"].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                stdev_val = values.std(ddof=0)
                stats["top5_STL_cv"] = stdev_val / mean_val if mean_val != 0 else np.nan
            else:
                stats["top5_STL_cv"] = np.nan
            
            # For column Min%: top 3 players median
            top3 = group.head(3)
            values = top3["Min%"].dropna()
            if len(values) > 0:
                stats["top3_Min%_median"] = values.median()
            else:
                stats["top3_Min%_median"] = np.nan
            
            # For column TS: top 8 players gini coefficient
            top8 = group.head(8)
            values = top8["TS"].dropna()
            if len(values) > 0:
                stats["top8_TS_gini"] = self.gini_coefficient(values)
            else:
                stats["top8_TS_gini"] = np.nan
            
            # For column USG: top 3 players gini coefficient
            top3 = group.head(3)
            values = top3["USG"].dropna()
            if len(values) > 0:
                stats["top3_USG_gini"] = self.gini_coefficient(values)
            else:
                stats["top3_USG_gini"] = np.nan

            top8 = group.head(8)
            mask = top8["BPM"].notna()
            if mask.sum() > 0:
                weighted_mean = np.average(top8.loc[mask, "BPM"], weights=top8.loc[mask, "Min%"])
            else:
                weighted_mean = np.nan
            stats["top8_BPM_weighted_mean"] = weighted_mean
            
            grouped_stats.append(stats)
        
        return pd.DataFrame(grouped_stats)


    def process(self):
        # Needs updating to call calculate_team_season_stats and then grab the columns needed
        # Maybe if I just use TeamID as the team column I don't need to worry (like below)

        data = self.data.copy()

        result = self.calculate_team_season_stats(data, "TeamID")

        return result 