import pandas as pd
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

def get_time_score_features(shots_df):
    shots_df= shots_df.copy()
    shots_df["score_difference"] = shots_df.score - shots_df.score_opp
    shots_df["clutch_time"] = np.where((shots_df.Time_Seconds<=120) & (abs(shots_df.score-shots_df.score_opp)<5 & (shots_df.Period>=2)),1,0)
    return shots_df[["shot_id","Time_Seconds","score_difference","clutch_time"]]

def create_player_level_segment_df(segments_df):
    segments_df = segments_df.copy()
    segments_df["all_players_segment"] = segments_df["players_team1"]+segments_df["players_team2"] 
    segments_df["all_players_team"] = [[1,1,1,1,1,2,2,2,2,2] for i in range(segments_df.shape[0])]
    segments_df_long = segments_df.explode(["all_players_segment","all_players_team"]).rename(columns={"all_players_segment":"player_id","all_players_team":"team"}).reset_index(drop=True)
    segments_df_long["team_id"] = np.where(segments_df_long.team==1,segments_df_long.teamid1,segments_df_long.teamid2)
    return segments_df_long

def get_player_game_stats(segments_df_long,segments_players_usage_blocking_df):

    segment_master_df = segments_df_long.merge(segments_players_usage_blocking_df,on=["segment_id","player_id"],how="inner")
    # Aggregate metrics of a player in a game across all segments  
    segment_master_df_game_player = segment_master_df.groupby(["game_id","player_id","team"],as_index=False)[["possessions_team1","possessions_team2","twoshots_team1","twoshots_team2","uses","blocks"]].sum()

    # When calculating block ratio take twoshot attempts of opponent team
    segment_master_df_game_player["two_shots_for_blocks"] = np.where(segment_master_df_game_player.team==1,\
                                                            segment_master_df_game_player.twoshots_team2,\
                                                            segment_master_df_game_player.twoshots_team1)
    #When calculating usage ratio take posessions of same team
    segment_master_df_game_player["posessions_for_usage"] = np.where(segment_master_df_game_player.team==1,\
                                                        segment_master_df_game_player.possessions_team1,\
                                                        segment_master_df_game_player.possessions_team2)  
    return segment_master_df,segment_master_df_game_player  

def pad_date(x):
    m,d,y = x.split("/")
    if int(m)<10:
        m="0"+str(int(m))
    if int(d)<10:
        d = "0"+str(int(d))
    if len(y)==2:
        y = "20" + y    
    return m+"/"+d+"/"+y

def get_corrected_score_feature(shots_df):
    shots_df=shots_df.copy()
    shots_df["two_three"] = np.where(shots_df.Stat.str.contains('Two'),"two","three")
    shots_df["whether_made"] = np.where(shots_df.Stat.str.contains('Make'),1,0)
    def cal_pre_shot_score(x):
        if (x.two_three =="two") & (x.whether_made==1):
            return x.score - 2
        elif (x.two_three =="three") & (x.whether_made==1):
            return x.score - 3
        else:
            return x.score
    shots_df['score_pre_shot'] = shots_df[['score','two_three','whether_made']].apply(lambda x:cal_pre_shot_score(x),axis=1)
    shots_df['score_pre_shot_diff'] = shots_df.score_pre_shot - shots_df.score_opp    
    del(shots_df["two_three"])
    del(shots_df["whether_made"])
    return shots_df

def get_all_team_player_game_stats(segment_master_df,segment_master_df_game_player,games_df):
    #Create master Team - Game - Player Dataframe to track which players played which game
    # Get all games of teams using segment_df
    # Get all players of a team - can be part of many different segments
    # Join 2 together on team id - we get all possible team_id x game_id x player_id  - some of these are ficticious
    
    #Get all possible team x game x player
    team_game_info = segment_master_df[["team_id","game_id"]].drop_duplicates()
    team_player_info = segment_master_df[["team_id","player_id"]].drop_duplicates()
    team_game_player_info = team_game_info.merge(team_player_info,how="left",on="team_id")

    #Add game-player stats  
    team_game_player_stats = team_game_player_info.merge(segment_master_df_game_player,on=["game_id","player_id"],how="left")
    team_game_player_stats["player_mode"] = np.where(team_game_player_stats.team.isna(),0,1)

    #Add date of game
    team_game_player_stats_df = team_game_player_stats.merge(games_df[["game_id","date"]],on="game_id",how="left")

    #Sort the dataframe by team, player and date - for each player we will be looking behind 

    team_game_player_stats_df = team_game_player_stats_df.sort_values(by=["team_id","player_id","date"],ascending=(1,1,1)).reset_index(drop=True)

    # Create features
    last_3_games_perf = team_game_player_stats_df.groupby(["team_id","player_id"],sort=False,as_index=False)[["uses","blocks","two_shots_for_blocks","posessions_for_usage"]].rolling(closed="left",window=3).sum()
    last_1_games_perf = team_game_player_stats_df.groupby(["team_id","player_id"],as_index=False)[["uses","blocks","two_shots_for_blocks","posessions_for_usage"]].rolling(closed="left",window=1).sum()
    last_5_games_perf = team_game_player_stats_df.groupby(["team_id","player_id"],sort=False,as_index=False)[["uses","blocks","two_shots_for_blocks","posessions_for_usage"]].rolling(closed="left",window=5).sum()
    last_10_games_perf = team_game_player_stats_df.groupby(["team_id","player_id"],sort=False,as_index=False)[["uses","blocks","two_shots_for_blocks","posessions_for_usage"]].rolling(closed="left",window=10).sum()
    current_season_perf = team_game_player_stats_df.groupby(["team_id","player_id"],sort=False,as_index=False)[["uses","blocks","two_shots_for_blocks","posessions_for_usage"]].expanding().sum().reset_index()
    
    #print(team_game_player_stats_df.shape,last_1_games_perf.shape,current_season_perf.shape)
    team_game_player_stats_df["last_1_game_usage_ratio"] = (last_1_games_perf.uses/last_1_games_perf.posessions_for_usage)
    team_game_player_stats_df["last_3_game_usage_ratio"] = (last_3_games_perf.uses/last_3_games_perf.posessions_for_usage)
    team_game_player_stats_df["last_5_game_usage_ratio"] = (last_5_games_perf.uses/last_5_games_perf.posessions_for_usage)
    team_game_player_stats_df["last_10_game_usage_ratio"] = (last_10_games_perf.uses/last_10_games_perf.posessions_for_usage)
    team_game_player_stats_df["current_season_usage_ratio"] = (current_season_perf.uses-team_game_player_stats_df.uses)/(current_season_perf.posessions_for_usage-team_game_player_stats_df.posessions_for_usage)

    team_game_player_stats_df["last_1_game_blocks_ratio"] = (last_1_games_perf.blocks/last_1_games_perf.two_shots_for_blocks)
    team_game_player_stats_df["last_3_game_blocks_ratio"] = (last_3_games_perf.blocks/last_3_games_perf.two_shots_for_blocks)
    team_game_player_stats_df["last_5_game_blocks_ratio"] = (last_5_games_perf.blocks/last_5_games_perf.two_shots_for_blocks)
    team_game_player_stats_df["last_10_game_blocks_ratio"] = (last_10_games_perf.blocks/last_10_games_perf.two_shots_for_blocks)
    team_game_player_stats_df["current_season_blocks_ratio"] = (current_season_perf.blocks-team_game_player_stats_df.blocks)/(current_season_perf.two_shots_for_blocks-team_game_player_stats_df.two_shots_for_blocks)

    return team_game_player_stats_df    

def get_all_team_player_stats_last_season(segment_master_df_game_player):

    past_season_perf = segment_master_df_game_player.groupby(["player_id"],sort=False,as_index=False)[["uses","blocks","two_shots_for_blocks","posessions_for_usage"]].sum().reset_index(drop=True)
    past_season_perf["past_season_usage_ratio"] = (past_season_perf.uses/past_season_perf.posessions_for_usage)
    return past_season_perf[["player_id","past_season_usage_ratio"]]

def get_segment_team_level_blockers(team_game_player_stats_df,segments_df_long,choose_threshold=False):
    segment_df_long_stats = segments_df_long.merge(team_game_player_stats_df,on=["team_id","game_id","player_id"],how="left")
    
    if choose_threshold:
        print((segment_df_long_stats.last_1_game_blocks_ratio.quantile(.75),
    segment_df_long_stats.last_1_game_blocks_ratio.quantile(.8),
    segment_df_long_stats.last_1_game_blocks_ratio.quantile(.9),
    segment_df_long_stats.last_1_game_blocks_ratio.quantile(.95)))
        print((segment_df_long_stats.last_3_game_blocks_ratio.quantile(.75),
    segment_df_long_stats.last_3_game_blocks_ratio.quantile(.8),
    segment_df_long_stats.last_3_game_blocks_ratio.quantile(.9),
    segment_df_long_stats.last_3_game_blocks_ratio.quantile(.95)))
        print(segment_df_long_stats.last_5_game_blocks_ratio.quantile(.75),
    segment_df_long_stats.last_5_game_blocks_ratio.quantile(.8),
    segment_df_long_stats.last_5_game_blocks_ratio.quantile(.9),
    segment_df_long_stats.last_5_game_blocks_ratio.quantile(.95))
        print(segment_df_long_stats.current_season_blocks_ratio.quantile(.75),
    segment_df_long_stats.current_season_blocks_ratio.quantile(.8),
    segment_df_long_stats.current_season_blocks_ratio.quantile(.9),
    segment_df_long_stats.current_season_blocks_ratio.quantile(.95))
        return None
    else:
        pass
    BLOCKING_THRESHOLD = segment_df_long_stats.last_10_game_blocks_ratio.quantile(.90)
    segment_df_long_stats["blocker"] = np.where(segment_df_long_stats.last_10_game_blocks_ratio>=BLOCKING_THRESHOLD,True,False)
 # This dataframe has 5 players each at game,segment and team id level, we will aggregate across these 5 players to count number of blockers
    segment_team_level_blockers = segment_df_long_stats.groupby(["game_id","segment_id","team_id"],as_index=False)["blocker"].sum().rename(columns={"blocker":"num_blockers_on_team"})
    return segment_team_level_blockers


def get_segment_usage_ratios(segments_df_long,team_game_player_stats_df):
    segment_df_long_stats = segments_df_long.merge(team_game_player_stats_df,on=["team_id","game_id","player_id"],how="left")
    return segment_df_long_stats[['game_id','segment_id','player_id','team_id','last_1_game_usage_ratio','last_3_game_usage_ratio','last_5_game_usage_ratio','last_10_game_usage_ratio','current_season_usage_ratio']].drop_duplicates()


def get_segment_positions(segments_df_long,players_df):
    segments_df_long_player_chars = segments_df_long.merge(players_df,how="left",left_on="player_id",right_on="player")
    segments_df_long_player_chars["Big"] = np.where(segments_df_long_player_chars.LBA_position=="Big",1,0)
    segments_df_long_player_chars["Mid"] = np.where(segments_df_long_player_chars.LBA_position=="Mid",1,0)
    segments_df_long_player_chars["Small"] = np.where(segments_df_long_player_chars.LBA_position=="Small",1,0)
    segments_df_player_chars = segments_df_long_player_chars.groupby(["game_id","segment_id","team_id"],as_index=False)[["Big","Mid","Small"]].sum()
    return segments_df_player_chars


def get_all_features_at_shot_level(shots_df,segments_df,feature_list_1_shot_level,feature_list_2_game_segment_team_level,feature_list_3_game_segment_player_level,feature_list_4_game_segment_team_level,feature_list_5_player_level):
    # feature_list_1_shot_level#, ['shot_id']
    #feature_list_2_game_segment_team_level['game_id','segment_id','team_id'] join on opponent_team_id
    #feature_list_3_game_segment_player_level['game_id','segment_id','player_id','team_id'] join on all 4
    #feature_list_4_game_segment_team_level #['game_id','segment_id','player_id','team_ids'] join_on opponent_team_id
    shots = shots_df[['game_id','team_id','season_id','player_id','shot_id','segment_id',"score_pre_shot","score_pre_shot_diff"]].merge(segments_df[['segment_id','teamid1','teamid2','players_team1','players_team2']],on="segment_id",how="left")
    shots = shots.merge(feature_list_1_shot_level,on='shot_id',how='left')
    shots['opponent_team_id'] = np.where(shots.team_id==shots.teamid1,shots.teamid2,shots.teamid1)
    shots = shots.merge(feature_list_2_game_segment_team_level,how="left",left_on=['game_id','segment_id','opponent_team_id'],right_on=['game_id','segment_id','team_id'])
    shots = shots.drop(columns='team_id_y').rename(columns={'team_id_x':'team_id'})
    shots = shots.merge(feature_list_3_game_segment_player_level,on=['game_id','segment_id','player_id','team_id'],how='left')
    shots = shots.merge(feature_list_4_game_segment_team_level,left_on=['game_id','segment_id','opponent_team_id'],right_on=['game_id','segment_id','team_id'],how='left')
    shots = shots.drop(columns='team_id_y').rename(columns={'team_id_x':'team_id'})
    shots = shots.merge(feature_list_5_player_level,on="player_id",how="left")
    return shots

