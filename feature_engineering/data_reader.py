import pandas as pd
import os
from bson.objectid import ObjectId
import logging
logging.basicConfig(level=logging.INFO)


def read_data(raw_data_path="../../raw_data/",league="NCAAM1",season="21-22"):
    logging.info(os.listdir(os.path.join(raw_data_path,league,"S-"+season)))
    dataframes ={}
    datapath = os.path.join(raw_data_path,league,"S-"+season)
    logging.info(f"reading files from {datapath}")
    logging.info(f"files present: {os.listdir(datapath)}")
    filenames = {"games_df":" ".join([league,season,"gamesdf.pickle"]),
                 "players_df":" ".join([league,season,"playerdictionary.pickle"]),
                 "segments_df":" ".join([league,season,"segment data.pickle"]),
                 "segments_players_usage_blocking_df":" ".join([league,season,"segment player block and usage data.pickle"]),
                 "shots_df":" ".join([league,season,"shotsdf.pickle"]),
                 "teams_df":" ".join([league,season,"teamsdf.pickle"]),}
    try: 
        dataframes['games_df'] = pd.read_pickle(os.path.join(datapath,filenames["games_df"]))
        dataframes['players_df'] = pd.read_pickle(os.path.join(datapath,filenames["players_df"]))
        dataframes['segments_df'] = pd.read_pickle(os.path.join(datapath,filenames["segments_df"]))
        dataframes['segments_players_usage_blocking_df'] = pd.read_pickle(os.path.join(datapath,filenames["segments_players_usage_blocking_df"]))
        dataframes['shots_df'] = pd.read_pickle(os.path.join(datapath,filenames["shots_df"]))
        dataframes['teams_df'] = pd.read_pickle(os.path.join(datapath,filenames["teams_df"]))
    except:
        raise ValueError("Check format of file names present in directory")
    for name,frame in dataframes.items():
        if type(frame)!= pd.DataFrame:
            dataframes[name] = pd.DataFrame(frame)
    return dataframes



def validate_data_types(dataframes):
    games_df_col_dtype_dict ={'game_id':ObjectId, 'season':str, 'league':str, 'date':str, 'has_locations':bool, 'has_segments':bool,
        'team1':ObjectId, 'team2':ObjectId, 'hometeam':ObjectId, 'winning_team':ObjectId}
    players_df_col_dtype_dict = {'player':ObjectId, 'height':int, 'LBA_position':str, 'Genius_position':str, 'year':str,
        'seasonid':ObjectId}
    segments_df_col_dtype_dict = {'game_id':ObjectId, 'segment_id':ObjectId, 'teamid1':ObjectId, 'teamid2':ObjectId, 'players_team1':"player_list",
        'players_team2':"player_list", 'possessions_team1':int, 'possessions_team2':int,
        'twoshots_team1':int, 'twoshots_team2':int}
    segments_players_usage_blocking_df_col_dtype_dict = {'segment_id':ObjectId, 'player_id':ObjectId, 'uses':float, 'blocks':int}
    shots_df_col_dtype_dict = {'game_id':ObjectId, 'team_id':ObjectId, 'season_id':ObjectId, 'player_id':ObjectId, 'shot_id':ObjectId, 'score':int,
        'score_opp':int, 'Period':int, 'segment_id':ObjectId, 'Time':str, 'Time_Seconds':int, 'Stat':str,
        'Zone':str, 'x_coordinate':float, 'y_coordinate':float, 'Angle':float, 'Distance':float}
    teams_df_col_dtype_dict = {'team_id':ObjectId,'season_id':ObjectId,'conferenceid':ObjectId}
    #sum([type(i)==ObjectId for i in games_df.game_id])==len(games_df)
    all_df_col_dtypes_dict = {"games_df":games_df_col_dtype_dict,"players_df":players_df_col_dtype_dict,"segments_df":segments_df_col_dtype_dict,\
                            "segments_players_usage_blocking_df":segments_players_usage_blocking_df_col_dtype_dict,\
                            "shots_df":shots_df_col_dtype_dict,"teams_df":teams_df_col_dtype_dict  }
    errors ={}
    for df,df_dict in all_df_col_dtypes_dict.items():
        for col,dtype in df_dict.items():
            if col not in dataframes[df].columns:
                logging.warn(f"{col} not present in {df}")
                continue
            if dtype=='player_list':
                dataframes[df]['_error']  = 1
                for index,value in enumerate(dataframes[df][col]):
                    if isinstance(value, list):
                         if all(isinstance(s, ObjectId) for s in value) and (len(value)==5):
                            dataframes[df].loc[index,'_error'] = 0
                if sum(dataframes[df]['_error'])!=0:
                    logging.warn(f"Check required in {df} for column {col}, expected datatype:{dtype}")
                    errors[df+"__"+col] = dataframes[df][dataframes[df]['_error']==1][col]
                    logging.warn(f"{df+'__'+col}: {errors[df+'__'+col].shape} flagged rows of {dataframes[df].shape} rows")
                del(dataframes[df]['_error'])

            elif sum([type(i)==dtype for i in dataframes[df][col]])!=len(dataframes[df]):
                logging.warn(f"Check required in {df} for column {col}, expected datatype:{dtype}")
                errors[df+"__"+col] = dataframes[df][[type(i)!=dtype for i in dataframes[df][col]]][col]
                logging.warn(f"{df+'__'+col}: {errors[df+'__'+col].shape} flagged rows of {dataframes[df].shape} rows")
    for error in errors.keys():
        logging.warn("Inspect examples to check for potential errors")
        logging.warn(errors[error].head(3))
    return errors

def clean_shots_data(shots_df):
    #Remove rows with no player_id
    shots_df = shots_df[shots_df.player_id!=False]
    #Remove all the corner 3s that are registered as 2s
    shots_df = shots_df[~(((shots_df.Zone==' 4-1')&(shots_df.x_coordinate<=3))|((shots_df.Zone==' 4-3')&(shots_df.x_coordinate>=47)))]
    #Remove all the above the break 3s that are registered as 2s
    shots_df = shots_df[~(((shots_df.Zone==' 4-1')|(shots_df.Zone==' 4-3')|(shots_df.Zone==' 4-2'))&(shots_df.Distance>=22.1458))]
    #Remove all the 2s that are registered as above the break 3s
    shots_df = shots_df[~(((shots_df.Zone==' 6-1')|(shots_df.Zone==' 6-3')|(shots_df.Zone==' 6-2'))&(shots_df.Distance<22.1458))]
    #Remove all the 2s that are registered as corner 3s
    shots_df = shots_df[~(((shots_df.Zone==' 5-1')&(shots_df.x_coordinate>3))|((shots_df.Zone==' 5-2')&(shots_df.x_coordinate<47)&(shots_df.x_coordinate>3)))]
    #Reclassify the misclassified corner 3s (left instead of right corner)
    shots_df.loc[((shots_df.Zone==' 5-2')&(shots_df.x_coordinate<=3)), 'Zone'] = ' 5-1'
    return shots_df.reset_index(drop=True)

def get_correct_game_date(dataframe,date_column,pad=False,format="%m/%d/%Y"):
    dataframe = dataframe.copy()
    def pad_date(x):
        m,d,y = x.split("/")
        if int(m)<10:
            m = "0"+str(int(m))
        if int(d)<10:
            d = "0"+str(int(d))
        y = "20"+y
        return "/".join([m,d,y])

    if pad==True:
        dataframe[date_column] = dataframe[date_column].apply(pad_date)
    return pd.to_datetime(dataframe[date_column],format = format)