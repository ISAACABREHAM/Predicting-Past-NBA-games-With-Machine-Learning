#!/usr/bin/env python
# coding: utf-8

# In[336]:


import pandas as pd


# In[337]:


df = pd.read_csv("/Users/ISAACABREHAM/Downloads/nba_games.csv", index_col=0)


# In[338]:


df


# In[339]:


# Each row in the data frame is represents a single NBA game


# In[340]:


# sorting Value by date recent to oldest

df = df.sort_values("date")


# In[341]:


df = df.reset_index(drop=True)


# In[342]:


df


# In[343]:


# removing columns that arent necessary

del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]


# In[344]:


df


# In[345]:


# Preparing the data for machine learning

def add_target (team):
    team["target"] = team["won"].shift(-1)
    return team

df = df.groupby("team", group_keys=False).apply(add_target)


# In[346]:


df[df["team"]== "LAL"]


# In[347]:


# Nul values from end of season

df["target"][pd.isnull(df["target"])] = 2


# In[348]:


df["target"] = df["target"].astype(int, errors="ignore")


# In[349]:


#Looking out our dataframe here target now has 3 values

# 0 is LOSE

# 1 is WIN

#2 NO DATA FOR NEXT GAME


df


# In[350]:


# Checkin if columns are balanced. NBA games always have a winner or a loser so true and flase must be even

df["won"].value_counts()


# In[351]:


df['target'].value_counts()


# In[352]:


# Removing columns with no values

nulls = pd.isnull(df)


# In[353]:


nulls = nulls.sum()


# In[354]:


nulls = nulls[nulls > 0]


# In[355]:


nulls


# In[356]:


## removing extra columns

valid_columns = df.columns[~df.columns.isin(nulls.index)]


# In[357]:


valid_columns


# In[358]:


# creating a copy of the data frame rather than a slice of it. Dont want to get a copy warning because when you assign a data frame back slices to itself you may get a copy warning 
   
df = df[valid_columns].copy()


# In[359]:


# Here We are done DATA CLEANING and beggining our MACHINE LEARNING


# In[360]:


###NOTES; Cross-validation is a strategy in machine learning where the dataset is partitioned into multiple subsets, and the model is trained and evaluated iteratively, ensuring that each subset serves both as a training and testing set, to provide a more robust assessment of the model's performance.

from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit

rr = RidgeClassifier(alpha=1)

split = TimeSeriesSplit(n_splits=3)

sfs = SequentialFeatureSelector(rr, 
                                n_features_to_select=30, 
                                direction="forward",
                                cv=split,
                                n_jobs=1
                               )


# In[361]:


removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = df.columns[~df.columns.isin(removed_columns)]


# In[362]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])


# In[363]:


df


# In[364]:


sfs.fit(df[selected_columns], df["target"])


# In[365]:


predictors = list(selected_columns[sfs.get_support()])


# In[366]:


# Creating the function that does our predictions

# The backtest function is going to split data up by season and use past seasons to predict future seasons

def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []
    
    seasons = sorted(data["season"].unique())  #creating a list of all our seasons
    
    for i in range(start, len(seasons), step):
        season = seasons[i]
        
        train = data[data["season"] < season] # training set 
        test = data[data["season"] == season] # test set 
        
        model.fit(train[predictors], train["target"]) 
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)   #concatenate method combines dataframes for us
        


# In[367]:


predictions = backtest(df, rr, predictors)


# In[368]:


predictions


# In[369]:


from sklearn.metrics import accuracy_score

predictions 

accuracy_score(predictions["actual"], predictions["prediction"]) #as you can see below our current accuracy rate is 54%


# In[370]:


df.groupby("home").apply(lambda x: x[x["won"] == 1].shape[0] / x.shape[0]) # apply function to see how much the team wins whether theyre at home or not


# In[371]:


# adding 10 day rolling average as a parameter to our prediction

df_rolling = df[list(selected_columns) + ["won", "team", "season"]]  

def find_team_averages(team):
    rolling = team.rolling(10).mean()
    return rolling

df_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages) #grouping by team and season. Want the rolling average for the last 10 games of that season
    


# In[372]:


df_rolling #missing rows at beggining becuase at start of season team hasnt played 10 games yet


# In[373]:


rolling_cols = [f"{col}_10" for col in df_rolling.columns]
df_rolling.columns = rolling_cols

df = pd.concat([df, df_rolling], axis=1)


# In[374]:


df = df.dropna()  #drops all missing values from those first 10 games


# In[375]:


df  


# In[376]:


#creating functions to give the algorithim info on the next game

def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):
    return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

df["home_next"] = add_col(df, "home")
df["team_opp_next"] = add_col(df, "team_opp")
df["date_next"] = add_col(df, "date")


# In[377]:


df


# In[378]:


df = df.copy()


# In[379]:


#pulling data from the opponents last 10 games

full = df.merge(
    df[rolling_cols + ["team_opp_next", "date_next", "team"]],
    left_on=["team", "date_next"], 
    right_on=["team_opp_next","date_next"]
)


# In[380]:


full


# In[381]:


full[["team_x", "team_opp_next_x", "team_y", "team_opp_next_y", "date_next"]]


# In[382]:


removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns


# In[383]:


selected_columns = full.columns[~full.columns.isin(removed_columns)]


# In[384]:


sfs.fit(full[selected_columns], full["target"]) #sfs is going to give us the 30 best features it think will give us the best value for predicting our target


# In[385]:


predictors = list(selected_columns[sfs.get_support()]) # returns a boleen whether a feature was selected or not


# In[386]:


predictors


# In[387]:


#GENERATING OUR FINAL PREDICTIONS


# In[388]:


predictions = backtest(full, rr, predictors)


# In[389]:


accuracy_score(predictions["actual"], predictions["prediction"])

