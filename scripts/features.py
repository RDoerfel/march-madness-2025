# taken from https://www.kaggle.com/code/robikscube/machine-learning-bracket-gpu-powered/notebook
# %%
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

# %%
data_dir = Path(__file__).parent.parent / "data"

# %% load seeds data
df_seeds = pd.concat(
    [
        pd.read_csv(data_dir / "MNCAATourneySeeds.csv").assign(League="M"),
        pd.read_csv(data_dir / "WNCAATourneySeeds.csv").assign(League="W"),
    ],
).reset_index(drop=True)

df_season_results = pd.concat(
    [
        pd.read_csv(data_dir / "MRegularSeasonCompactResults.csv").assign(League="M"),
        pd.read_csv(data_dir / "WRegularSeasonCompactResults.csv").assign(League="W"),
    ]
).reset_index(drop=True)

df_tourney_results = pd.concat(
    [
        pd.read_csv(data_dir / "MNCAATourneyCompactResults.csv").assign(League="M"),
        pd.read_csv(data_dir / "WNCAATourneyCompactResults.csv").assign(League="W"),
    ]
).reset_index(drop=True)

# %%
df_team_season_results = pd.concat(
    [
        df_season_results[["Season", "League", "WTeamID", "DayNum", "WScore", "LScore"]]
        .assign(GameResult="W")
        .rename(columns={"WTeamID": "TeamID", "WScore": "TeamScore", "LScore": "OppScore"}),
        df_season_results[["Season", "League", "LTeamID", "DayNum", "WScore", "LScore"]]
        .assign(GameResult="L")
        .rename(columns={"LTeamID": "TeamID", "LScore": "TeamScore", "WScore": "OppScore"}),
    ]
).reset_index(drop=True)

# %%
# Score Differential
df_team_season_results["ScoreDiff"] = df_team_season_results["TeamScore"] - df_team_season_results["OppScore"]
df_team_season_results["Win"] = (df_team_season_results["GameResult"] == "W").astype("int")

# %%
# Aggregate the data
team_season_agg = (
    df_team_season_results.groupby(["Season", "TeamID", "League"])
    .agg(
        AvgScoreDiff=("ScoreDiff", "mean"),
        MedianScoreDiff=("ScoreDiff", "median"),
        MinScoreDiff=("ScoreDiff", "min"),
        MaxScoreDiff=("ScoreDiff", "max"),
        VarScoreDiff=("ScoreDiff", "var"),
        Wins=("Win", "sum"),
        Losses=("GameResult", lambda x: (x == "L").sum()),
        WinPercentage=("Win", "mean"),
    )
    .reset_index()
)

# %%
df_seeds["ChalkSeed"] = df_seeds["Seed"].str.replace("a", "").str.replace("b", "").str[1:].astype("int")

team_season_agg = team_season_agg.merge(df_seeds, on=["Season", "TeamID", "League"], how="left")

# %%
df_team_tourney_results = pd.concat(
    [
        df_tourney_results[["Season", "League", "WTeamID", "LTeamID", "WScore", "LScore"]]
        .assign(GameResult="W")
        .rename(
            columns={
                "WTeamID": "TeamID",
                "LTeamID": "OppTeamID",
                "WScore": "TeamScore",
                "LScore": "OppScore",
            }
        ),
        df_tourney_results[["Season", "League", "LTeamID", "WTeamID", "LScore", "WScore"]]
        .assign(GameResult="L")
        .rename(
            columns={
                "LTeamID": "TeamID",
                "WTeamID": "OppTeamID",
                "LScore": "TeamScore",
                "WScore": "OppScore",
            }
        ),
    ]
).reset_index(drop=True)

df_team_tourney_results["Win"] = (df_team_tourney_results["GameResult"] == "W").astype("int")

# %%
df_historic_tourney_features = df_team_tourney_results.merge(
    team_season_agg[["Season", "League", "TeamID", "WinPercentage", "MedianScoreDiff", "VarScoreDiff", "ChalkSeed"]],
    on=["Season", "League", "TeamID"],
    how="left",
).merge(
    team_season_agg[
        ["Season", "League", "TeamID", "WinPercentage", "VarScoreDiff", "MedianScoreDiff", "ChalkSeed"]
    ].rename(
        columns={
            "TeamID": "OppTeamID",
            "WinPercentage": "OppWinPercentage",
            "MedianScoreDiff": "OppMedianScoreDiff",
            "VarScoreDiff": "OppVarScoreDiff",
            "ChalkSeed": "OppChalkSeed",
        }
    ),
    on=["Season", "League", "OppTeamID"],
)

# %%
df_historic_tourney_features["WinPctDiff"] = (
    df_historic_tourney_features["WinPercentage"] - df_historic_tourney_features["OppWinPercentage"]
)

df_historic_tourney_features["ChalkSeedDiff"] = (
    df_historic_tourney_features["ChalkSeed"] - df_historic_tourney_features["OppChalkSeed"]
)

df_historic_tourney_features["MedianScoreDiffDiff"] = (
    df_historic_tourney_features["MedianScoreDiff"] - df_historic_tourney_features["OppMedianScoreDiff"]
)

df_historic_tourney_features["VarScoreDiffDiff"] = (
    df_historic_tourney_features["VarScoreDiff"] - df_historic_tourney_features["OppVarScoreDiff"]
)
# %%
df_historic_tourney_features["BaselinePred"] = (
    df_historic_tourney_features["ChalkSeed"] < df_historic_tourney_features["OppChalkSeed"]
)

df_historic_tourney_features.loc[
    df_historic_tourney_features["ChalkSeed"] == df_historic_tourney_features["OppChalkSeed"],
    "BaselinePred",
] = (
    df_historic_tourney_features["WinPercentage"] > df_historic_tourney_features["OppWinPercentage"]
)

# %%
cv_scores_baseline = []
for season in df_historic_tourney_features["Season"].unique():
    pred = df_historic_tourney_features.query("Season == @season")["BaselinePred"].astype("int")
    y = df_historic_tourney_features.query("Season == @season")["Win"]
    score = accuracy_score(y, pred)
    score_ll = log_loss(y, pred)
    cv_scores_baseline.append(score)
    print(f"Holdout season {season} - Accuracy {score:0.4f} Log Loss {score_ll:0.4f}")

print(f"Baseline accuracy {np.mean(cv_scores_baseline):0.4f}")

# %%
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, accuracy_score
import numpy as np

FEATURES = ["WinPctDiff", "ChalkSeedDiff", "MedianScoreDiffDiff", "VarScoreDiffDiff"]
TARGET = "Win"

X = df_historic_tourney_features[FEATURES]
y = df_historic_tourney_features[TARGET]
groups = df_historic_tourney_features["Season"]
seasons = df_historic_tourney_features["Season"].unique()

# Setup cross-validation
gkf = GroupKFold(n_splits=df_historic_tourney_features["Season"].nunique())
cv_results = []
models = []

season_idx = 0
for train_index, test_index in gkf.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Prepare the model
    model = lgb.LGBMRegressor(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.001,
    )
    holdout_season = seasons[season_idx]
    print(f"Holdout Season: {holdout_season}")

    # Train the model
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    # Predict on the test set
    y_pred = model.predict(X_test)
    score_ll = log_loss(y_test, y_pred)
    y_pred = y_pred > 0.5

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cv_results.append(accuracy)
    season_idx += 1
    print(f"Season {holdout_season}: {accuracy:.4f}, Log Loss: {score_ll:.4f}")
    models.append(model)

# Print the average accuracy across all folds
print("Average CV Accuracy:", np.mean(cv_results))

# %%
from catboost import CatBoostClassifier

X = df_historic_tourney_features[FEATURES]
y = df_historic_tourney_features[TARGET]
groups = df_historic_tourney_features["Season"]
seasons = df_historic_tourney_features["Season"].unique()

# Setup cross-validation
gkf = GroupKFold(n_splits=df_historic_tourney_features["Season"].nunique())
cv_results = []
models = []

season_idx = 0
for train_index, test_index in gkf.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Prepare the model
    model = CatBoostClassifier(iterations=1000, learning_rate=0.001, depth=6, loss_function="Logloss", verbose=100)
    holdout_season = seasons[season_idx]
    print(f"Holdout Season: {holdout_season}")

    # Train the model
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

    # Predict on the test set
    y_pred = model.predict(X_test)
    score_ll = log_loss(y_test, y_pred)
    y_pred = y_pred > 0.5

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cv_results.append(accuracy)
    season_idx += 1
    print(f"Season {holdout_season}: {accuracy:.4f}, Log Loss: {score_ll:.4f}")
    models.append(model)

# Print the average accuracy across all folds
print("Average CV Accuracy:", np.mean(cv_results))

# %%
import xgboost as xgb

season_idx = 0
for train_index, test_index in gkf.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Prepare the model
    model = xgb.XGBRegressor(
        eval_metric="logloss",
        n_estimators=1_000,
        learning_rate=0.001,
    )
    holdout_season = seasons[season_idx]
    print(f"Holdout Season: {holdout_season}")
    # Train the model
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    # Predict on the test set
    y_pred = model.predict(X_test)
    score_ll = log_loss(y_test, y_pred)
    y_pred = y_pred > 0.5
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cv_results.append(accuracy)
    season_idx += 1
    print(f"Season {holdout_season}: {accuracy} {score_ll}")
    models.append(model)
# Print the average accuracy across all folds
print("Average CV Accuracy:", np.mean(cv_results))

# %%
