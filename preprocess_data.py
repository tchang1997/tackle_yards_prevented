from argparse import ArgumentParser
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

tqdm.pandas()

# TODO: put these constants in a YAML file
PLAYS_RELEVANT_COLS = ['ballCarrierId', 'ballCarrierDisplayName', 'quarter', 'down', 'yardsToGo', 'possessionTeam', 'defensiveTeam', 'yardlineSide', 'yardlineNumber', 'gameClock',
    'preSnapHomeScore', 'preSnapVisitorScore', 'passResult', 'absoluteYardlineNumber', 'prePenaltyPlayResult', 'playResult', 'offenseFormation', 'defendersInTheBox', 'passProbability']
RELEVANT_EVENTS = ["first_contact", "ball_snap", "pass_outcome_caught", "handoff", "pass_arrived", "out_of_bounds", "run", "man_in_motion", "play_action", "touchdown", "fumble"]
RELEVANT_GAME_INFO = ["gameId", "playId", "displayName", "time", "club"]
RELEVANT_PLAYER_PLAY_INFO = ["x", "y", "s", "a", "o", "dir"]
YARDAGE_BINS = [0, 3, 7, 10, 100]
YARDAGE_CATEGORIES = ["short", "medium", "long", "very_long"]
DEFENDERS_IN_BOX_BINS = [0, 4, 5, 6, 7, 8, 12]
class NFLBDBDataLoader(object):
    def __init__(self, base_path="./data/nfl-big-data-bowl-2024", weeks=[1, 2, 3, 4, 5, 6, 7, 8, 9]):

        print("Loading game info...")
        self.games = pd.read_csv(os.path.join(base_path, "games.csv"))

        print("Loading play info...")
        self.plays = pd.read_csv(os.path.join(base_path, "plays.csv"))

        print("Loading player info...")
        self.players = pd.read_csv(os.path.join(base_path, "players.csv")).drop("displayName", axis=1)

        print("Loading tracking data (this could take a while)...")
        self.weeks = weeks
        self.tracking_data = pd.concat([pd.read_csv(os.path.join(base_path, f"tracking_week_{i}.csv")) for i in self.weeks], axis=0)
        self.tackles = pd.read_csv(os.path.join(base_path, "tackles.csv"))

        play_not_nullified_by_penalty = (self.plays['playNullifiedByPenalty'] != "Y")
        full_play_cols = PLAYS_RELEVANT_COLS + ['gameId', 'playId']
        self.non_penalty_plays = pd.merge(self.plays.loc[play_not_nullified_by_penalty, full_play_cols], self.games, how="inner", on="gameId") \
            .drop(columns=["homeFinalScore", "visitorFinalScore"]).set_index(["gameId", "playId"])

        self.base_path = base_path
        self.intermediate_data_path = os.path.join(self.base_path, "_play_by_play_intermediate.pkl")
        self.geometric_feature_path = os.path.join(self.base_path, "_play_by_play_geometric_features.pkl")
        self.final_split_path_stub = os.path.join(self.base_path, "play_by_play")
        print("Dataloader ready.")

    def encode_play_features(self):
        quarter = pd.get_dummies(self.non_penalty_plays["quarter"], prefix="Q", prefix_sep="").rename(columns={"Q5": "OT"}).astype(int)
        down = pd.get_dummies(self.non_penalty_plays["down"], prefix="down").astype(int)
        yardage = pd.get_dummies(pd.cut(self.non_penalty_plays["yardsToGo"], bins=YARDAGE_BINS, right=True, include_lowest=True, labels=YARDAGE_CATEGORIES), prefix="yardage").astype(int)

        time = self.non_penalty_plays["gameClock"].str.split(":", expand=True)
        time_remaining = pd.Series(time.loc[:, 0].astype(int) + time.loc[:, 1].astype(int) / 60., name="time")

        home_team_has_the_ball = (self.non_penalty_plays["possessionTeam"] == self.non_penalty_plays["homeTeamAbbr"])
        possession_team_score = self.non_penalty_plays["preSnapHomeScore"].where(home_team_has_the_ball, self.non_penalty_plays["preSnapVisitorScore"])
        defensive_team_score = self.non_penalty_plays["preSnapVisitorScore"].where(home_team_has_the_ball, self.non_penalty_plays["preSnapHomeScore"])
        possession_team_deficit = possession_team_score - defensive_team_score

        formation = pd.get_dummies(self.non_penalty_plays["offenseFormation"], prefix="formation").astype(int)
        defenders_in_box = pd.get_dummies(pd.cut(self.non_penalty_plays["defendersInTheBox"].fillna(-1), bins=DEFENDERS_IN_BOX_BINS, right=True), prefix="n_def_in_box").astype(int)

        self._play_features = pd.concat([
            quarter,
            down,
            yardage,
            time_remaining,
            possession_team_score,
            defensive_team_score,
            possession_team_deficit,
            formation,
            defenders_in_box], axis=1)
        return self._play_features

    def preprocess_intermediate(self):

        if os.path.isfile(self.intermediate_data_path):
            print("Intermediate data already exists at", self.intermediate_data_path, "-- reloading.")
            self._intermediate = pickle.load(open(self.intermediate_data_path, "rb"))
            return self._intermediate

        def collect_play_by_play_information(play_data):
            """
                This is an intermediate, human-readable representation of play-by-play data.
            """

            try:
                play_groups = play_data.groupby("nflId")
                game_id, play_id = play_data[["gameId", "playId"]].iloc[0]
                first_group = play_groups.get_group(list(play_groups.groups.keys())[0])
                time_to_first_contact = first_group.event.tolist().index("first_contact")

                play_timeseries = pd.concat([g.reset_index(drop=True).loc[:, RELEVANT_PLAYER_PLAY_INFO] for _, g in play_groups], axis=1, keys=play_groups.groups.keys())
                truncated_play_timeseries = play_timeseries.loc[:time_to_first_contact]

                event_timeseries = first_group.event.reset_index(drop=True).loc[:time_to_first_contact]
                event_timeseries_encoded = pd.DataFrame(event_timeseries.values.reshape(-1, 1) == np.array(RELEVANT_EVENTS), columns=RELEVANT_EVENTS).astype(int)
                player_data_for_play = pd.merge(play_groups.first().loc[:, RELEVANT_GAME_INFO], self.players.drop(columns="collegeName"), how="left", on="nflId")
                player_data_for_play_with_tackles = pd.merge(player_data_for_play, self.tackles, how="left", on=["gameId", "playId", "nflId"])

                # whether there was a successful tackle/forced fumble or a missed tackle, who tackled, and such
                tackle_label = int(player_data_for_play_with_tackles.loc[:, ["pff_missedTackle"]].sum(axis=0).max() == 0)

                # yards after first_contact
                play_features = self.non_penalty_plays.loc[(game_id, play_id), PLAYS_RELEVANT_COLS]
                # play_features_with_game_data = pd.merge(play_features, self.games, how="left", on="gameId")
                direction = 2 * int(first_group["playDirection"].iloc[0] == "right") - 1
                ball_carrier_trajectory = play_timeseries.xs(play_features.ballCarrierId, axis=1)
                yds_at_first_contact = ball_carrier_trajectory.loc[time_to_first_contact, "x"]
                yds_final = ball_carrier_trajectory.loc[len(ball_carrier_trajectory) - 1, "x"]
                outcome = direction * (yds_final - yds_at_first_contact)

            except Exception as e:
                raise RuntimeError(f"During preprocessing of game_id, play_id ({game_id}, {play_id}), the following exception was raised: {e}")

            return {
                "game_id": game_id,
                "play_id": play_id,
                "player_tracking": truncated_play_timeseries,
                "event_timeseries": event_timeseries_encoded,
                "players_on_the_field": player_data_for_play_with_tackles,
                "play_features": play_features,
                "play_features_encoded": self._play_features.loc[(game_id, play_id)],
                "tackle_successful": tackle_label,
                "yards_after_contact": outcome,
            }
        self._intermediate = self.tracking_data.groupby(["gameId", "playId"]) \
            .filter(lambda g: ("first_contact" in g["event"].values) and (g.name in self.non_penalty_plays.index) and (g["nflId"].nunique() > 0)) \
            .groupby(["gameId", "playId"]).progress_apply(collect_play_by_play_information)

        print("Saving intermediate data to", self.intermediate_data_path, f"({len(self._intermediate)} rows)")
        with open(self.intermediate_data_path, 'wb') as f:
            pickle.dump(self._intermediate, f)
        return self._intermediate

    def create_geometric_features(self, include_events=True):
        if os.path.isfile(self.geometric_feature_path):
            print("Geometric features already extracted at", self.geometric_feature_path, "-- reloading.")
            self.final_geometric_features = pickle.load(open(self.geometric_feature_path, "rb"))
            return self.final_geometric_features

        def get_relative_features_to_ball_carrier(group):
            group_ = group.T.droplevel(0, axis=1)
            dist = np.sqrt(np.square(group_.loc[:, ['x', 'y']] - ball_carrier_tracking_data.loc[:, ['x', 'y']]).sum(axis=1))

            delta_angle = ball_carrier_tracking_data.loc[:, 'dir'] - group_.loc[:, 'dir']

            player_speed = group_.loc[:, 's']
            ball_carrier_speed = ball_carrier_tracking_data.loc[:, 's']
            rel_spd = np.sqrt(player_speed ** 2 + ball_carrier_speed ** 2 - 2 * player_speed * ball_carrier_speed * np.cos(delta_angle))

            player_accel = group_.loc[:, 'a']
            ball_carrier_accel = ball_carrier_tracking_data.loc[:, 'a']
            rel_accel = np.sqrt(player_accel ** 2 + ball_carrier_accel ** 2 - 2 * player_accel * ball_carrier_accel * np.cos(delta_angle))

            return pd.DataFrame({
                "distance": dist,
                "relative_speed": rel_spd,
                "relative_acceleration": rel_accel,
                "delta_angle": delta_angle,
                "cosine_similarity": np.cos(delta_angle),
            })

        final_features = []
        for play in tqdm(self._intermediate):  # TODO: parallelize
            tracking_data = play["player_tracking"]
            ball_carrier_id = play["play_features"].loc["ballCarrierId"]
            ball_carrier_tracking_data = tracking_data.loc[:, ball_carrier_id]
            non_ball_carriers = tracking_data.T.groupby(level=0).filter(lambda g: g.name != ball_carrier_id)
            geometric_features = non_ball_carriers.groupby(level=0).apply(get_relative_features_to_ball_carrier)

            onfield = play['players_on_the_field']
            play_features = play['play_features']
            offense = onfield.loc[(onfield["club"] == play_features["possessionTeam"]) & (onfield["nflId"] != play_features["ballCarrierId"]), "nflId"]
            defense = onfield.loc[onfield["club"] == play_features["defensiveTeam"], "nflId"]

            offense_feature_groups = geometric_features.loc[pd.IndexSlice[offense]].groupby(level=0)
            defense_feature_groups = geometric_features.loc[pd.IndexSlice[defense]].groupby(level=0)

            offense_features = pd.concat([g.reset_index(drop=True) for _, g in offense_feature_groups], axis=1, keys=offense_feature_groups.groups.keys())
            defense_features = pd.concat([g.reset_index(drop=True) for _, g in defense_feature_groups], axis=1, keys=defense_feature_groups.groups.keys())
            offense_raw = tracking_data.loc[:, pd.IndexSlice[offense]]
            defense_raw = tracking_data.loc[:, pd.IndexSlice[defense]]

            tacklers = onfield.loc[(onfield["tackle"] > 0) | (onfield["assist"] > 0)].nflId
            tackler_tracking_data = defense_raw.loc[:, pd.IndexSlice[tacklers.tolist(), :]]

            feature_dict = {
                "offense_geometric": offense_features,
                "defense_geometric": defense_features,
                "offense_raw": offense_raw,
                "defense_raw": defense_raw,
                "ball_carrier_raw": ball_carrier_tracking_data,
                "tacklers_raw": tackler_tracking_data,
            }
            final_features.append({**feature_dict, **play})
        self.final_geometric_features = final_features
        print("Saving final features data to", self.geometric_feature_path)
        with open(self.geometric_feature_path, 'wb') as f:
            pickle.dump(self.final_geometric_features, f)
        return self.final_geometric_features

    def make_splits(self, split_names=["train", "val", "test"], split_sizes=[0.7, 0.2, 0.1]):
        remaining = self.final_geometric_features
        split_sizes = np.array(split_sizes)
        for i in range(len(split_names)):
            if len(split_sizes) > 1:
                remaining, X_result = train_test_split(remaining, test_size=split_sizes[-1], random_state=i)
                print(f"Saving split '{split_names[-(i+1)]}' with {len(X_result)} records")
                with open(self.final_split_path_stub + f"_{split_names[-(i+1)]}.pkl", 'wb') as f:
                    pickle.dump(X_result, f)
                split_sizes = split_sizes[:-1] / split_sizes[:-1].sum()
            else:
                print(f"Saving split '{split_names[0]}' with {len(remaining)} records")
                with open(self.final_split_path_stub + f"_{split_names[0]}.pkl", 'wb') as f:
                    pickle.dump(remaining, f)

    def run_full_pipeline(self):
        self.encode_play_features()
        self.preprocess_intermediate()
        self.create_geometric_features()
        self.make_splits()


if __name__ == '__main__':
    dataloader = NFLBDBDataLoader()
    dataloader.run_full_pipeline()
