import numpy

import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from consts import *
from transformers import BertTokenizer
import os
from collections import defaultdict
from utils.functions import learn_sigmoid_weighting_by_reaction_time, get_model_name, move_to
import Simulation.strategies_code as bot_strategies
import Simulation.dm_strategies as user_strategies
import random
import utils.basic_nature_options
from sklearn.linear_model import LogisticRegression
import pickle
from tqdm import trange
import pandas as pd
from consts import *
from utils import personas


class OfflineDataSet(Dataset):
    def __init__(self, user_groups, config, weight_type, strategies=None, users=None):
        self.config = config
        reviews_path = DATA_GAME_REVIEWS_PATH
        x_path = DATA_CLEAN_ACTION_PATH_X
        y_path = DATA_CLEAN_ACTION_PATH_Y
        self.actions_df = None
        if "X" in user_groups:  # Offline Human - training groups (E_A)
            self.actions_df = pd.read_csv(x_path)
            assert self.actions_df.user_id.max() + 1 == DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS
        if "Y" in user_groups: # Offline Human - testing groups (E_B)
            Y_dataset = pd.read_csv(y_path)
            assert Y_dataset.user_id.max() + 1 == DATA_CLEAN_ACTION_PATH_Y_NUMBER_OF_USERS
            Y_dataset.user_id += DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS
            if self.actions_df is None:
                self.actions_df = Y_dataset
            else:
                self.actions_df = pd.concat([self.actions_df, Y_dataset])
        if "L" in user_groups:  # LLM Simulated users
            self.actions_df = pd.read_csv(config["OFFLINE_SIM_DATA_PATH"])
            if self.config.personas_group_number > -1:
                g = personas.get_personas_in_group(self.config.personas_group_number)
                print("In this run, we using data from personas", g)
                self.actions_df = self.actions_df[self.actions_df["persona"].isin(g)]

        if strategies is not None:
            self.actions_df = self.actions_df[self.actions_df["strategy_id"].isin(strategies)]
            strategies_in_data = self.actions_df["strategy_id"].drop_duplicates().tolist()
            for strategy in strategies:
                assert strategy in strategies_in_data, f"You have no games against strategy #{strategy} " \
                                                       f"in the entire dataset!"

        if users is not None:
            self.actions_df = self.actions_df[self.actions_df["user_id"].isin(users.tolist())]
            assert self.actions_df["user_id"].nunique() == len(users.tolist()), "some of the users that chosen to used"\
                                                                                "are not exists in the dataset!"

        if "persona" in self.actions_df.columns:
            print("user per persona:", self.actions_df[["persona", "user_id"]].drop_duplicates().groupby("persona").count())

        grouped_counts = self.actions_df.groupby(["user_id", "strategy_id"]).size().reset_index(name="N")
        self.actions_df = self.actions_df.merge(grouped_counts, on=["user_id", "strategy_id"], how="left")
        number_of_groups = len(grouped_counts)
        total_samples = len(self.actions_df)
        self.actions_df["weight"] = 1
        if weight_type == "sender_receiver" or weight_type == "both":
            self.actions_df["weight"] *= total_samples / (self.actions_df["N"] * number_of_groups)
        if weight_type == "didGo" or weight_type == "both":
            p = self.actions_df["didGo"].mean()
            p_weight = (1 - p) / p
            q_weight = p / (1 - p)
            self.actions_df["weight"] *= np.where(self.actions_df["didGo"].to_numpy(), p_weight, q_weight)

        self.actions_df = self.actions_df.drop("N", axis=1)

        self.actions_df = self.actions_df.groupby(["user_id", "gameId"])

        self.idx_to_group = list(self.actions_df.indices.keys())
        self.group_to_idx = {g: i for i, g in enumerate(self.idx_to_group)}
        self.n_groups_by_user_id = defaultdict(list)
        for u, i in sorted(self.actions_df.indices.keys()):
            self.n_groups_by_user_id[u].append(i)

        self.review_reduced = pd.read_csv(config['FEATURES_PATH'], index_col=0).T.astype(int).to_dict(orient='list')
        self.review_reduced = {int(rid): torch.Tensor(vec) for rid, vec in self.review_reduced.items()}
        self.review_reduced[-1] = torch.zeros(config["REVIEW_DIM"])

        self.reviews = {}
        for h in range(1, N_HOTELS + 1):
            hotel_df = pd.read_csv(os.path.join(reviews_path, f"{h}.csv"),
                                   header=None)
            for review in hotel_df.iterrows():
                self.reviews[review[1][0]] = {"positive": review[1][2],
                                              "negative": review[1][3],
                                              "score": review[1][4]}
            self.reviews[-1] = {"positive": "",
                                "negative": "",
                                "score": 8}

    def __len__(self):
        return len(self.idx_to_group)

    def __getitem__(self, item):
        if isinstance(item, int):
            group = self.idx_to_group[item]
        else:
            group = item
        game = self.actions_df.get_group(group).reset_index()
        user_id = game["user_id"][0]
        n_rounds = len(game)

        game["is_sample"] = np.ones(n_rounds).astype(bool)
        if n_rounds < DATA_ROUNDS_PER_GAME:
            game = pd.concat([game] + [DATA_BLANK_ROW_DF(game["strategy_id"][0])] * (DATA_ROUNDS_PER_GAME - n_rounds),
                             ignore_index=True)

        bot_strategy = game["strategy_id"].to_numpy()

        hotels_scores = game["hotelScore"].to_numpy()

        action_taken = game["didGo"].to_numpy().astype(np.int64)
        is_hotel_good = (game["didGo"] == game["didWin"]).to_numpy()

        reaction_time = game["reaction_time"].to_numpy()
        last_reaction_time = game["last_reaction_time"].to_numpy()

        user_points = game["user_points"].to_numpy()
        bot_points = game["bot_points"].to_numpy()
        user_earned_more = user_points >= bot_points
        user_not_earned_more = user_points < bot_points

        reviewId = game["reviewId"]
        round_num = np.full(10, -1)
        round_num[:n_rounds] = np.arange(n_rounds)

        sample = {"user_id": user_id, "bot_strategy": bot_strategy, "n_rounds": n_rounds, "roundNum": round_num,
                  "hotels_scores": hotels_scores, "action_taken": action_taken, "is_hotel_good": is_hotel_good,
                  "reaction_time": reaction_time, "last_reaction_time": last_reaction_time,
                  "last_didGo_True": game["last_didGo_True"].to_numpy(),
                  "last_didWin_True": game["last_didWin_True"].to_numpy(),
                  "last_didGo_False": game["last_didGo_False"].to_numpy(),
                  "last_didWin_False": game["last_didWin_False"].to_numpy(),
                  "last_last_didGo_False": game["last_last_didGo_False"].to_numpy(),
                  "last_last_didGo_True": game["last_last_didGo_True"].to_numpy(),
                  "last_last_didWin_False": game["last_last_didWin_False"].to_numpy(),
                  "last_last_didWin_True": game["last_last_didWin_True"].to_numpy(),
                  "user_points": user_points / 10,
                  "bot_points": bot_points / 10, "user_earned_more": user_earned_more,
                  "user_not_earned_more": user_not_earned_more,
                  "review_vector": reviewId.apply(lambda r: self.review_reduced[r]).tolist(),
                  "is_sample": game["is_sample"].to_numpy(),
                  "weight": game["weight"].to_numpy(),
                  "action_id": game["index"].to_numpy()}

        for column_name, (lower, upper) in zip(reaction_time_columns_names, reaction_time_bins):
            sample[column_name] = (lower <= last_reaction_time) & (last_reaction_time < upper)

        sample["review_vector"] = torch.stack(sample["review_vector"])
        return sample


class OnlineSimulationDataSet(Dataset):
    def __init__(self, config):
        self.config = config
        simulation_th = SIMULATION_TH
        max_active = SIMULATION_MAX_ACTIVE_USERS
        favorite_topic_method = self.config["favorite_topic_method"]
        user_improve = config["simulation_user_improve"]
        signal_error = config["simulation_signal_error"]
        n_users = config["online_simulation_size"]
        basic_nature = config["basic_nature"]
        zero_knowledge = config["zero_knowledge"]
        weight_type = self.config["loss_weight_type"]
        self.bots_per_user = config["bots_per_user"]

        self.n_users = int(n_users / self.bots_per_user * 6)
        max_active = int(max_active / self.bots_per_user * 6)

        self.SIMULATION_TH = simulation_th
        self.SIMULATION_SIGNAL_EPSILON = signal_error
        self.users = defaultdict(list)
        self.n_games_per_user = {}
        self.active_users = []
        self.next_user = 0
        self.total_games_created = 0
        self.user_improve = user_improve
        self.zero_knowledge = zero_knowledge
        self.favorite_topic_method = favorite_topic_method
        self.weight_type = weight_type

        self.hotels = [np.array([0] * 7)]
        self.reviews_id = [np.array([0] * 7)]
        for hotel in range(1, N_HOTELS + 1):
            hotel_path = f"{DATA_GAME_REVIEWS_PATH}/{hotel}.csv"
            hotel_csv = pd.read_csv(hotel_path, header=None)
            self.hotels.append(hotel_csv.iloc[:, 4].to_numpy())
            self.reviews_id.append(hotel_csv.iloc[:, 0].to_numpy())
        self.hotels = np.array(self.hotels)
        self.reviews_id = np.array(self.reviews_id)

        self.review_reduced = pd.read_csv(config['FEATURES_PATH'], index_col=0).T.astype(int).to_dict(orient='list')
        self.review_reduced = {int(rid): torch.Tensor(vec) for rid, vec in self.review_reduced.items()}
        self.review_reduced[-1] = torch.zeros(config["REVIEW_DIM"])

        self.gcf = pd.read_csv(config['SIMULATION_EFs_PATH'], index_col=0, dtype=float).T
        self.gcf.columns = self.gcf.columns.astype(int)

        self.basic_nature = utils.basic_nature_options.pers[basic_nature]

        self.max_active = min(max_active, self.n_users)
        self.n_go = 0
        self.n_dont_go = 0

        pbar = trange(self.max_active)
        for i in pbar:
            self.new_user()
            pbar.set_description(f"Create online-simulation users for this epoch. "
                                 f"mean games/user: {(self.total_games_created / self.next_user):.2f}")

        self.add_to_user_id = DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS + DATA_CLEAN_ACTION_PATH_Y_NUMBER_OF_USERS

    class SimulatedUser:
        def __init__(self, user_improve, basic_nature, favorite_topic_method, **args):
            history_window = np.random.negative_binomial(2, 1 / 2) + np.random.randint(0, 2)
            quality_threshold = np.random.normal(8, 0.5)
            good_topics = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 19, 28, 42]
            good_topics = [str(t) for t in good_topics]
            bad_topics = [11, 20, 21, 22, 23, 24, 25, 26, 27, 36, 40]
            bad_topics = [str(t) for t in bad_topics]
            if favorite_topic_method == "random":
                positive_topics = np.random.choice(good_topics, 3)
                negative_topics = np.random.choice(bad_topics, 3)
            elif favorite_topic_method == "review":
                review_features = args['favorite_review']
                positive_topics = review_features[(review_features > 0) & review_features.index.isin(good_topics)].index
                positive_topics = np.array(positive_topics)
                np.random.shuffle(positive_topics)
                negative_topics = review_features[(review_features > 0) & review_features.index.isin(bad_topics)].index
                negative_topics = np.array(negative_topics)
                np.random.shuffle(negative_topics)

            self.ACTIONS = {0: ("correct - oracle", 0, user_strategies.correct_action),
                            1: ("random", basic_nature[0], user_strategies.random_action),
                            2: ("history_and_review_quality (Trustful)", basic_nature[1],
                                user_strategies.history_and_review_quality(history_window, quality_threshold)),
                            3: ("topic based", basic_nature[2], user_strategies.topic_based(positive_topics,
                                                                                            negative_topics,
                                                                                            quality_threshold)),
                            4: ("LLM stochastic (Language-based)",  basic_nature[3], user_strategies.LLM_based(is_stochastic=True)),
                            5: ("LLM deterministic", basic_nature[4], user_strategies.LLM_based(is_stochastic=False)),
                            6: ("hotel_review_model", basic_nature[5], user_strategies.hotel_review_model())  # TODO added strategy
                            }
            self.nature = np.random.rand(len(self.ACTIONS)) * np.array([v[1] for v in self.ACTIONS.values()])
            self.nature = self.nature / sum(self.nature)
            self.user_proba = self.nature.copy()
            self.user_improve = user_improve

        def return_to_init_proba(self):
            self.user_proba = self.nature.copy()

        def update_proba(self):
            reduce_feelings = np.random.rand(len(self.ACTIONS) - 1) * self.user_improve * 10/9 - (self.user_improve / 10)
            total_reduced = self.user_proba[1:] * reduce_feelings
            self.user_proba[1:] -= total_reduced
            self.user_proba[1:] = np.maximum(0, self.user_proba[1:])
            self.user_proba[0] = 1 - self.user_proba[1:].sum()

    def play_round(self, bot_message, user, previous_rounds, hotel, review_id):
        user_strategy = self.sample_from_probability_vector(user.user_proba)
        user_strategy_function = user.ACTIONS[user_strategy][2]
        review_features = self.gcf[review_id]
        information = {"bot_message": bot_message,
                       "previous_rounds": previous_rounds,
                       "hotel_value": hotel.mean(),
                       "review_features": review_features,
                       "review_id": review_id}
        user_action = user_strategy_function(information)
        return user_action

    @staticmethod
    def sample_from_probability_vector(probabilities):
        # Select a random number between 0 and 1
        random_num = np.random.rand()

        # Initialize a variable to keep track of the cumulative probability
        cumulative_probability = 0

        # Iterate through the probabilities
        for i, probability in enumerate(probabilities):
            # Add the probability to the cumulative probability
            cumulative_probability += probability

            # If the random number is less than the cumulative probability, return the index
            if random_num < cumulative_probability:
                return i

    def get_hotel(self, hotel_id=None):
        if hotel_id is None:
            hotel_id = np.random.randint(N_HOTELS) + 1
        hotel = self.hotels[hotel_id]
        return hotel_id, hotel

    def get_review_id(self, hotel_id, review_number):
        return self.reviews_id[hotel_id, review_number]

    def get_review(self, review_id=None):
        if review_id is None:
            review_id = np.random.choice(self.reviews_id[1:].flatten())
        return self.gcf[review_id]

    @staticmethod
    def bot_plays(bot_strategy, hotel, previous_rounds):
        bot_message = bot_strategy(hotel, previous_rounds)
        return bot_message

    @staticmethod
    def check_choice(hotel, action):
        return (hotel.mean() >= 8) == action

    def add_game(self, user, game):
        game = pd.DataFrame.from_records(game)
        self.users[user].append(game)

    def sample_bots(self):
        if self.zero_knowledge:
            return random.sample(range(DATA_N_BOTS), self.bots_per_user)
        else:
            if self.config["ENV_HPT_mode"]:
                return [3, 0, 2, 5, 19, 59]
            else:
                options = [3, 0, 2, 5, 19, 59] + [132, 23, 107, 43, 17, 93]
                return random.sample(options, self.bots_per_user)


    def new_user(self):
        user_id = self.next_user
        assert user_id < self.n_users
        args = {"favorite_review": self.get_review()}
        user = self.SimulatedUser(user_improve=self.user_improve, basic_nature=self.basic_nature,
                                  favorite_topic_method="review", **args)
        bots = self.sample_bots()
        game_id = 0
        for bot in bots:
            user.return_to_init_proba()
            bot_strategy = getattr(bot_strategies, f"strategy_{bot}")
            correct_answers = 0
            games_until_winning = 0
            while (correct_answers < self.SIMULATION_TH) and not (
                    self.user_improve == 0 and (games_until_winning == 100)):  # start a new game
                correct_answers = 0
                games_until_winning += 1
                previous_rounds = []
                game = []
                user_points, bot_points = 0, 0
                last_didGo, last_didWin = -1, -1
                last_last_didGo, last_last_didWin = -1, -1
                reaction_time = -1
                last_reaction_time = -1
                weight = 1
                for round_number in range(1, DATA_ROUNDS_PER_GAME + 1):  # start a new round
                    hotel_id, hotel = self.get_hotel()  # get a hotel

                    bot_message = self.bot_plays(bot_strategy, hotel, previous_rounds)  # expert plays
                    review_id = self.get_review_id(hotel_id, np.argmax(hotel == bot_message))

                    signal_error = np.random.normal(0, self.SIMULATION_SIGNAL_EPSILON)
                    user_action = self.play_round(bot_message + signal_error, user, previous_rounds,
                                                  hotel, review_id)  # DM plays
                    round_result = self.check_choice(hotel, user_action)  # round results
                    correct_answers += round_result

                    if user_action:
                        self.n_go += 1
                    else:
                        self.n_dont_go += 1

                    user.update_proba()  # update user vector
                    previous_rounds += [(hotel, bot_message, user_action)]

                    last_didGo_True = last_didGo == 1
                    last_didWin_True = last_didWin == 1
                    last_didGo_False = last_didGo == 0
                    last_didWin_False = last_didWin == 0

                    last_last_didGo_False = last_last_didGo == 0
                    last_last_didGo_True = last_last_didGo == 1
                    last_last_didWin_False = last_last_didWin == 0
                    last_last_didWin_True = last_last_didWin == 1

                    row = {"user_id": user_id, "strategy_id": bot, "gameId": game_id, "roundNum": round_number,
                           "hotelId": hotel_id, "reviewId": review_id, "hotelScore": float(f"{hotel.mean():.2f}"),
                           "didGo": user_action, "didWin": round_result, "correctAnswers": correct_answers,
                           "last_reaction_time": last_reaction_time,
                           "reaction_time": reaction_time,
                           "last_didWin_True": last_didWin_True, "last_didGo_True": last_didGo_True,
                           "last_didWin_False": last_didWin_False, "last_didGo_False": last_didGo_False,
                           "last_last_didGo_False": last_last_didGo_False, "last_last_didGo_True": last_last_didGo_True,
                           "last_last_didWin_False": last_last_didWin_False,
                           "last_last_didWin_True": last_last_didWin_True,
                           "user_points": user_points, "bot_points": bot_points, "weight": weight, "is_sample": True}

                    # if self.advanced_reaction_time:
                    #     last_reaction_time = self.get_reaction_time(row)

                    user_points += round_result
                    bot_points += user_action

                    last_last_didGo, last_last_didWin = last_didGo, last_didWin
                    last_didGo, last_didWin = int(user_action), int(round_result)
                    game.append(row)
                self.add_game(user_id, game)
                game_id += 1
        self.next_user += 1
        self.n_games_per_user[user_id] = game_id
        self.total_games_created += game_id
        self.active_users.append(user_id)

    def __len__(self):
        if self.next_user <= 50:
            return self.n_users * 42
        else:
            return int(self.n_users * self.total_games_created / self.next_user)

    def get_game(self, user_id):
        game = self.users[user_id].pop(0)
        if not len(self.users[user_id]):
            self.active_users.remove(user_id)
            if self.next_user < self.n_users:
                self.new_user()
        return game

    def __getitem__(self, user_id):
        game = self.get_game(user_id)

        user_id = game["user_id"][0] + self.add_to_user_id
        n_rounds = len(game)

        if n_rounds < DATA_ROUNDS_PER_GAME:
            game = pd.concat([game] + [DATA_BLANK_ROW_DF(game["strategy_id"][0])] * (DATA_ROUNDS_PER_GAME - n_rounds),
                             ignore_index=True)

        bot_strategy = game["strategy_id"].to_numpy()
        hotels_scores = game["hotelScore"].to_numpy()

        action_taken = game["didGo"].to_numpy()
        is_hotel_good = (game["didGo"] == game["didWin"]).to_numpy()

        p_go = self.n_go / (self.n_go + self.n_dont_go)
        odds = (1 - p_go) / p_go
        weight = np.where(game["didGo"].to_numpy(), odds, 1)

        reaction_time = game["reaction_time"].to_numpy()
        last_reaction_time = game["last_reaction_time"].to_numpy()

        user_points = game["user_points"].to_numpy()
        bot_points = game["bot_points"].to_numpy()
        user_earned_more = user_points >= bot_points
        user_not_earned_more = user_points < bot_points

        reviewId = game["reviewId"]

        round_num = np.full(10, -1)
        round_num[:n_rounds] = np.arange(n_rounds)
        action_ids = -np.ones_like(user_points)

        sample = {"user_id": user_id, "bot_strategy": bot_strategy, "n_rounds": n_rounds, "roundNum": round_num,
                  "hotels_scores": hotels_scores, "action_taken": action_taken, "is_hotel_good": is_hotel_good,
                  "reaction_time": reaction_time, "last_reaction_time": last_reaction_time,
                  "last_didGo_True": game["last_didGo_True"].to_numpy(),
                  "last_didWin_True": game["last_didWin_True"].to_numpy(),
                  "last_didGo_False": game["last_didGo_False"].to_numpy(),
                  "last_didWin_False": game["last_didWin_False"].to_numpy(),
                  "last_last_didGo_False": game["last_last_didGo_False"].to_numpy(),
                  "last_last_didGo_True": game["last_last_didGo_True"].to_numpy(),
                  "last_last_didWin_False": game["last_last_didWin_False"].to_numpy(),
                  "last_last_didWin_True": game["last_last_didWin_True"].to_numpy(),
                  "user_points": user_points / 10,
                  "bot_points": bot_points / 10, "user_earned_more": user_earned_more,
                  "user_not_earned_more": user_not_earned_more,
                  "review_vector": reviewId.apply(lambda r: self.review_reduced[r]).tolist(),
                  "is_sample": game["is_sample"].to_numpy(),
                  "weight": weight,
                  "action_id": action_ids}

        for column_name, (lower, upper) in zip(reaction_time_columns_names, reaction_time_bins):
            sample[column_name] = (lower <= last_reaction_time) & (last_reaction_time < upper)

        sample["review_vector"] = torch.stack(sample["review_vector"])

        return sample


class ConcatDatasets(IterableDataset):
    def __init__(self, dataloader1, dataloader2):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.iterator1 = iter(dataloader1)
        self.iterator2 = iter(dataloader2)

    def __iter__(self):
        return self

    def __next__(self):
        # Get next batch from both dataloaders
        batch1 = next(self.iterator1, None)
        batch2 = next(self.iterator2, None)

        # If either batch is None (i.e., we've exhausted the dataloader),
        # return the batch from the other dataloader
        if batch1 is None and batch2 is None:
            raise StopIteration
        if batch1 is None:
            return batch2
        if batch2 is None:
            return batch1

        # Concatenate batches along the batch dimension (assuming the first dimension is the batch dimension)
        # If the batches are dictionaries, this concatenates each item separately
        if isinstance(batch1, dict) and isinstance(batch2, dict):
            return {key: torch.cat((batch1[key], batch2[key])) for key in batch1}
        else:
            return torch.cat((batch1, batch2))

    def __len__(self):
        return max(len(self.dataloader1), len(self.dataloader2))
