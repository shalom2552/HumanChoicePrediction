import numpy as np
import json

################################
# CONSTS
################################

REVIEWS = 0
BOT_ACTION = 1
USER_DECISION = 2


################################

def correct_action(information):
    if information["hotel_value"] >= 8:
        return (1, 'oracle')
    else:
        return (0, 'oracle')


def random_action(information):
    return np.random.randint(2)


def user_rational_action(information):
    if information["bot_message"] >= 8:
        return 1
    else:
        return 0


def user_picky(information):
    if information["bot_message"] >= 9:
        return (1, 'oracle')
    else:
        return (0, 'oracle')


def user_sloppy(information):
    if information["bot_message"] >= 7:
        return 1
    else:
        return 0


def user_short_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or (information["previous_rounds"][-1][BOT_ACTION] >= 8 and
                information["previous_rounds"][-1][REVIEWS].mean() >= 8) \
            or (information["previous_rounds"][-1][BOT_ACTION] < 8 and
                information["previous_rounds"][-1][REVIEWS].mean() < 8):  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def user_picky_short_t4t(information):
    if information["bot_message"] >= 9 or ((information["bot_message"] >= 8) and (
            len(information["previous_rounds"]) == 0 or (
            information["previous_rounds"][-1][REVIEWS].mean() >= 8))):  # good hotel
        return 1
    else:
        return 0


def user_hard_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                 or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                information["previous_rounds"]])) == 1:  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def history_and_review_quality(history_window, quality_threshold):
    def func(information):
        if len(information["previous_rounds"]) == 0 \
                or history_window == 0 \
                or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                     or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                    information["previous_rounds"][
                                    -history_window:]])) == 1:  # cooperation from *result's* perspective
            if information["bot_message"] >= quality_threshold:  # good hotel from user's perspective
                return 1
            else:
                return 0
        else:
            return 0
    return func


def topic_based(positive_topics, negative_topics, quality_threshold):
    def func(information):
        review_personal_score = information["bot_message"]
        for rank, topic in enumerate(positive_topics):
            review_personal_score += int(information["review_features"].loc[topic])*2/(rank+1)
        for rank, topic in enumerate(negative_topics):
            review_personal_score -= int(information["review_features"].loc[topic])*2/(rank+1)
        if review_personal_score >= quality_threshold:  # good hotel from user's perspective
            return 1
        else:
            return 0
    return func


def LLM_based(is_stochastic):
    with open(f"data/baseline_proba2go.txt", 'r') as file:
        proba2go = json.load(file)
        proba2go = {int(k): v for k, v in proba2go.items()}

    if is_stochastic:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(np.random.rand() <= review_llm_score)
        return func
    else:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(review_llm_score >= 0.5)
        return func
        

def hotel_review_model():
    baseline_path = './data/new_baseline.txt'

    with open(baseline_path, 'r') as file:
        predicted_rating = json.load(file)
        predicted_rating = {int(k): v for k, v in predicted_rating.items()}
    
    def func(information):
        review_llm_score = predicted_rating.get(information["review_id"], 0)
        return int(review_llm_score >= 8)
    
    return func
    

def fetch_sentiment_confidence(file_location):
    scores = {}
    with open(file_location, 'r', encoding='utf-8') as f:
        for entry in f:
            data = entry.strip().split('\t')
            review_identifier = int(data[0])
            sentiment_value = data[3]
            confidence_level = float(data[4])
            scores[review_identifier] = {
                "sentiment": sentiment_value,
                "confidence": confidence_level
            }
    return scores

def confidence_based_decision(threshold_value, backup_strategy):
    def decision_maker(info):
        review_key = info["review_id"]
        sentiment_scores = fetch_sentiment_confidence('sentiment_confidence_results.txt')
        
        # If the review exists in the sentiment scores
        if review_key in sentiment_scores:
            score_data = sentiment_scores[review_key]
            confidence_level = score_data["confidence"]
            sentiment_type = score_data["sentiment"]
            
            # Check if confidence exceeds threshold
            if confidence_level >= threshold_value:
                # Return based on sentiment
                if sentiment_type == "positive":
                    return (1, 'sentiment_based')
                else:
                    return (0, 'sentiment_based')
        
        # Use backup strategy if confidence is too low
        return backup_strategy(info)
    
    return decision_maker
