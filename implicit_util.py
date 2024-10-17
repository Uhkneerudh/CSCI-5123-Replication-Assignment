# Inspired by: https://github.com/benfred/implicit/blob/main/examples/lastfm.py

import codecs
import logging
import time

import implicit.evaluation
import numpy as np
import pandas as pd
from tqdm import tqdm
from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (
    AnnoyAlternatingLeastSquares,
    NMSLibAlternatingLeastSquares,
)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (
    BM25Recommender,
    CosineRecommender,
    TFIDFRecommender,
    ItemItemRecommender
)

# maps command line model argument to class name
MODELS = {
    "als": AlternatingLeastSquares,
    "nmslib_als": NMSLibAlternatingLeastSquares,
    "annoy_als": AnnoyAlternatingLeastSquares,
    "tfidf": TFIDFRecommender,
    "cosine": CosineRecommender,
    "lmf": LogisticMatrixFactorization,
    "bm25": BM25Recommender,
    "ii": ItemItemRecommender
}


def get_model(model_name, params):
    print(f"getting model {model_name}")
    model_class = MODELS.get(model_name)
    if not model_class:
        raise ValueError(f"Unknown Model '{model_name}'")

    # some default params
    if not (model_name.endswith("als") or model_name == "bm25" or model_name == "bpr" or model_name == "lmf"):
        params = {"num_threads": 8}

    return model_class(**params)


def train_model(user_item_matrix, model_name, params):
    # create a model from the input data
    model = get_model(model_name, params)

    # if we're training an ALS based model, disable building approximate recommend index
    if model_name.endswith("als"):
        model.approximate_recommend = False
        model.approximate_similar_items = False

    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(user_item_matrix)
    logging.debug("trained model '%s' in %0.2fs", model_name, time.time() - start)

    return model


def calculate_similar_items(items, output_filename, model):
    """generates a list of similar items by utilizing the 'similar_items' api of the models"""
    similar_items = []
    start = time.time()

    # write out as a TSV of item_id, other_item_id, score
    logging.debug("writing similar items")
    with tqdm(total=len(items)) as progress:
        with codecs.open(output_filename, "w", "utf8") as o:
            batch_size = 1000
            for startidx in range(0, len(items), batch_size):
                batch = items[startidx: startidx + batch_size]
                ids, scores = model.similar_items(batch, 11)
                for i, item_id in enumerate(batch):
                    for other_id, score in zip(ids[i], scores[i]):
                        o.write(f"{item_id}\t{other_id}\t{score}\n")
                        similar_items.append((item_id, other_id, score))
                progress.update(batch_size)

    logging.debug("generated similar items in %0.2fs", time.time() - start)
    return pd.DataFrame(similar_items, columns=['item_id', 'other_item_id', 'score'])


def calculate_similar_users(users, output_filename, model):
    """generates a list of similar users by utilizing the 'similar_users' api of the models"""
    similar_users = []
    start = time.time()

    # write out as a TSV of user_id, other_user_id, score
    try:
        logging.debug("writing similar users")
        with tqdm(total=len(users)) as progress:
            with codecs.open(output_filename, "w", "utf8") as o:
                batch_size = 1000
                for startidx in range(0, len(users), batch_size):
                    batch = users[startidx: startidx + batch_size]
                    ids, scores = model.similar_users(batch, 11)
                    for i, user_id in enumerate(batch):
                        for other_id, score in zip(ids[i], scores[i]):
                            o.write(f"{user_id}\t{other_id}\t{score}\n")
                            similar_users.append((user_id, other_id, score))
                    progress.update(batch_size)

        logging.debug("generated similar users in %0.2fs", time.time() - start)
    except NotImplementedError:
        print("Current model does not support the method similar_users()")
    return pd.DataFrame(similar_users, columns=['user_id', 'other_user_id', 'score'])


def calculate_recommendations(users, output_filename, model, csr_matrix):
    """Generates item recommendations for each user in the dataset"""
    recommendations = []

    # generate recommendations for each user and write out to a file
    start = time.time()
    with tqdm(total=len(users)) as progress:
        with codecs.open(output_filename, "w", "utf8") as o:
            batch_size = 1000
            for startidx in range(0, len(users), batch_size):
                batch = users[startidx: startidx + batch_size]
                ids, scores = model.recommend(
                    batch, csr_matrix[batch], filter_already_liked_items=True
                )
                for i, user_id in enumerate(batch):
                    for item_id, score in zip(ids[i], scores[i]):
                        o.write(f"{user_id}\t{item_id}\t{score}\n")
                        recommendations.append((user_id, item_id, score))
                progress.update(batch_size)

    logging.debug("generated recommendations in %0.2fs", time.time() - start)
    return pd.DataFrame(recommendations, columns=['user_id', 'item_id', 'score'])


def tuple_to_unique(tuple):
    a, b = tuple
    a = np.unique(a)
    b = np.unique(b)
    return a, b

#NEW CODE: Count # of hits
def hits(model, train_matrix, test_matrix, K):

    #NOTE: Code is based on the ranking_metrics_at_k function from the implicit GitHub: https://github.com/benfred/implicit/blob/main/implicit/evaluation.pyx
    users = test_matrix.shape[0] 
    items = test_matrix.shape[1]
    u = 0
    i = 0
    batch_idx = 0

    batch_size = 1000
    start_idx = 0

    test_indptr = test_matrix.indptr
    test_indices = test_matrix.indices

    total = 0
    hits = 0

    ids = []
    batch = []
    likes = set()
    to_generate = np.arange(users, dtype="int32")
    to_generate = to_generate[np.ediff1d(test_matrix.indptr) > 0]
    progress = tqdm(total=len(to_generate), disable=False)

    while start_idx < len(to_generate):
        batch = to_generate[start_idx: start_idx + batch_size]
        ids, _ = model.recommend(batch, train_matrix[batch], N=K)
        start_idx += batch_size
        for batch_idx in range(len(batch)):
            u = batch[batch_idx]
            likes.clear()
            for i in range(test_indptr[u], test_indptr[u+1]):
                likes.add(test_indices[i])

            num_pos_items = len(likes)
            num_neg_items = items - num_pos_items

            for i in range(K):
                if ids[batch_idx, i] in likes:
                    #increase number of hits
                    hits += 1
            total += 1            
        progress.update(len(batch))
    progress.close()
    #return hit ratio (number of hits divided by total)
    return hits / total
        
#NEW CODE ADDED IN THIS FUNCTION
def calc_scores(mode_name, model, train, test, K=10):
    # Evaluation (@K=10 --> for the best K items):
    # Normalized Discounted Cumulative Gain (NDCG), Mean Average Precision (MAP), Precision (P),
    # Area Under ROC Curve (AUC)
    # --> https://flowthytensor.medium.com/some-metrics-to-evaluate-recommendation-systems-9e0cf0c8b6cf

    ndcg_score = implicit.evaluation.ndcg_at_k(model, train, test, K=K, show_progress=False)
    map_score = implicit.evaluation.mean_average_precision_at_k(model, train, test, K=K, show_progress=False)
    p_score = implicit.evaluation.precision_at_k(model, train, test, K=K, show_progress=False)

    #NEW CODE -- Calculate AUC Score and Hits Score
    auc_score = implicit.evaluation.AUC_at_k(model, train, test, K=K, show_progress=False)
    hits_score = hits(model, train, test, K=K)

    # Show scores (NEW Code - Add Hits)
    evaluation = pd.DataFrame([[mode_name, ndcg_score, map_score, p_score, auc_score,hits_score]],
                              columns=['model', 'ndcg@' + str(K), 'map@' + str(K), 'p@' + str(K), 'auc@' + str(K),'hits@' + str(K)])

    return evaluation


def print_recommendations(recommendations, recipes_df):
    print(f'You might also like:')
    for i, rec in enumerate(recommendations['item_id']):
        print(recipes_df[recipes_df['new_recipe_id'] == rec].iloc[0]['title'])


# TODO use cross validation and try different params
def generate_params():
    return 0


#CODE MODIFIED IN THIS FUNCTION
def train_and_execute_all(train_matrix, test_matrix, train_user, train_recipe, excluded_models=[], factors=8, regularization=0.1, K1=100, B=0.5, K=100):
    model_params = {
        "als": {"factors": factors, "regularization": regularization, "iterations": 30, "num_threads": 8}, #8, 0.1
        "nmslib_als": {"num_threads": 8},
        "lmf": {"factors": factors, "iterations": 40, "regularization": regularization, "num_threads": 8}, #30, 1.5
        "bm25": {"K1": K1, "B": B, "num_threads": 8},
        "cosine": {"num_threads": 8},
        "tfidf": {"num_threads": 8},
        "ii": {"num_threads": 8},
        "annoy_als": {"num_threads": 8},
    }

    #MODIFIED CODE: Add hits and auc to evaluation dataframe
    evaluation = pd.DataFrame(columns=['model', 'ndcg@' + str(K), 'map@' + str(K), 'p@' + str(K), 'auc@' + str(K),'hits@' + str(K)])
    recommendations = {}
    similar_items = {}
    similar_users = {}

    for model_name in MODELS:
        if model_name not in excluded_models:
            # Train model
            params = model_params[model_name]
            model = train_model(train_matrix, model_name, params)

            # Evaluation: Get relevant scores
            current_evaluation = calc_scores(model_name, model, train_matrix, test_matrix, K=K)

            #Evaluation

            # Calculate recommendations for users and similar items/recipes
            current_recommendations = calculate_recommendations(train_user,
                                                                'output/' + model_name + '_recommendations.tsv',
                                                                model, train_matrix)
            current_similar_items = calculate_similar_items(train_recipe,
                                                            'output/' + model_name + '_similar_items.tsv', model)
            current_similar_users = calculate_similar_users(train_user,
                                                            'output/' + model_name + '_similar_users.tsv', model)

            # Add results
            evaluation = pd.concat([evaluation, current_evaluation])
            recommendations[model_name] = current_recommendations
            similar_items[model_name] = current_similar_items
            similar_users[model_name] = current_similar_users

    # Add columns on evaluation to describe the model
    evaluation = evaluation.set_index('model')

    # Highlight best values for each column/metrix
    evaluation.style.highlight_max(color='lightgreen', axis=0)

    return evaluation, recommendations, similar_items, similar_users

