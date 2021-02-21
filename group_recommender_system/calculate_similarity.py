import pandas as pd
import networkx as nx
from collections import defaultdict
from .graph_funcs import get_nodes_of_type
import statistics

def shared_neighbors(G, user1, user2):
    nbrs1 = G.neighbors(user1)
    nbrs2 = G.neighbors(user2)

    overlap = set(nbrs1).intersection(nbrs2)
    return overlap

def user_similarity(dict1, dict2):
    dif = []
    for key, val in dict1.items():
        if key in dict2:
            dif.append(abs(val - dict2[key])/5)
    return 1 - statistics.mean(dif)

def restaurant_similarity(G, rest1, rest2):
    shared_nodes = shared_neighbors(G, rest1, rest2)
    shared_nodes = list(filter(lambda x: G.nodes[x]['type'] == 'category', shared_nodes))

    nbrs1 = G.neighbors(rest1)
    nbrs2 = G.neighbors(rest2)
    total = set(nbrs1).union(set(nbrs2))
    total = list(filter(lambda x: G.nodes[x]['type'] == 'category', total))
    return len(shared_nodes) / len(total)

def category_preference_list(G, user):
    ratings = {}
    for r in G.neighbors(user): 
        #print("*", G.nodes[user]['name'], "-", G.nodes[r]['name'], "-",  G.get_edge_data(test_user, r))
        for c in G.neighbors(r): 
            if G.nodes[c]['type'] == 'category':
                if c in ratings:
                    ratings[c].append(G.get_edge_data(user, r)['rating'])
                else:
                    ratings[c] = [G.get_edge_data(user, r)['rating']]

    for k, v in ratings.items():
        ratings[k] = statistics.mean(v)
    return ratings

def most_similar_users(G, user):
    similar_users = {}
    category_ratings = category_preference_list(G, user)

    for r in G.neighbors(user):
        for c in G.neighbors(r): 
            if G.nodes[c]['type'] == 'user' and c != user and c not in similar_users:
                similar_users[c] = category_preference_list(G, c)
    
    similarities = defaultdict(list)
    for user, preferences in similar_users:
        similarity = user_similarity(category_ratings, preferences)
        similarities[similarity].append(user)

    max_similarity = max(similarities.keys(), 0)
    if(max_similarity == 0): return []

    return similarities[max_similarity]

def recommend_restaurants(G, from_user, to_user):
    from_rests = set(G.neighbors(from_user))
    to_rests = set(G.neighbors(to_user))

    return from_rests.difference(to_rests)

def predict_category_ratings(G, user, similar_users):
    category_ratings = category_preference_list(G, user)
    rating_predictions = {}
    for u in similar_users: 
        for key, val in category_preference_list(G, u).items():
            if key not in category_ratings:
                if key in rating_predictions:
                    rating_predictions[key].append(val)
                else:
                    rating_predictions[key] = [val]
    
    for key, val in rating_predictions.items():
        category_ratings[key] = statistics.mean(val)

    return category_ratings

def predict_restaurant_ratings(G, category_ratings, restaurant):
    ratinglist = []
    for cat in G.neighbors(restaurant):
        if G.nodes[cat]['type'] == 'category':
            ratinglist.append(category_ratings[cat])
            
    return statistics.mean(ratinglist)