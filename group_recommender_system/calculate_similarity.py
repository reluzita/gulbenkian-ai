import pandas as pd
import networkx as nx
from collections import defaultdict
from .graph_funcs import get_nodes_of_type

def shared_neighbors(G, user1, user2):
    nbrs1 = G.neighbors(user1)
    nbrs2 = G.neighbors(user2)

    overlap = set(nbrs1).intersection(nbrs2)
    return overlap

def user_similarity(G, user1, user2):
    shared_nodes = shared_neighbors(G, user1, user2)

    #nbrs1 = G.neighbors(user1)
    #nbrs2 = G.neighbors(user2)
    #total = set(nbrs1).union(set(nbrs2))
    return len(shared_nodes) / len(get_nodes_of_type(G, 'restaurant'))

def restaurant_similarity(G, rest1, rest2):
    shared_nodes = shared_neighbors(G, rest1, rest2)
    shared_nodes = list(filter(lambda x: G.nodes[x]['type'] == 'category', shared_nodes))

    nbrs1 = G.neighbors(rest1)
    nbrs2 = G.neighbors(rest2)
    total = set(nbrs1).union(set(nbrs2))
    total = list(filter(lambda x: G.nodes[x]['type'] == 'category', total))
    return len(shared_nodes) / len(total)

def most_similar_users(G, user):
    nbrs = G.neighbors(user)

    user_nodes = []
    for r in nbrs:
        user_nbrs = G.neighbors(r)
        user_nbrs = list(filter(lambda x: G.nodes[x]['type'] == 'user', user_nbrs))
        for u in user_nbrs:
            user_nodes.append(u)
    
    user_nodes = set(user_nodes)

    similarities = defaultdict(list)
    for n in user_nodes:
        similarity = user_similarity(G, user, n)
        similarities[similarity].append(n)

    max_similarity = max(similarities.keys())
    if(max_similarity == 0): return []

    return similarities[max_similarity]

def recommend_restaurants(G, from_user, to_user):
    from_rests = set(G.neighbors(from_user))
    to_rests = set(G.neighbors(to_user))

    return from_rests.difference(to_rests)