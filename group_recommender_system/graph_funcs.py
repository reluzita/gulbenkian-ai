import networkx as nx
import pandas as pd

def build_graph(rest_df, user_df, review_df):
    G = nx.Graph()

    categories = []

    # add restaurant nodes
    for index, row in rest_df.iterrows():
        G.add_node(row['business_id'], type="restaurant", name=row['name'])

        # add category nodes and edges between those and restaurants
        cat = row['categories'].split(", ")
        for c in cat:
            if c not in categories:
                G.add_node(c, type="category")
                categories.append(c)
            G.add_edge(row['business_id'], c)

    # add user nodes
    for index, row in user_df.iterrows():
        G.add_node(row['user_id'], type="user", name=row['name'])
         
    # add edges between users and the restaurants they reviewed
    for index, row in review_df.iterrows():
        G.add_edge(row['user_id'], row['business_id'], rating=row['stars'])

    return G

def get_nodes_of_type(G, node_type):
    res = list()
    for node, data in G.nodes(data=True):
        if data['type'] == node_type:
            res.append(node)
    return res