"""

drs_network

Digital Research Seminar - Group-making & digital culture
Author: Bharath Ganesh, b.ganesh@uva.nl, University of Amsterdam
Week 4: Networks

This is a package of helper functions used in the Week 4 Hashtag Networks Notebook

Note that you also have to use Gephi to calculate proximity (or do any network statistics)!

"""

import pandas as pd
import re
from collections import Counter

def create_hashtag2hashtag_network(docs, save_name = "myfile.csv"):
    edges = list()
    pattern = "(#+[A-Za-zÀ-ÖØ-öø-ÿ0-9_(_)]{1,})"
    for doc in docs:
        hashtags =  re.findall(pattern, doc)
        for i in range(len(hashtags)):
            for j in range(i + 1, len(hashtags)):
                edge = (hashtags[i], hashtags[j])
                edges.append(edge)
    edgelist = pd.DataFrame.from_records(edges, columns = ["source","target"])
    edgelist.to_csv(save_name, index=False)
    print("Network file saved. Open in Gephi for further processing using File | Import Spreadsheet. This is an undirected edgelist.")
    return edges
    
def create_user2hashtag_network(df, user_column = "username", doc_column = "caption", save_name = "myfile.csv"):
    edges = list()
    pattern = "(#+[A-Za-zÀ-ÖØ-öø-ÿ0-9_(_)]{1,})"
    for user, doc in zip(list(df[user_column]), list(df[doc_column])):
        hashtags = re.findall(pattern, doc)
        if len(hashtags) > 0:
            for hashtag in hashtags:
                edges.append((user, hashtag))
    edgelist = pd.DataFrame.from_records(edges, columns = ["source","target"])
    edgelist.to_csv(save_name, index=False)
    print("Network file saved. Open in Gephi for further processing using File | Import Spreadsheet. This is a directed edgelist.")
    return edges

def calculate_proximity(mc_a, mc_b, edge_counter, E):
    """
    Proximity = A -> B observed (n) / total A
    
    """
    
    tA = E[(E["source_modularity_class"] == mc_a)|(E["target_modularity_class"] == mc_a)].shape[0]
    n = edge_counter[(mc_a, mc_b)]
    p = n / tA if tA != 0 else 0  # Avoid division by zero
    return p

def create_proximity_matrix(gephi_nodes_table = "./Modularity_Example.csv", edgelist = "user2hashtag.csv", save_name = "partition_matrix.csv"):
    """
    
    Replicates the proximity matrix proposed by Freelon (2018), see course reading
    Freelon's other metrics can be replicated with Excel using a GUI.
    NOTE: Gephi Nodes Table MUST be exported from Gephi and must include a column "modularity_class", written exactly as such.
    NOTE: Edgelist is a table created by "create_hashtag2hashtag_network" and "create_user2hashtag_network"
    
    """
    # load the node and edge table
    N = pd.read_csv(gephi_nodes_table)
    E = pd.read_csv(edgelist)
    community_index = {k: v for k, v in zip(list(N["Id"]), list(N["modularity_class"]))}
    
    # transform E to include modularity information (so we can have "source_modularity_class" and "target_modularity_class" as values to count
    E["source_modularity_class"] = E["source"].apply(lambda x: community_index[x])
    E["target_modularity_class"] = E["target"].apply(lambda x: community_index[x])
    
    # create a counter to count the pairs (we use this to calculate proximity)
    edge_counter = Counter([(a,b) for a, b in zip(list(E["source_modularity_class"]), list(E["target_modularity_class"]))])
    
    # create proximity matrix using Freelon's metric
    modularity_classes = set(E['source_modularity_class']).union(set(E['target_modularity_class']))
    X = pd.DataFrame(index=list(modularity_classes), columns=list(modularity_classes))
    for mc_a in modularity_classes:
        for mc_b in modularity_classes:
            X.at[mc_a, mc_b] = round(calculate_proximity(mc_a, mc_b, edge_counter, E), 4)
    
    E.to_csv(edgelist, index=False)
    X.to_csv(save_name)
    
    print("Proximity Matrix created!")
    
    return X