# Data manipulation
import pandas as pd
import random
import json
import os
import pickle
import time
# DOC2VEC
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
nltk.download('punkt') # Download NLTK tokenizer data

# Example documents

# Wikipedia API
import wikipedia as wp
from wikipedia.exceptions import DisambiguationError, PageError

# Plotting
import networkx as nx
import matplotlib.pyplot as plt

#Parsing args
import argparse

#Converting to PyG
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx

# print(wp.page("data science").links[:100])

#@title
class RelationshipGenerator():
    """Generates relationships between terms, based on wikipedia links"""
    def __init__(self, save_dir):
        """Links are directional, start + end, they should also have a weight"""
        self.links = [] # [start, end, weight]
        self.features = {} #{page: page_content}
        self.page_links = {}
        self.save_dir = save_dir
        with open(os.path.join(self.save_dir, "features.json"), "r+") as fp:
            self.features = json.load(fp)
        with open(os.path.join(self.save_dir, "page_links.json"), "r+") as fp:
            self.page_links = json.load(fp)
        print("Got memoized features ", self.features.keys())


    def scan(self, start=None, repeat=0):
        print("On depth: ", repeat)
        """Start scanning from a specific word, or from internal database
        
        Args:
            start (str): the term to start searching from, can be None to let
                algorithm decide where to start
            repeat (int): the number of times to repeat the scan
        """
        while repeat >= 0:

            # should check if start page exists
            # and haven't already scanned
            # if start in [l[0] for l in self.links]:
            #     raise Exception("Already scanned")
            
            #iteratively saves in case we get throttled

            term_search = True if start is not None else False

            # If a start isn't defined, we should find one
            if start is None: 
                start = self.find_starting_point()

            # Scan the starting point specified for links
            print(f"Scanning page {start}...")
            try:
                # Fetch the page through the Wikipedia API
                page = wp.page(start)
                self.features[start] = page.content
                print(self.features.keys())
                links = list(set(page.links))
                # ignore some uninteresting terms
                links = [l for l in links if not self.ignore_term(l)]

                # Add links to database
                link_weights = []
                for link in links:
                    weight = self.weight_link(page, link)
                    link_weights.append(weight)
                
                link_weights = [w / max(link_weights) for w in link_weights]

                #add the links
                for i, link in enumerate(links):
                    if i % 10 == 0:
                        with open(os.path.join(self.save_dir, "links.npy"), "wb+") as fp:
                            np.save(fp, np.array(self.links))
                        with open(os.path.join(self.save_dir, "features.json"), "w+") as fp:
                            fp.write(json.dumps(self.features))
                        with open(os.path.join(self.save_dir, "page_links.json"), "w+") as fp:
                            fp.write(json.dumps(self.page_links))
                    try:
                        link = link.lower()
                        if link not in self.features or link not in self.page_links:
                            time.sleep(np.random.randint(0, 10))
                            page = wp.page(link)
                            self.features[link] = page.content
                            self.page_links[link] = [l.lower() for l in page.links]
                            print("GOT IT: ", link)
                        else:
                            print("HAVE IT: ", link)
                        total_nodes = set([l[1].lower() for l in self.links])
                        for links_to in set([l.lower() for l in self.page_links[link]]).intersection(total_nodes):
                            self.links.append([link, links_to, 0.1]) # 3 works pretty well
                        print("INTERSECTION COUNT: ", len(set([l.lower() for l in self.page_links[link]]).intersection(total_nodes)))
                        self.links.append([start, link, link_weights[i] + 2 * int(term_search)]) # 3 works pretty well
                    except (DisambiguationError, PageError):
                        print("ERROR, I DID NOT GET THIS PAGE: ", link)
                

                # Print some data to the user on progress
                explored_nodes = set([l[0] for l in self.links])
                explored_nodes_count = len(explored_nodes)
                total_nodes = set([l[1] for l in self.links])
                total_nodes_count = len(total_nodes)
                new_nodes = [l.lower() for l in links if l not in total_nodes]
                new_nodes_count = len(new_nodes)
                print(f"New nodes added: {new_nodes_count}, Total Nodes: {total_nodes_count}, Explored Nodes: {explored_nodes_count}")

            except (DisambiguationError, PageError):
                # This happens if the page has disambiguation or doesn't exist
                # We just ignore the page for now, could improve this
                # self.links.append([start, "DISAMBIGUATION", 0])
                print("ERROR, I DID NOT GET THIS PAGE")
                pass

            repeat -= 1
            start = None
        
    def find_starting_point(self):
        """Find the best place to start when no input is given"""
        # Need some links to work with.
        if len(self.links) == 0:
            raise Exception("Unable to start, no start defined or existing links")
                
        # Get top terms
        res = self.rank_terms()
        sorted_links = list(zip(res.index, res.values))
        all_starts = set([l[0] for l in self.links])

        # Remove identifiers (these are on many Wikipedia pages)
        all_starts = [l for l in all_starts if '(identifier)' not in l]
        
        # print(sorted_links[:10])
        # Iterate over the top links, until we find a new one
        for i in range(len(sorted_links)):
            if sorted_links[i][0] not in all_starts and len(sorted_links[i][0]) > 0:
                return sorted_links[i][0]
        
        # no link found
        raise Exception("No starting point found within links")
        return

    @staticmethod
    def weight_link(page, link):
        """Weight an outgoing link for a given source page
        
        Args:
            page (obj): 
            link (str): the outgoing link of interest
        
        Returns:
            (float): the weight, between 0 and 1
        """
        weight = 0.1
        
        link_counts = page.content.lower().count(link.lower())
        weight += link_counts
        
        if link.lower() in page.summary.lower():
            weight += 3
        
        return weight


    def get_database(self):
        return sorted(self.links, key=lambda x: -x[2])


    def rank_terms(self, with_start=True):
        # We can use graph theory here!
        # tws = [l[1:] for l in self.links]
        df = pd.DataFrame(self.links, columns=["start", "end", "weight"])
        
        if with_start:
            df = df.append(df.rename(columns={"end": "start", "start":"end"}))
        
        return df.groupby("end").weight.sum().sort_values(ascending=False)
    
    def get_key_terms(self, n=20):
        return "'" + "', '".join([t for t in self.rank_terms().head(n).index.tolist() if "(identifier)" not in t]) + "'"

    @staticmethod
    def ignore_term(term):
        """List of terms to ignore"""
        if "(identifier)" in term or term == "doi":
            return True
        return False
    

def simplify_graph(rg, max_nodes=1000):
    """Simplify a graph which has many nodes
    
    Remove items with low total weights
    This is an alterantive to restricted_view in networkx.
    
    Args:
        rg (RelationshipGenerator): object containing knowledge graph
        max_nodes (float): the number of nodes to search, or percentage of nodes
            to keep
    
    Returns:
        (RelationshipGenerator): simplified knowledge graph
    """
    # Get most interesting terms.
    nodes = rg.rank_terms()

    # Get nodes to keep
    if max_nodes >= 1:
        keep_nodes = nodes.head(max_nodes).index.tolist()
    elif max_nodes >= 0:
        keep_nodes = nodes.head(int(max_nodes * len(nodes))).index.tolist()
    
    # Filter list of nodes so that there are no nodes outside those of interest
    filtered_links = list(filter(lambda x: x[1] in keep_nodes, rg.links))
    filtered_links = list(filter(lambda x: x[0] in keep_nodes, filtered_links))

    # Define a new object and define its dictionary
    ac = RelationshipGenerator()
    ac.links = filtered_links

    return ac

#@title
def remove_self_references(l):
    return [i for i in l if i[0]!=i[1]]

# def add_focus_point(links, focus="on me", focus_factor=3):
#     for i, link in enumerate(links):
#         if not (focus in link[0] or focus in link[1]):
#             links[i] = [link[0], link[1], link[2] / focus_factor]
#         else:
#             links[i] = [link[0], link[1], link[2] * focus_factor]

#     return links

def recover_graph(save_dir=None):
    features = {}
    links = []

    with open("../dataset/model/features.json", "r+") as fp:
        features = json.load(fp)
    links = np.load("../dataset/model/links.npy")

    print("Loaded")

    rg = RelationshipGenerator(save_dir=save_dir)
    rg.links = links
    links = remove_self_references(links)

    print("Loaded into RG", rg)

    node_data = rg.rank_terms()
    nodes = node_data.index.tolist()
    node_weights = node_data.values.tolist()
    node_weights = [nw * 100 for nw in node_weights]
    nodelist = nodes

    print("Node info gathered")


    G = nx.DiGraph() # MultiGraph()

    print("Created graph G")
 
    # G.add_node()
    G.add_nodes_from(nodes)

    print("Added nodes G")
    # List of tuples page title, page content
    features = dict(filter(lambda x: x[0] in nodes, rg.features.items()))
    features = sorted(features.items(), key=lambda key_value: nodes.index(key_value[0]))

    print("New Features created")
    tokenized_docs = [nltk.word_tokenize(' '.join(doc).lower()) for doc in features]
    tagged_docs = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(tokenized_docs)]
    print("Tagged docs")
    # Model 
    model = Doc2Vec(vector_size=300, min_count=1, epochs=50)
    model.build_vocab(tagged_docs)
    print("Training Model")
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    print("Trained Model!")
    feature_vectors = {node: model.infer_vector(tokenized_docs[i]) for i, node in enumerate(nodes)}
    nx.set_node_attributes(G, feature_vectors, name="features")
    print("Set node attributes")

    # Add edges
    G.add_weighted_edges_from(links)
    print("Recovered: ", G)
    return G, nodelist, node_weights, model

def create_graph(topics=["tests"], depth=20, max_size=20, simplify=False, plot=False, save_dir=None):

    rg = RelationshipGenerator(save_dir=save_dir)
    
    for topic in topics:
        rg.scan(topic)

    rg.scan(repeat=depth)

    print(f"Created {len(rg.links)} links with {rg.rank_terms().shape[0]} nodes.")

    if simplify:
        rg = simplify_graph(rg, max_size)

    links = rg.links
    links = remove_self_references(links)
    
    node_data = rg.rank_terms()
    nodes = node_data.index.tolist()
    node_weights = node_data.values.tolist()
    node_weights = [nw * 100 for nw in node_weights]
    nodelist = nodes


    G = nx.DiGraph() # MultiGraph()
 
    # G.add_node()
    G.add_nodes_from(nodes)
    # List of tuples page title, page content
    features = dict(filter(lambda x: x[0] in nodes, rg.features.items()))
    features = sorted(rg.features.items(), key=lambda key_value: nodes.index(key_value[0]))
    tokenized_docs = [nltk.word_tokenize(' '.join(doc).lower()) for doc in features]
    tagged_docs = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(tokenized_docs)]
    # Model 
    model = Doc2Vec(vector_size=300, min_count=1, epochs=50)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    feature_vectors = {node: model.infer_vector(tokenized_docs[i]) for i, node in enumerate(nodes)}
    nx.set_node_attributes(G, feature_vectors, name="features")

    # Add edges
    G.add_weighted_edges_from(links)
    return G, nodelist, node_weights, model

#@title
def simplified_plot(G, nodelist, node_weights):
    pos = nx.spring_layout(G, k=1, seed=7)  # positions for all nodes - seed for reproducibility

    fig = plt.figure(figsize=(12,12))

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodelist,
        node_size=node_weights,
        node_color='lightblue',
        alpha=0.7
    )

    widths = nx.get_edge_attributes(G, 'weight')    
    nx.draw_networkx_edges(
        G, pos,
        edgelist = widths.keys(),
        width=list(widths.values()),
        edge_color='lightblue',
        alpha=0.6
    )

    nx.draw_networkx_labels(G, pos=pos,
                            labels=dict(zip(nodelist,nodelist)),
                            font_color='black')
    fig = plt.show()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, help="How deep the search should go")
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--simplify', action='store_true')
    parser.add_argument('--search', action='append')
    parser.add_argument('--save-dir', type=str, default="../dataset/model/")
    parser.add_argument('--recover', action='store_true')

    args = parser.parse_args()

    if args.recover:
        print("recovering...")
        G, nodelist, node_weights, model = recover_graph(args.save_dir)
    else:
        G, nodelist, node_weights, model = create_graph(topics=args.search, depth=args.depth, simplify=args.simplify, save_dir=args.save_dir)
    
    with open(os.path.join(args.save_dir, 'model.bin'), 'wb+') as model_file:
        model.save(model_file)
    with open(os.path.join(args.save_dir, 'graph.pickle'), 'wb+') as graph_file:
        pickle.dump(G, graph_file) # G = pickle.load(graph_file)
    with open(os.path.join(args.save_dir, 'nodelist.npy'), 'wb+') as nodelist_file:
        np.save(nodelist_file, np.array(nodelist))
    with open(os.path.join(args.save_dir, 'node_weights.npy'), 'wb+') as node_weights_file:
        np.save(node_weights_file, np.array(node_weights))
    
    if args.plot:
        tg = simplified_plot(G, nodelist, node_weights)