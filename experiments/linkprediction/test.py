import os
import torch
from torch_geometric.nn import VGAE
from train_vgae import VGAE_Encoder
import load_wiki
import json
import argparse
import nltk
import numpy as np
from gensim.models.doc2vec import Doc2Vec

class VGAETester():
    def __init__(self, args):
        # Loading Data
        self.x =torch.load(os.path.join(args.save_dir, 'data.pt'))
        print(self.x.shape)
        self.train_pos_edge_index =torch.load(os.path.join(args.save_dir, 'train_pos_edge_index.pt'))
        self.val_pos_edge_index =torch.load(os.path.join(args.save_dir, 'val_pos_edge_index.pt'))
        self.test_pos_edge_index =torch.load(os.path.join(args.save_dir, 'test_pos_edge_index.pt'))
        self.pos_edge_index = torch.cat([self.train_pos_edge_index, self.val_pos_edge_index, 
                                         self.test_pos_edge_index], dim=1)
        # Loading Model
        self.doc2vec = Doc2Vec.load(args.doc2vec_model_dir)
        num_features = self.x.shape[1]
        self.model =VGAE(VGAE_Encoder(num_features))
        self.model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model.pt')))
        self.model.eval()

    def test_new_graph(self, data_path, k=10):
        test_data = []
        test_data_titles = []
        for path, _subdirs, files in os.walk(data_path):
            for name in files:
                if name != 'vectors.json': continue
                test_data_point = load_wiki.load_data(os.path.join(path, name))
                test_data.append(test_data_point.x)
                test_data_titles.append(os.path.join(path, 'readable.json'))
        test_data = torch.concat(test_data)
        N = len(test_data)
        test_z = self.model.encode(test_data, torch.tensor([[], []], dtype=torch.long))
        # pot_adj = [[0, 0], [0, 1], ..., [N-1, N-1]]
        pot_adj = torch.meshgrid(torch.arange(N), torch.arange(N))
        pot_adj = torch.stack(pot_adj).T.reshape(-1, 2)[:N*N]
        # remove edges with self
        pot_adj = pot_adj[pot_adj[:, 0] != pot_adj[:, 1]]
        pot_adj = pot_adj.to(torch.long)
        print(pot_adj)
        adj = self.model.decoder(test_z, pot_adj.T)
        adj = torch.sigmoid(adj)
        connections = torch.topk(-adj.flatten(), k, largest=True).indices
        docs = [] 
        for connection in connections:
            doc1, doc2 = pot_adj[connection]
            doc1 = self.load_document(test_data_titles[doc1])
            doc2 = self.load_document(test_data_titles[doc2])
            print(adj[connection].item(), doc1['title'].split('/')[-1], doc2['title'].split('/')[-1])
            docs.append((doc1, doc2))
        return docs

    def test_all_with_wiki(self, data_path):
        for path, _subdirs, files in os.walk(data_path):
            for name in files:
                if name != 'node.txt': continue
                closest_docs = self.test_with_wiki(os.path.join(path, name))
                with open(os.path.join(path, "closest_docs.json"), "w") as outfile:
                    outfile.write(json.dumps(closest_docs))

    def test_with_wiki(self, data_path):
        with open(data_path) as file:
            doc = file.read()
        tokenized_docs = nltk.word_tokenize(' '.join(doc).lower())
        test_data = self.doc2vec.infer_vector(tokenized_docs)
        test_data = torch.from_numpy(test_data)
        test_data = torch.unsqueeze(test_data, dim=0)
        print(test_data.shape)
        x = torch.cat([self.x, test_data], dim=0)
        print(x)
        z = self.model.encode(x, self.pos_edge_index)
        N, F = x.shape # Num Nodes, Num Features
        pot_adj = torch.stack([(N-1)*torch.ones(N-1), torch.arange(N-1)], dim=0)
        pot_adj = pot_adj.to(torch.long)
        adj = self.model.decoder(z, pot_adj)
        doc_nums = torch.topk(adj.flatten(), 5, largest=True).indices
        docs = [] 
        for doc_num in doc_nums:
            doc = self.find_document(doc_num)
            print(data_path, doc)
            docs.append(doc)
        return docs


    def test(self, data_path):
        test_data = load_wiki.load_data(data_path)
        self.x = torch.cat([self.x, test_data.x], dim=0)
        z = self.model.encode(self.x, self.pos_edge_index)
        N, F = self.x.shape # Num Nod# es, Num Features
        pot_adj = torch.stack([(N-1)*torch.ones(N-1), torch.arange(N-1)], dim=0)
        pot_adj = pot_adj.to(torch.long)
        adj = self.model.decoder(z, pot_adj)
        doc_nums = torch.topk(adj.flatten(), 2, largest=True).indices
        docs = [] 
        for doc_num in doc_nums:
            doc = self.find_document(doc_num)
            docs.append(doc)
        return docs
    
    def find_document(self, doc_num):
        nodes = np.load(args.metadata_path)
        # f = open(args.metadata_path)
        # data = json.load(f)
        return nodes[doc_num - 1]

    def load_document(self, doc_path):
        f = open(doc_path)
        data = json.load(f)
        return data['nodes'][0]
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGAE TEST')
    parser.add_argument('--save_dir', type=str, default="../../results/data/vgae/")
    parser.add_argument('--new_graph_data_path', type=str, default="../../dataset/test/")
    parser.add_argument('--data_path', type=str, default="../../dataset/test/cross_platform")
    parser.add_argument('--metadata-path', type=str, default="../../dataset/model/nodelist.npy")
    parser.add_argument('--doc2vec-model-dir', type=str, default="../../dataset/model_test/model.bin")
    args = parser.parse_args()
    # args.vectors_data_path = os.path.join(args.data_path, 'vectors.json')
    tester = VGAETester(args)
    tester.test_all_with_wiki(args.new_graph_data_path)
    # docs = tester.test_new_graph(args.new_graph_data_path)
    # docs = tester.test(args.vectors_data_path)
    # with open(os.path.join(args.new_graph_data_path, "closest_docs.json"), "w") as outfile:
        # outfile.write(json.dumps(docs))