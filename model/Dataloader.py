from model.utils import preprocessing_infobox
from settings import *

from keras_preprocessing.sequence import pad_sequences
from transformers import BertModel, BertTokenizer
from numpy.random import shuffle
from skimage.measure import block_reduce
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.ma as ma
import torch
from random import sample
import random
import re


class CrossNetworkLoaderWithoutCandidates(object):
    def __init__(self, sources: list, training=False, batch_size=64, num_candidates=10, num_neighbors=20,
                 max_sentence_length=25, max_attr_num=8, mode="test", use_numpy_embedding=False, no_loading=False,
                 seed=43):
        # config loader
        self.bert = None  # BertClient()
        self.sources = sources  # example: [mag, en_wiki, wikidata, freebase]
        self.training = training
        self.num_graphs = len(sources)
        self.batch_size = batch_size
        self.num_candidates = num_candidates
        self.num_neighbors = num_neighbors
        self.max_sentence_length = max_sentence_length
        self.max_attr_num = max_attr_num
        self.mode = mode
        self.seed = seed
        self.dir = join(DATASET_DIR, '_'.join(sources))
        if not exists(self.dir):
            os.makedirs(self.dir, exist_ok=True)

        self.graphs = None
        self.graph_size = None
        self.id_vocab = None
        self.id_to_idx = None

        self.links = None
        self.test_links = None
        self.table = None
        self.unique_vocab_size = None

        # semantic
        self.semantics = None

        self.neighbor_nodes = list()
        self.neighbor_adj = list()
        self.hier_nodes = list()
        self.hier_pos = list()
        self.hier_adj = list()

        # load graphs
        if self.training:
            print("training, loading graph information...")
            info = read_json('_'.join(sources) + '_graph_info.json')
            self.graph_size, self.id_vocab, self.vocab_size = info['graph_size'], info['id_vocab'], info['vocab_size']
            print("\tsize of each graph", self.graph_size)
            self.id_to_idx = dict((_id, idx) for idx, _id in enumerate(self.id_vocab))
            if len(self.sources) == 1:
                self.table = np.load(join(self.dir, 'trans_table.npy'), allow_pickle=True)
            else:
                # load links
                self.load_links()

                # build transform table

                print("\tbuilding transform table, total links", len(self.links), "...")
                self.table = [i for i in range(len(self.id_vocab))]
                for i, idx in enumerate(self.table.copy()):
                    link_idx = self.idx_links.get(idx)
                    if link_idx is not None:
                        self.table[i] = self.table[link_idx]
                self.unique_vocab_size = len(set(self.table))
                self.table = np.array(self.table)
                # np.save(join(self.dir, 'trans_table.npy'), self.table)

                print("\tcross links number:", self.table.shape[0])
            self.table = np.concatenate((self.table, np.array([self.table.shape[0]])), axis=0)
            self.bidirect_table = dict()
            for i in range(self.table.shape[0]):
                if self.table[i] != i:
                    self.bidirect_table[i] = self.table[i]
                    self.bidirect_table[self.table[i]] = i
            self.unique_vocab_size = self.table.shape[0]
            if len(self.sources) > 1:
                self.idx_test_links = np.load(join(self.dir, 'test_links.npy'), allow_pickle=True)
        else:
            print("loading graphs...")
            self.graphs = list(
                json.load(codecs.open(join(DATASET_DIR, self.dir, source + '_sub.json'), 'r', 'utf-8')) for source in
                self.sources)
            self.graph_size = [len(g) for g in self.graphs]
            self.id_vocab = list()
            for graph in self.graphs:
                self.id_vocab = self.id_vocab + list(graph.keys())
            self.vocab_size = len(self.id_vocab)
            self.id_to_idx = dict((_id, idx) for idx, _id in enumerate(self.id_vocab))
            write_json({'graph_size': self.graph_size, 'id_vocab': self.id_vocab, 'vocab_size': self.vocab_size},
                       self.dir, '_'.join(sources) + '_graph_info.json', overwrite=True)
            print("\tsize of each graph", self.graph_size)
            print("\tnumber of nodes:", self.vocab_size)
            graphs = dict()
            for g in self.graphs:
                graphs.update(g)
            self.graphs = graphs

            # load links
            self.load_links(1)

            # build transform table

            print("building transform table...")
            self.table = [i for i in range(len(self.id_vocab))]
            for i, idx in enumerate(self.table.copy()):
                link_idx = self.idx_links.get(idx)
                if link_idx is not None:
                    self.table[i] = self.table[link_idx]
            self.unique_vocab_size = len(set(self.table))
            self.table = np.array(self.table)
            np.save(join(self.dir, 'trans_table.npy'), self.table)

        # generate data
        # 1. semantic data
        print("generate semantics...")
        if not no_loading:
            if self.training:
                self.semantics = RawSemanticsHandler(self.sources, self.dir,
                                                     max_sentence_length=self.max_sentence_length,
                                                     max_attr_num=self.max_attr_num,
                                                     use_numpy_embedding=use_numpy_embedding,
                                                     training=self.training)
            else:
                for i, source in enumerate(self.sources):
                    if exists(join(self.dir, source + '_raw_semantics.npy')):
                        print("\t{} raw semantics exists...".format(source))
                        continue
                    print("\tcreate raw semantics for {}".format(source))
                    offset = np.sum(self.graph_size[:i]) if i > 0 else 0
                    self.generate_raw_semantic_data(self.id_vocab[offset:offset + self.graph_size[i]], source)
        # 2. neighbor data
        print("generate neighbor data...")
        if exists(join(self.dir, 'neighbor_nodes.npy')):
            print("\tuse existed neighbor data...")
            self.neighbor_nodes = np.load(join(self.dir, 'neighbor_nodes.npy'), allow_pickle=True)
            self.neighbor_adj = np.load(join(self.dir, 'neighbor_adj.npy'), allow_pickle=True)
        else:
            # if self.training:
            # raise RuntimeError("No prepared neighbor data! Please set training=False to create.")
            print("\tno existed neighbor data...")
            for i, source in enumerate(self.sources):
                if exists(join(self.dir, source + '_neighbor_nodes.npy')):
                    print("\tload data for {}".format(source))
                    neighbor_nodes = np.load(join(self.dir, source + '_neighbor_nodes.npy'), allow_pickle=True)
                    neighbor_adj = np.load(join(self.dir, source + '_neighbor_adj.npy'), allow_pickle=True)
                else:
                    print("\tcreate data for {}".format(source))
                    offset = np.sum(self.graph_size[:i]) if i > 0 else 0
                    neighbor_nodes, neighbor_adj = self.generate_neighbor(
                        self.id_vocab[offset:offset + self.graph_size[i]])
                    np.save(join(self.dir, source + '_neighbor_nodes.npy'), neighbor_nodes)
                    np.save(join(self.dir, source + '_neighbor_adj.npy'), neighbor_adj)
                self.neighbor_nodes = np.concatenate((self.neighbor_nodes, neighbor_nodes),
                                                     axis=0) if i != 0 else neighbor_nodes
                self.neighbor_adj = np.concatenate((self.neighbor_adj, neighbor_adj),
                                                   axis=0) if i != 0 else neighbor_adj
            np.save(join(self.dir, 'neighbor_nodes.npy'), self.neighbor_nodes)
            np.save(join(self.dir, 'neighbor_adj.npy'), self.neighbor_adj)

        # 4. parents data
        print("generate parents data...")
        if exists(join(self.dir, 'parents_nodes.npy')):
            print("\tuse existed parent data...")
            self.parents_nodes = np.load(join(self.dir, 'parents_nodes.npy'), allow_pickle=True)
            self.parents_adj = np.load(join(self.dir, 'parents_adj.npy'), allow_pickle=True)
        else:
            print("\tno existed parents data...")
            for i, source in enumerate(self.sources):
                if exists(join(self.dir, source + '_parents_nodes.npy')):
                    print("\tload data for {}".format(source))
                    parents_nodes = np.load(join(self.dir, source + '_parents_nodes.npy'), allow_pickle=True)
                    parents_adj = np.load(join(self.dir, source + '_parents_adj.npy'), allow_pickle=True)
                else:
                    print("\tcreate data for {}".format(source))
                    offset = np.sum(self.graph_size[:i]) if i > 0 else 0
                    parents_nodes, parents_adj = self.generate_parents(
                        self.id_vocab[offset:offset + self.graph_size[i]])
                    np.save(join(self.dir, source + '_parents_nodes.npy'), parents_nodes)
                    np.save(join(self.dir, source + '_parents_adj.npy'), parents_adj)
                self.parents_nodes = np.concatenate((self.parents_nodes, parents_nodes),
                                                    axis=0) if i != 0 else parents_nodes
                self.parents_adj = np.concatenate((self.parents_adj, parents_adj),
                                                  axis=0) if i != 0 else parents_adj
            np.save(join(self.dir, 'parents_nodes.npy'), self.parents_nodes)
            np.save(join(self.dir, 'parents_adj.npy'), self.parents_adj)

        # 3. hierarchical data
        """
        self.base_embedding = None
        print("load base embedding...")
        if exists(join(self.dir, '_'.join(self.sources) + '_name_semantics.npy')):
            self.base_embedding = np.load(join(self.dir, '_'.join(self.sources) + '_name_semantics.npy'),
                                          allow_pickle=True)
        else:
            print("\tcreate_base_embedding...")
            self.create_base_embedding()
        """
        # loading contrastive candidates

        self.contrastive = list()
        for i, source in enumerate(self.sources):
            if i == 0:
                self.contrastive = np.load(join(self.dir, source + '_contrastive.npy'), allow_pickle=True)
            else:
                self.contrastive = np.concatenate(
                    (self.contrastive, np.load(join(self.dir, source + '_contrastive.npy'),
                                               allow_pickle=True)), axis=0)

        print("data loaded...")

    def iterator_on_graphs(self, equal=False):
        used_batch = [0] * self.num_graphs
        num_batch_on_graphs = np.array(self.graph_size) // self.batch_size
        if equal:
            probability = [1 / len(self.sources)] * len(self.sources)
            graphs_iteration_time = [6, 1]  # hard coded for mag & en_wiki
            while np.sum(graphs_iteration_time) != 0:
                graph_idx = np.random.choice([i for i in range(self.num_graphs)], p=probability)
                num_batch_on_graphs[graph_idx] = num_batch_on_graphs[graph_idx] - 1
                used_batch[graph_idx] += 1
                if num_batch_on_graphs[graph_idx] == 0:
                    graphs_iteration_time[graph_idx] -= 1
                    used_batch[graph_idx] = 0
                yield graph_idx, used_batch[graph_idx] - 1
        else:
            if self.num_graphs > 1:
                probability = num_batch_on_graphs / np.sum(num_batch_on_graphs)
            else:
                probability = [1.0]
            while np.sum(num_batch_on_graphs) != 0:
                graph_idx = np.random.choice([i for i in range(self.num_graphs)], p=probability)
                num_batch_on_graphs[graph_idx] = num_batch_on_graphs[graph_idx] - 1
                used_batch[graph_idx] += 1
                if self.num_graphs > 1:
                    probability = num_batch_on_graphs / np.sum(num_batch_on_graphs)
                yield graph_idx, used_batch[graph_idx] - 1

    def __iter__(self):
        np.random.seed(self.seed)
        graphs = [np.array(np.arange(self.graph_size[i])) for i in range(self.num_graphs)]
        for graph in graphs:
            shuffle(graph)
        cnt = 0
        for graph_idx, batch_num in self.iterator_on_graphs():
            offset = int(np.sum(self.graph_size[:graph_idx]))
            start, end = batch_num * self.batch_size, min((batch_num + 1) * self.batch_size, self.graph_size[graph_idx])
            batch_nodes = graphs[graph_idx][start: end] + offset
            batch_neighbors = self.neighbor_nodes[batch_nodes]
            neighbor_mask = batch_neighbors == -1
            batch_neighbors = np.where(~neighbor_mask, batch_neighbors + offset, -1)
            batch_neighbor_adj = self.neighbor_adj[batch_nodes]
            batch_features = np.array([1])  # self.semantics(graph_idx, batch_nodes)
            batch_contrast = self.contrastive[batch_nodes] + offset
            batch_parents = self.parents_nodes[batch_nodes] + offset
            batch_parents_adj = self.parents_adj[batch_nodes]
            if batch_nodes.shape[0] == 0:
                continue
            cnt += 1
            yield cnt, batch_nodes, batch_features, batch_neighbors, batch_neighbor_adj, batch_contrast, graph_idx, batch_parents, batch_parents_adj

    def evaluate_batch(self):
        for graph_idx in range(self.num_graphs):
            offset = int(np.sum(self.graph_size[:graph_idx]))
            batch_num = self.graph_size[graph_idx] // self.batch_size + 1
            for i in range(batch_num):
                start, end = i * self.batch_size, min((i + 1) * self.batch_size, self.graph_size[graph_idx])
                batch_nodes = np.array(np.arange(start, end)) + offset
                batch_neighbors = self.neighbor_nodes[batch_nodes]
                neighbor_mask = batch_neighbors == -1
                batch_neighbors = np.where(~neighbor_mask, batch_neighbors + offset, -1)
                batch_neighbor_adj = self.neighbor_adj[batch_nodes]
                batch_features = np.array([1])  # self.semantics(graph_idx, batch_nodes)
                batch_contrast = np.array([1])  # self.contrastive[batch_nodes]
                yield batch_nodes, batch_features, batch_neighbors, batch_neighbor_adj, batch_contrast, graph_idx

    def evaluate_contrast_batch(self):
        for graph_idx in range(self.num_graphs):
            offset = int(np.sum(self.graph_size[:graph_idx]))
            batch_num = self.graph_size[graph_idx] // self.batch_size + 1
            for i in range(batch_num):
                start, end = i * self.batch_size, min((i + 1) * self.batch_size, self.graph_size[graph_idx])
                batch_nodes = np.array(np.arange(start, end)) + offset
                batch_neighbors = self.neighbor_nodes[batch_nodes]
                neighbor_mask = batch_neighbors == -1
                batch_neighbors = np.where(~neighbor_mask, batch_neighbors + offset, -1)
                batch_neighbor_adj = self.neighbor_adj[batch_nodes]
                batch_features = np.array([1])  # self.semantics(graph_idx, batch_nodes)
                batch_contrast = self.contrastive[batch_nodes] + offset
                batch_parents = self.parents_nodes[batch_nodes] + offset
                batch_parents_adj = self.parents_adj[batch_nodes]
                yield batch_nodes, batch_features, batch_neighbors, batch_neighbor_adj, batch_contrast, graph_idx, batch_parents, batch_parents_adj

    def get_input(self, batch_nodes, graph_idx):
        offset = int(np.sum(self.graph_size[:graph_idx]))
        batch_neighbors = self.neighbor_nodes[batch_nodes]
        neighbor_mask = batch_neighbors == -1
        batch_neighbors = np.where(~neighbor_mask, batch_neighbors + offset, -1)
        batch_neighbor_adj = self.neighbor_adj[batch_nodes]
        batch_features = np.array([1])  # self.semantics(graph_idx, batch_nodes)
        batch_contrast = self.contrastive[batch_nodes] + offset
        batch_parents = self.parents_nodes[batch_nodes] + offset
        batch_parents_adj = self.parents_adj[batch_nodes]
        return batch_nodes, batch_features, batch_neighbors, batch_neighbor_adj, batch_contrast, batch_parents, batch_parents_adj

    def get_contrast(self, node, graph_idx):
        offset = int(np.sum(self.graph_size[:graph_idx]))
        return self.contrastive[node] + offset

    def generate_raw_semantic_data(self, id_vocab, source):
        raw_semantics = []
        # raw_semantics_all = []
        for i, _id in enumerate(id_vocab):
            doc = self.graphs[_id]
            data = [doc['NormalizedName_En']]
            """
            if doc['source'] == 'mag':
                pass
            elif doc['source'] == 'en_wiki':
                if doc['Definitions'] is not None and re.search(r'[a-z]', doc['Definitions']) is not None:
                    data.extend(['definition', doc['Definitions']])
                infobox = doc['Others'].get('infobox')
                if infobox and len(infobox) > 0:
                    data.extend(preprocessing_infobox(infobox))
            elif doc['source'] == 'wikidata':
                if doc['Definitions'] is not None and re.search(r'[a-z]', doc['Definitions']) is not None:
                    data.extend(['definition', doc['Definitions']])
                if doc['Links'] and type(doc['Links']) == dict:
                    for attr, _id in doc['Links'].items():
                        data.extend([attr, self.graphs[doc['_id']]['NormalizedName_En']])
            elif doc['source'] == 'freebase':
                if doc['Definitions'] is not None and re.search(r'[a-z]', doc['Definitions']) is not None:
                    data.extend(['definition', doc['Definitions']])
                if doc['Links'] and type(doc['Links']) == dict:
                    for attr, _id in doc['Links'].items():
                        data.extend([attr, self.graphs[doc['_id']]['NormalizedName_En']])
            """
            raw_semantics.append(data)
            # raw_semantics_all.extend(data)
        # write_json(raw_semantics, self.dir, source + '_raw_semantics.json', overwrite=True)
        np.save(join(self.dir, source + '_raw_semantics.npy'), raw_semantics)
        # return raw_semantics

    def transform_links(self, id_links):
        return [self.id_to_idx.get(nid) for nid in id_links if self.id_to_idx.get(nid)]

    def load_links(self, ratio=1.0):
        print("\tloading links, ratio =", ratio)
        # load links
        if len(self.sources) == 1:
            self.idx_links = dict((i, i) for i in range(self.graph_size[0]))
        print("loading links...")
        self.links, self.test_links = dict(), dict()
        for i in range(len(self.sources) - 1):
            links = json.load(
                codecs.open(join(self.dir, self.sources[i] + '2' + self.sources[i + 1] + '_anchor.json'), 'r', 'utf-8'))
            if ratio < 1.0:
                links = dict(item for item in list(links.items())[:int(len(links) * ratio)])
            self.links.update(dict(item for item in list(links.items())))
        self.idx_links = dict()
        for x, y in self.links.items():
            x_idx = self.id_to_idx.get(x)
            y_idx = self.id_to_idx.get(y)
            if x_idx and y_idx:
                self.idx_links[y_idx] = x_idx
        if self.mode == 'test':
            self.test_links = json.load(codecs.open(join(self.dir, 'test_links_old.json'), 'r', 'utf-8'))
            self.idx_test_links = list()
            for x, y in self.test_links.items():
                x_idx = self.id_to_idx.get(x)
                y_idx = self.id_to_idx.get(y)
                if x_idx and y_idx:
                    self.idx_test_links.append(np.array([x_idx, y_idx]))
            self.idx_test_links = np.array(self.idx_test_links)
            np.save(join(self.dir, 'test_links_old.npy'), self.idx_test_links)
        self.idx_links[-1] = -1

    def build_fine_tune_training_set(self, neg_sample=1, use_random=True):
        if self.training:
            self.load_links()
        test_links = np.load(join(self.dir, 'test_links_old.npy'))
        ambiguous_test = np.load(join(self.dir, 'test_ambiguous.npy'))
        ambiguous_mag = set(ambiguous_test[:, 1].tolist())
        self.fine_tune_set = []
        random.seed = 43
        if not use_random:
            for pair in test_links.tolist():
                m, e = pair[0], pair[1]
                #if m in ambiguous_mag:
                    #continue
                self.fine_tune_set.append([1, m, e])
                neg = self.get_contrast(e, 1).tolist()[:neg_sample]
                for i in range(neg_sample):
                    self.fine_tune_set.append([0, m, neg[i]])
        else:
            for pair in list(self.idx_links.items())[-10000:]:
                e, m = pair[0], pair[1]
                self.fine_tune_set.append([1, m, e])
                neg = [random.randint(self.graph_size[0], sum(self.graph_size)) for i in range(neg_sample)]
                for i in range(neg_sample):
                    self.fine_tune_set.append([0, m, neg[i]])
        np.save(join(self.dir, 'fine_tune_training.npy'), np.array(self.fine_tune_set))

    def load_fine_tune_set(self):
        self.fine_tune_set = np.load(join(self.dir, 'test_synonym.npy'))
        self.ft_train, self.ft_valid = train_test_split(self.fine_tune_set, test_size=0.4, random_state=43)
        self.ft_valid, self.ft_test = train_test_split(self.ft_valid, test_size=0.5, random_state=43)

    def generate_adjacency_matrix(self, nodes, is_taxonomy):
        adj = np.zeros((self.num_neighbors, self.num_neighbors), dtype=np.bool)
        for i, node_idx in enumerate(nodes):
            if node_idx == -1:
                break
            doc = self.graphs[self.id_vocab[node_idx]]
            links = self.transform_links(doc['Parents'] + doc['Children']) if is_taxonomy else self.transform_links(
                doc['Links'])
            for j, _idx in enumerate(nodes[i:]):
                adj[i, i + j] = _idx in links or node_idx == _idx
        adj = adj + np.transpose(adj)
        return adj

    def generate_neighbor(self, id_vocab):
        neighbor_nodes, neighbor_adj = [], []
        for i, _id in enumerate(id_vocab):
            doc = self.graphs[_id]
            if doc.get('source') != 'mag':
                nodes = self.transform_links(doc['Links'])
            else:
                nodes = self.transform_links(doc['Parents'] + doc['Children'])
            while self.id_to_idx[_id] in nodes:
                nodes.remove(self.id_to_idx[_id])
            # if len(nodes) == 0:
            #     nodes = [self.id_to_idx[_id]]
            if len(nodes) < self.num_neighbors:
                # nodes.extend(np.random.choice(nodes, self.num_neighbors - len(nodes) - 1, replace=True).tolist())
                nodes = nodes + [-1] * (self.num_neighbors - len(nodes) - 1)
            else:
                nodes = nodes[:self.num_neighbors - 1]
            nodes = [self.id_to_idx[_id]] + nodes
            neighbor_nodes.append(nodes)
            neighbor_adj.append(self.generate_adjacency_matrix(nodes, doc.get('source') == 'mag'))
            if i % 10000 == 0:
                print('\t', i)
        return np.array(neighbor_nodes), np.array(neighbor_adj)

    def generate_parents_adj(self, nodes, num=5):
        adj = np.eye(num, dtype=np.bool)
        np_nodes = 5 - (np.array(nodes) == -1).sum()
        adj[0, :np_nodes] = 1
        adj = adj + np.transpose(adj)
        return adj

    def generate_parents(self, id_vocab):
        parents_nodes, parents_adj = [], []
        for i, _id in enumerate(id_vocab):
            doc = self.graphs[_id]
            nodes = self.transform_links(doc['Parents'])
            while self.id_to_idx[_id] in nodes:
                nodes.remove(self.id_to_idx[_id])
            if len(nodes) == 0:
                nodes = [-1] * 4
            elif len(nodes) < 5:
                self.parents_of_parents(nodes)
                if len(nodes) < 5:
                    nodes = nodes + [-1] * (5 - len(nodes) - 1)
                else:
                    nodes = nodes[:4]
            else:
                nodes = nodes[:4]
            nodes = [self.id_to_idx[_id]] + nodes
            parents_nodes.append(nodes)
            parents_adj.append(self.generate_parents_adj(nodes))
            if i % 10000 == 0:
                print('\t', i)
        return np.array(parents_nodes), np.array(parents_adj)

    def parents_of_parents(self, nodes):
        initial_len = len(nodes)
        for n in nodes.copy():
            doc = self.graphs[self.id_vocab[n]]
            nodes.extend(self.transform_links(doc['Parents']))
        if initial_len < len(nodes) < 4:
            self.parents_of_parents(nodes)

    def create_base_embedding(self):
        self.base_embedding = [np.zeros((1, 150,))] * self.unique_vocab_size
        for graph_idx in range(len(self.sources)):
            semantics = self.semantics.base_embedding[graph_idx]
            for idx in range(semantics.shape[0]):
                self.base_embedding[self.table[idx + int(np.sum(self.graph_size[:graph_idx]))]] = semantics[idx][
                                                                                                  np.newaxis, :]
                if idx % 12800 == 0:
                    print(idx)
        self.base_embedding = np.concatenate(self.base_embedding, axis=0)
        np.save(join(self.dir, '_'.join(self.sources) + '_name_semantics.npy'), self.base_embedding)

    def evaluate(self, proj_dir, mag, en_wiki, concat=150):
        embedding = en_wiki
        test_links = np.load(join(DATASET_DIR, 'mag_en_wiki', 'test_links.npy'))
        test_links[:, 1] -= self.graph_size[0]  # mag.shape[0]
        query = mag[test_links[:, 0]]
        contrast = np.load(join(self.dir, 'mag_en_wiki_candidates.npy'))
        candidates_embedding = torch.from_numpy(embedding[contrast][:, :, concat:])
        candidates_score = torch.bmm(candidates_embedding,
                                     torch.from_numpy(query).unsqueeze(-1)[:, concat:]).squeeze().numpy()
        result = np.multiply(query[:, concat:], embedding[test_links[:, 1]][:, concat:]).sum(1)[:, np.newaxis]
        r = (candidates_score > result).sum(1)
        self.print_evaluate(r)

    def evaluate_with_sample(self, iteration=7499, loader=None, proj_dir=None, mag=None, en_wiki=None, use_norm=True):
        if loader is None:
            loader = CrossNetworkLoaderWithoutCandidates(['mag', 'en_wiki'], training=True, mode="load",
                                                         no_loading=True)
            proj_dir = join('LinKG_MAG_ENWIKI_v4_2020-02-01 15:33:59.913098', str(iteration))
            embedding = np.load(join(OUT_DIR, proj_dir, 'en_wiki.npy'))
        else:
            embedding = en_wiki
        embedding = np.concatenate((embedding, np.zeros((1, embedding.shape[1]))))
        test_links = np.load(join(DATASET_DIR, 'mag_en_wiki', 'test.npy'), allow_pickle=True)  # 730 * 20 * 3
        test_links[:, :, 2] -= loader.graph_size[0]
        mask = test_links[:, :, 2] == -1
        if mag is None:
            query = np.load(join(OUT_DIR, proj_dir, 'mag.npy'))[test_links[:, :, 1]][:, :, 150:]
        else:
            query = mag[test_links[:, :, 1]][:, :, 150:]
        target = embedding[test_links[:, :, 2]][:, :, 150:]
        if use_norm:
            score = np.linalg.norm(query - target, axis=2)
            score = ma.array(score, mask=mask, fill_value=10000)
            score.filled()
            rank = np.argmin(score, axis=1)
        else:
            score = np.multiply(query, target).sum(2)
            rank = np.argmax(score, axis=1)
        print("iteration:", iteration)
        return self.print_evaluate(rank)

    def print_evaluate(self, r):
        print("Hit@1:", (r == 0).sum() / r.shape[0])
        print("Hit@2:", (r < 2).sum() / r.shape[0])
        print("Hit@3:", (r < 3).sum() / r.shape[0])
        print("Hit@5:", (r < 5).sum() / r.shape[0])
        print("Hit@10:", (r < 10).sum() / r.shape[0])
        print("Hit@15:", (r < 15).sum() / r.shape[0])
        print("Hit@20:", (r < 20).sum() / r.shape[0])
        result = "Hit@1:{}\nHit@2:{}\nHit@3:{}\nHit@5:{}\n\n".format((r == 0).sum() / r.shape[0],
                                                                     (r < 2).sum() / r.shape[0],
                                                                     (r < 3).sum() / r.shape[0],
                                                                     (r < 5).sum() / r.shape[0])
        return result


class RawSemanticsHandler(object):
    def __init__(self, sources, working_dir, max_sentence_length=25, max_attr_num=8, use_numpy_embedding=False,
                 use_pool_output=False, training=True, loading_bert=False):
        print("\tload raw semantics...")
        self.sources = sources
        self.dir = working_dir
        if training:
            self.raw_semantics = list(
                np.load(join(self.dir, source + '_token_semantics.npy'), allow_pickle=True) for source in sources)
            # self.base_embedding = list(
            # np.load(join(self.dir, source + '_name_semantics.npy'), allow_pickle=True) for source in sources)
        else:
            self.raw_semantics = list(
                np.load(join(self.dir, source + '_raw_semantics.npy'), allow_pickle=True) for source in sources)
            self.base_embedding = None
        self.use_numpy_embedding = use_numpy_embedding
        self.use_pool_output = use_pool_output

        self.max_attr_num = max_attr_num
        self.max_sentence_length = max_sentence_length
        if use_numpy_embedding:
            self.embedding = list()
            for i, source in enumerate(self.sources):
                if i != len(self.sources) - 1:
                    self.embedding.append(np.load(join(self.dir, source + '_semantics.npy'))[:-1, :])
                else:
                    self.embedding.append(np.load(join(self.dir, source + '_semantics.npy')))
            self.base_embedding = np.concatenate(self.embedding)
        print("\tload bert model...")
        if loading_bert:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            self.bert_encoder = BertModel.from_pretrained(
                'bert-base-uncased').cuda() if torch.cuda.is_available() else None

        print("\tsemantics prepared...")

    def get_pad_and_mask(self, raw_tokens):
        tokens = np.array([pad_sequences(token, self.max_sentence_length, padding='post') for token in raw_tokens])
        return tokens, tokens > 0

    def tokenize(self, node_attrs):
        return [self.bert_tokenizer.encode(n, max_length=self.max_sentence_length) for
                n in node_attrs[:self.max_attr_num * 2]] + [[0]] * 2 * (self.max_attr_num - len(node_attrs) // 2)

    def pad_into_embedding(self, raw_embedding, attrs_num):
        tensor = []
        start = 0
        for i in range(len(attrs_num)):
            attrs = torch.zeros((self.max_attr_num, 2 * 768)).cuda()
            attrs[:min(self.max_attr_num, attrs_num[i]), :] = raw_embedding[start:start + min(self.max_attr_num * 2,
                                                                                              attrs_num[i] * 2),
                                                              :].view(-1, 2 * 768)
            tensor.append(attrs)
            start += min(self.max_attr_num, attrs_num[i] * 2)
        return torch.stack(tuple(tensor), dim=0)

    def __call__(self, graph_idx, batch_nodes):
        max_num_attrs = 0
        if self.use_numpy_embedding:
            embedding = self.embedding[graph_idx][batch_nodes]
            semantics = []
            for i in range(batch_nodes.shape[0]):
                max_num_attrs = max(max_num_attrs, embedding[i].shape[0])
            for i in range(batch_nodes.shape[0]):
                semantics.append(np.pad(embedding[i], ((0, max_num_attrs - embedding[i].shape[0]),), 'constant',
                                        constant_values=(0,)))
            return np.array(semantics)

        semantics = self.raw_semantics[graph_idx]
        with torch.no_grad():
            batch_semantics = semantics[batch_nodes]
            batch_semantics, batch_attrs_num = np.array(batch_semantics[:, 0].tolist()), batch_semantics[:, 1].tolist()
            try:
                tokens = np.concatenate(batch_semantics, axis=0).reshape((-1, self.max_sentence_length))
            except:
                return None
            raw_embedding = self.bert_encoder(torch.LongTensor(tokens).cuda(),
                                              attention_mask=torch.tensor(tokens > 0).cuda())
            if self.use_pool_output:
                raw_embedding = raw_embedding[1].view(-1, self.max_attr_num, 2 * 768)
            else:
                raw_embedding = raw_embedding[0][:, 1:, :].mean(dim=(1,))
            # padded_embedding = torch.zeros(max_num_attrs, 768 * 2)
            # padded_embedding[:raw_embedding.shape[0]] = raw_embedding
            return raw_embedding

    def dump_tokens(self):
        token_semantics = []
        for graph_idx in range(len(self.raw_semantics)):
            semantics = self.raw_semantics[graph_idx]
            for idx in range(semantics.shape[0]):
                tokens = []
                node = semantics[idx] if type(semantics[0]) == list else semantics[idx].tolist()
                for attr in node[:self.max_attr_num * 2]:
                    token = self.bert_tokenizer.encode(attr, max_length=self.max_sentence_length)
                    tokens.extend(token + [0] * (self.max_sentence_length - len(token)))
                token_semantics.append(
                    np.array([np.array(tokens), 1]))  # len(tokens) // (2 * self.max_sentence_length)]))
                if idx % 12800 == 0:
                    print(graph_idx, idx)
            token_semantics = np.array(token_semantics)
            np.save(join(self.dir, self.sources[graph_idx] + '_token_semantics.npy'), np.array(token_semantics))
            self.raw_semantics = token_semantics

    def dump_name_tokens(self):
        token_semantics = []
        for graph_idx in range(len(self.raw_semantics)):
            semantics = self.raw_semantics[graph_idx]
            for idx in range(semantics.shape[0]):
                tokens = []
                node = semantics[idx] if type(semantics[0]) == list else semantics[idx].tolist()
                for attr in node[:2]:
                    token = self.bert_tokenizer.encode(attr, max_length=self.max_sentence_length)
                    tokens.extend(token + [0] * (self.max_sentence_length - len(token)))
                token_semantics.append(np.array([np.array(tokens), len(tokens) // (2 * self.max_sentence_length)]))
                if idx % 12800 == 0:
                    print(graph_idx, idx)
            token_semantics = np.array(token_semantics)
            np.save(join(self.dir, self.sources[graph_idx] + '_name_token_semantics.npy'), np.array(token_semantics))
            print(self.sources[graph_idx], token_semantics.shape)

    def save_initial_embedding(self):
        for graph_idx in range(len(self.raw_semantics)):
            semantics = self.raw_semantics[graph_idx]
            name_semantics = []
            for idx in range(0, semantics.shape[0], 256):
                nodes = np.arange(idx, min(idx + 256, semantics.shape[0]))
                embedding = self(graph_idx, nodes).squeeze(1).cpu().numpy()
                embedding = block_reduce(embedding, (1, 1, 10), np.mean)
                name_semantics.append(embedding[:, 0, :150])
                if idx % 12800 == 0:
                    print(self.sources[graph_idx], idx)
            np.save(join(self.dir, self.sources[graph_idx] + '_name_semantics.npy'),
                    np.concatenate(name_semantics, axis=0))
            print(self.sources[graph_idx], len(name_semantics))

    def get_embedding(self):
        for graph_idx in range(len(self.raw_semantics)):
            semantics = self.raw_semantics[graph_idx]
            name_semantics = []
            for idx in range(0, semantics.shape[0], 256):
                nodes = np.arange(idx, min(idx + 256, semantics.shape[0]))
                embedding = self(graph_idx, nodes).squeeze(1).cpu().numpy().astype(np.float16)
                # embedding = block_reduce(embedding, (1, 1, 10), np.mean)
                name_semantics.append(embedding[:, :])
                if idx % 12800 == 0:
                    print(self.sources[graph_idx], idx)
            name_semantics = np.concatenate(name_semantics, axis=0)
            name_semantics = np.concatenate((name_semantics, np.zeros((1, 768), dtype=np.float16)), axis=0)
            np.save(join(self.dir, self.sources[graph_idx] + '_semantics.npy'),
                    name_semantics)
            print(self.sources[graph_idx], len(name_semantics))


if __name__ == '__main__':
    CrossNetworkLoaderWithoutCandidates(['mag', 'en_wiki'], training=True, batch_size=128, mode='test',
                                        use_numpy_embedding=True).build_fine_tune_training_set(use_random=False)
