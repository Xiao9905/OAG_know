import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
import torch
from torch.optim import Adam
import argparse
import numpy as np
from tensorboardX import SummaryWriter
from datetime import datetime
import warnings

from settings import *
from model.layers import BatchMultiHeadGraphAttention
from model.Dataloader import CrossNetworkLoaderWithoutCandidates
from model.utils import *

parser = argparse.ArgumentParser('argument for training')
warnings.filterwarnings("ignore")


def parse_options():
    parser.add_argument('--model', type=str, default='LinKG_ENWIKI_v4')
    parser.add_argument('--time', type=str, default=str(datetime.now()))

    parser.add_argument('--n_dim_u', type=int, default=150)
    parser.add_argument('--n_dim_r', type=int, default=150)
    parser.add_argument('--n_dim_v', type=int, default=150)

    parser.add_argument('--n_head_sem', type=int, default=4)
    parser.add_argument('--n_head_hier', type=int, default=3)
    parser.add_argument('--n_head_struc', type=int, default=4)
    parser.add_argument('--n_type_node', type=int, default=1)
    parser.add_argument('--attn_dropout', type=float, default=0.2)

    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_neighbors', type=int, default=1)
    parser.add_argument('--max_depth', type=int, default=2)
    parser.add_argument('--update_freq', type=int, default=5)
    parser.add_argument('--queue_length', type=int, default=32)

    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--t', type=float, default=1)
    parser.add_argument('--loss_weight', type=float, default=1)
    parser.add_argument('--lr_decay_iteration', type=int, default=2000)
    parser.add_argument('--lr_decay', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)

    return parser.parse_args()


args = parse_options()


class SemanticEncoder(nn.Module):
    def __init__(self, f_in, f_out):
        super(SemanticEncoder, self).__init__()
        self.w = nn.MaxPool1d(5)

    def forward(self, v):
        if len(v.shape) == 2:
            v = v.unsqueeze(1)
        output = self.w(v)
        return output[:, :, :150].squeeze()


class NCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


class LinKG(nn.Module):
    def __init__(self, vocab_size, unique_vocab_size, num_neighbor, table, base_embedding, max_up_down_depth=5):
        super(LinKG, self).__init__()
        self.u_vocab_size = unique_vocab_size
        self.vocab_size = vocab_size
        self.num_neighbor = num_neighbor
        self.max_up_down_depth = max_up_down_depth
        self.table = torch.from_numpy(table).cuda()
        self.fake_table = torch.from_numpy(np.arange(self.vocab_size)).cuda()

        # embedding
        self.b = torch.from_numpy(base_embedding)
        self.u = nn.Embedding(self.vocab_size + 1, args.n_dim_u)

        # transform matrix from u to r
        self.w = nn.Linear(args.n_dim_u, args.n_dim_r)
        self.w1 = nn.Linear(3 * args.n_dim_u, args.n_dim_v)
        self.w2 = nn.Linear(2 * args.n_dim_v, args.n_dim_v)

        # context embedding
        self.c = nn.Embedding(self.u_vocab_size, args.n_dim_v)

        # local-global relation discriminator
        self.d = nn.Parameter(torch.Tensor(args.n_dim_r, args.n_dim_v))
        nn.init.xavier_uniform_(self.d)

        # semantic self-attention encoder
        self.semantic_encoder = SemanticEncoder(768, args.n_dim_v)

        # structural networks
        self.gat_b = BatchMultiHeadGraphAttention(args.n_head_struc, args.n_dim_v, args.n_dim_v, args.attn_dropout)
        self.gat_u = BatchMultiHeadGraphAttention(args.n_head_struc, args.n_dim_u, args.n_dim_v, args.attn_dropout)
        self.gat_p = BatchMultiHeadGraphAttention(args.n_head_struc, args.n_dim_u, args.n_dim_v, args.attn_dropout)
        self.out_b = nn.Linear(args.n_head_struc * args.n_dim_v, args.n_dim_v)
        self.out_u = nn.Linear(args.n_head_struc * args.n_dim_v, args.n_dim_v)
        self.out_p = nn.Linear(args.n_head_struc * args.n_dim_v, args.n_dim_v)
        self.norm = nn.BatchNorm1d(args.n_dim_v)

        # loss
        self.criterion = NCESoftmaxLoss()
        self.criterion_ = NCESoftmaxLoss()

    def local_global_loss_c(self, l, neighbors, neg_neighbors):
        bsz = neighbors.shape[0]
        s_pos, s_neg = self.c(neighbors).mean(1), self.c(neg_neighbors).mean(1)
        l_prime = torch.matmul(l, self.d).unsqueeze(1)
        l_pos = torch.bmm(l_prime, s_pos.unsqueeze(-1))
        l_neg = torch.bmm(l_prime, s_neg.unsqueeze(-1))
        logits = torch.cat((l_pos, l_neg), dim=1)
        logits = logits.squeeze().contiguous()
        logits_softmax = torch.softmax(logits, 1)
        loss = self.criterion(logits)
        return loss, logits_softmax[:, 0], logits_softmax[:, 1:]

    def contrastive_loss(self, q, q_plus, k):
        """
        calculate contrastive loss
        :param q, q_plus: [bs, n_dim]
        :param k: [bs * args.queue_length, n_dim]
        :return:
        """
        queue_length = k.shape[0]
        bsz = q.shape[0]
        l_pos = torch.bmm(q.view(bsz, 1, -1), q_plus.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)
        l_neg = torch.mm(q.view(bsz, args.n_dim_v), k.view(args.n_dim_v, queue_length))
        logits = torch.cat((l_pos, l_neg), dim=1)
        logits = torch.div(logits, args.t)
        logits = logits.squeeze().contiguous()
        logits_softmax = torch.softmax(logits, 1)
        loss = self.criterion(logits)
        return loss, logits_softmax[:, 0], logits_softmax[:, 1:]

    def contrastive_loss_c(self, q, nodes, candidates):
        """

        :param q: [bsz, args.n_dim_v]
        :param nodes: [bsz, args.n_dim_v]
        :param candidates: [bsz, num_cands]
        :return:
        """
        bsz = q.shape[0]
        v_pos, v_neg = self.c(self.table[nodes]), self.c(self.table[candidates])
        l_pos = torch.bmm(q.view(bsz, 1, -1), v_pos.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)
        l_neg = torch.bmm(v_neg, q.unsqueeze(-1)).squeeze()
        logits = torch.cat((l_pos, l_neg), dim=1)
        logits = logits.squeeze().contiguous()
        logits_softmax = torch.softmax(logits / args.t, 1)
        return self.criterion(logits), logits_softmax[:, 0], logits_softmax[:, 1:]

    def contrastive_loss_m(self, q, v_pos, v_neg):
        bsz = q.shape[0]
        l_pos = torch.bmm(q.view(bsz, 1, -1), v_pos.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)
        l_neg = torch.bmm(v_neg, q.unsqueeze(-1)).squeeze()
        logits = torch.cat((l_pos, l_neg), dim=1)
        logits = logits.squeeze().contiguous()
        logits_softmax = torch.softmax(logits, 1)
        return self.criterion(logits), logits_softmax[:, 0], logits_softmax[:, 1:]

    def cross_linkage_loss_m(self, q, v_pos, v_neg):
        bsz = q.shape[0]
        l_pos = torch.bmm(q.view(bsz, 1, -1), v_pos.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)
        l_neg = torch.bmm(v_neg, q.unsqueeze(-1)).squeeze()
        logits = torch.cat((l_pos, l_neg), dim=1)
        logits = logits.squeeze().contiguous()
        #labels = torch.zeros((bsz, 1 + v_neg.shape[1])).cuda().float()
        #loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return self.criterion_(logits / args.t), None, None

    def cross_linkage_loss(self, q, v_pos, v_neg):
        bsz = q.shape[0]
        pos_score = torch.log(torch.sigmoid(torch.sum(torch.mul(q.unsqueeze(1), v_pos.unsqueeze(1)), -1)) + 1e-6).sum(dim=1)
        v_neg = torch.neg(v_neg)
        neg_score = torch.log(torch.sigmoid(torch.bmm(v_neg, q.view(bsz, -1, 1))) + 1e-6).squeeze().sum(dim=1)
        loss = pos_score + neg_score
        return -loss.sum() / bsz

    def first_order_similarity(self, v_pos, v_neg):
        bsz = v_pos.shape[0]
        logits = torch.mul(v_pos, v_neg).sum(1)
        logits = torch.neg(logits)
        loss = torch.sigmoid(logits).sum()
        return loss / bsz

    def linkage_loss(self, q, nodes, neighbors, start, end):
        """
        calculate linking loss inner graph, use v to calcualte
        :param q: [bs, n_dim_v]
        :param neighbors: [bs, num_of_neighbors]
        :return:
        """
        bsz = q.shape[0]
        neg_index = np.random.randint(start, end - bsz, bsz).transpose()[:, np.newaxis]
        neg_nodes = torch.from_numpy(
            np.array([i for i in range(int(args.num_neighbors ** (4 / 3)))])[np.newaxis, :].repeat(bsz,
                                                                                                   0) + neg_index).cuda()
        v_pos, v_neg = self.c(self.table[neighbors]), self.c(self.table[neg_nodes])
        # mask = (neighbors == self.vocab_size - 1)
        pos_score = torch.log(torch.sigmoid(torch.sum(torch.mul(q.unsqueeze(1), v_pos), -1)) + 1e-6).sum(dim=1)
        # pos_score.data.masked_fill_(torch.all(mask, dim=1).unsqueeze(1).repeat((1, args.num_neighbors - 1)), float(0))
        v_neg = torch.neg(v_neg)
        neg_score = torch.log(torch.sigmoid(torch.bmm(v_neg, q.view(bsz, -1, 1))) + 1e-6).squeeze().sum(dim=1)
        loss = pos_score + neg_score
        return -loss.sum() / bsz

    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= args.momentum
            key_param.data += (1 - args.momentum) * query_param.data
        self.eval()
        # print("synchronize query & key model...")

    def aggregate_b(self, neighbors, adj):
        b = self.b[neighbors].cuda()
        b = self.semantic_encoder(b)
        b_prime = self.gat_b(b, adj)
        b_prime = b_prime.permute(0, 2, 1, 3)
        b_prime = self.out_b(b_prime.reshape(b_prime.shape[0], b_prime.shape[1], -1))
        return b_prime

    def aggregate_p(self, neighbors, adj):
        u = self.b[neighbors].cuda()
        u = self.semantic_encoder(u)
        u_prime = self.gat_p(u, adj)
        u_prime = u_prime.permute(0, 2, 1, 3)
        u_prime = self.out_p(u_prime.reshape(u_prime.shape[0], u_prime.shape[1], -1))
        return u_prime

    def aggregate_u(self, neighbors, adj):
        u = self.u(self.fake_table[neighbors])
        u_prime = self.gat_u(u, adj)
        u_prime = u_prime.permute(0, 2, 1, 3)
        u_prime = self.out_u(u_prime.reshape(u_prime.shape[0], u_prime.shape[1], -1))
        return u_prime

    def forward(self, nodes, neighbors, adj, parents, parents_adj):
        b = self.b[nodes].cuda()
        b = self.semantic_encoder(b)
        b_prime = self.aggregate_b(neighbors, adj)[:, 0, :]
        p_prime = self.aggregate_p(parents, parents_adj)[:, 0, :]
        u_prime = self.aggregate_u(neighbors, adj)[:, 0, :]
        v = torch.cat((u_prime, b_prime), dim=1)
        v = self.w2(v)
        v = self.w1(torch.cat((b, v, p_prime), dim=1))
        v = self.norm(v)
        return v

    def fine_tune(self, nodes, neighbors, adj, parents, parents_adj):
        b = self.b[nodes].cuda()
        b = self.semantic_encoder(b)
        b_prime = self.aggregate_b(neighbors, adj)[:, 0, :]
        p_prime = self.aggregate_p(parents, parents_adj)[:, 0, :]
        u_prime = self.aggregate_u(neighbors, adj)[:, 0, :]
        v = torch.cat((u_prime, b_prime), dim=1)
        v = self.w2(v)
        v = self.w1(torch.cat((b, v, p_prime), dim=1))
        v = self.norm(v)
        return v, u_prime, b_prime, p_prime, b


class Trainer(object):
    def __init__(self, training=True, seed=37):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.sources = ['mag', 'en_wiki']
        self.single_graph = len(self.sources) == 1
        self.loader = CrossNetworkLoaderWithoutCandidates(self.sources, training=True, batch_size=args.batch_size,
                                                          use_numpy_embedding=True)
        self.model = None
        self.iteration = 0
        if training:
            self.writer = SummaryWriter(log_dir=join(PROJ_DIR, 'log', args.model + '_' + args.time), comment=args.time)

            self.model = LinKG(self.loader.vocab_size, self.loader.unique_vocab_size, self.loader.num_neighbors,
                               self.loader.table, self.loader.semantics.base_embedding.astype(np.float32))
            self._model = LinKG(self.loader.vocab_size, self.loader.unique_vocab_size, self.loader.num_neighbors,
                                self.loader.table, self.loader.semantics.base_embedding.astype(np.float32))
            self.model = self.model.cuda()
            self._model = self.model.cuda()
            self._model.update(self.model)

            self.iteration = 0

            self.optimizer = Adam(self.model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        # a queue for summary tensors
        self.queue = [list()] * len(self.sources)
        self.summary_queue = list()

    def adjust_learning_rate(self, optimizer, iteration, epoch):
        lr = args.lr * (args.lr_decay ** (iteration // args.lr_decay_iteration))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def add_to_queue(self, q, graph_idx):
        if len(self.queue[graph_idx]) >= args.queue_length:
            self.queue[graph_idx].pop(0)
        self.queue[graph_idx].append(q.detach())

    def add_to_summary_queue(self, s):
        if len(self.summary_queue) >= args.queue_length:
            self.summary_queue.pop(0)
        self.summary_queue.append(s.detach())

    def get_summary(self):
        return torch.cat(tuple(self.summary_queue), dim=0).cuda()

    def get_queue(self, graph_idx):
        return torch.cat(tuple(self.queue[graph_idx]), dim=0).cuda()

    def save_model(self, to_save=False):
        path = join(OUT_DIR, args.model + '_' + args.time)
        if not exists(path):
            os.mkdir(path)
        if not exists(join(path, str(self.iteration))) and to_save:
            os.mkdir(join(path, str(self.iteration)))
        torch.save(self._model, join(path, 'model'))
        # np.save(join(path, 'unique_embedding.npy'), self.model.u.weight.cpu().data.numpy())
        write_json(args.__dict__, path, 'args.json', overwrite=True, indent=4)
        vlist = list()
        for nodes, features, neighbors, adj, contrast, graph_idx, parents, parents_adj in self.loader.evaluate_contrast_batch():
            self._model.eval()
            nodes, neighbors, adj, features, contrast, parents, parents_adj = torch.from_numpy(
                nodes), torch.from_numpy(
                neighbors), torch.BoolTensor(adj), torch.from_numpy(features.astype(np.float32)), torch.from_numpy(
                contrast), torch.from_numpy(parents), torch.from_numpy(parents_adj)
            nodes, features, neighbors, adj, contrast, parents, parents_adj = nodes.cuda(), features.cuda(), neighbors.cuda(), adj.cuda(), contrast.cuda(), parents.cuda(), parents_adj.cuda()
            v = self._model(nodes, neighbors, adj, parents, parents_adj)
            b = self._model.semantic_encoder(self._model.b[nodes].cuda())
            vlist.append(torch.cat((b, v), dim=1).detach().cpu().numpy())
        vlist = np.concatenate(tuple(vlist), axis=0)
        if len(self.loader.sources) == 2:
            mag, en_wiki = vlist[:self.loader.graph_size[0], :], vlist[self.loader.graph_size[0]:, :]
        elif len(self.loader.sources) == 1:
            if self.loader.sources[0] == 'mag':
                mag, en_wiki = vlist, np.array([])
            else:
                mag, en_wiki = np.array([]), vlist
        else:
            raise NotImplementedError()
        if to_save:
            np.save(join(path, str(self.iteration), 'mag.npy'), mag)
            np.save(join(path, str(self.iteration), 'en_wiki.npy'), en_wiki)
        if len(self.loader.sources) > 1:
            print("evaluate general linking...")
            self.loader.evaluate(path, mag, en_wiki)
            print("evaluate ambiguous samples...")
            result = self.loader.evaluate_with_sample(self.iteration, self.loader, path, mag, en_wiki)
            with open(join(path, 'ambiguous.txt'), 'a') as f:
                f.write("iteration:{}\n".format(self.iteration))
                f.write(result)
        print("model saved!")

    def load_model(self, path):
        self.model = torch.load(join(path, 'model'))
        self.model.cuda()
        # self.model.u = torch.from_numpy(np.load(join(path, 'unique_embedding.npy'))).cuda()
        # self._model.update(self.model)
        print("model load!")

    def train(self, start=0):
        tot_loss = 0
        for e in range(start, args.epoch):
            #print("preparing queue...", end=' ')
            for batch_idx, nodes, features, neighbors, adj, contrast, graph_idx, parents, parents_adj in self.loader:
                if (batch_idx + 1) % 1000 == 0:
                    print("save model...")
                    self.save_model()
                self.model.train()
                self.adjust_learning_rate(self.optimizer, self.iteration, e)
                nodes, neighbors, adj, features, contrast, parents, parents_adj = torch.from_numpy(
                    nodes), torch.from_numpy(
                    neighbors), torch.BoolTensor(adj), torch.from_numpy(features.astype(np.float32)), torch.from_numpy(
                    contrast), torch.from_numpy(parents), torch.from_numpy(parents_adj)
                nodes, features, neighbors, adj, contrast, parents, parents_adj = nodes.cuda(), features.cuda(), neighbors.cuda(), adj.cuda(), contrast.cuda(), parents.cuda(), parents_adj.cuda()
                self.optimizer.zero_grad()
                query_v = self.model(nodes, neighbors, adj, parents, parents_adj)

                # prepare candidates embedding
                bsz = contrast.shape[0]
                contrast = contrast.reshape(-1)
                data = self.loader.get_input(contrast.cpu().numpy(), graph_idx)
                data = [torch.from_numpy(d).cuda() for d in data]
                with torch.no_grad():
                    self._model.eval()
                    v_neg = self._model(data[0], data[2], data[3], data[5], data[6])
                    v_neg = v_neg.view(bsz, 10, -1)
                    v_pos = self._model(nodes, neighbors, adj, parents, parents_adj)

                # contrastive loss
                contrastive_loss, _, __ = self.model.contrastive_loss_m(query_v, v_pos, v_neg)
                self.writer.add_scalar(join(args.model, 'contrastive_loss'), contrastive_loss.data, self.iteration)
                # contrastive loss with queue
                #if len(self.queue[graph_idx]) > 0:
                #    contrast_queue_loss, _, __ = self.model.contrastive_loss(query_v, v_pos, self.get_queue(graph_idx))
                #    self.writer.add_scalar(join(args.model, 'contrast_queue_loss'), contrast_queue_loss.data, self.iteration)
                #else:
                #    contrast_queue_loss = 0
                #self.add_to_queue(v_pos, graph_idx)
                """
                # linkage loss
                summary, new_neighbors, index = list(), list(), list()
                neg_neighbors = list()
                for i in range(neighbors.shape[0]):
                    # prepare linkage loss
                    mask = neighbors[i, :] != -1
                    if not mask[1:].any():
                        continue
                    # summary.append(query_s[i, 1: mask.sum(), :].unsqueeze(0).sum(1))
                    weights = mask.float().cpu()
                    weights[0] = 0
                    index.append(i)
                    new_neighbors.append(
                        neighbors[i, torch.multinomial(weights, args.num_neighbors, replacement=True)].unsqueeze(0))
                    # prepare local global loss
                    if int(nodes[i].cpu()) in self.loader.bidirect_table:
                        neg = self.loader.neighbor_nodes[self.loader.bidirect_table[int(nodes[i].cpu())]]
                    else:
                        neg = torch.LongTensor([-1, -1])
                    while not (neg[1:] != -1).any():
                        neg = self.loader.neighbor_nodes[np.random.randint(0, self.loader.vocab_size)]
                    neg = torch.from_numpy(neg)
                    mask = neg != -1
                    weights = mask.float().cpu()
                    weights[0] = 0
                    neg = neg[torch.multinomial(weights, args.num_neighbors, replacement=True)]
                    neg_neighbors.append(neg.numpy())
                new_neighbors = torch.cat(tuple(new_neighbors), dim=0)
                index = torch.LongTensor(index).cuda()
                query_v = query_v[index]
                nodes = nodes[index]
                
                # linkage loss
                linkage_loss = self.model.linkage_loss(query_v, nodes, new_neighbors.cuda(),
                                                       sum(self.loader.graph_size[:graph_idx]),
                                                       sum(self.loader.graph_size[:graph_idx + 1]))
                writer.add_scalar(join(args.model, 'linkage_loss'), linkage_loss.data, self.iteration)
                """
                # cross linkage loss
                pos, neg, indices = [], [], []
                for i in range(neighbors.shape[0]):
                    node = int(nodes[i].cpu())
                    if node in self.loader.bidirect_table:
                        indices.append(i)
                        pos.append(self.loader.bidirect_table[node])
                        neg.append(self.loader.get_contrast(
                            self.loader.bidirect_table[node], 1 - graph_idx).tolist())
                if len(pos) <= 1:
                    cross_linkage_loss = 0
                else:
                    bsz = len(neg)
                    pos, neg = np.array(pos), np.array(neg).reshape(bsz * 10)
                    pos_data = self.loader.get_input(pos, graph_idx)
                    neg_data = self.loader.get_input(neg, 1 - graph_idx)
                    pos_data, neg_data = [torch.from_numpy(d).cuda() for d in pos_data], [torch.from_numpy(d).cuda() for
                                                                                          d
                                                                                          in neg_data]
                    indices = torch.LongTensor(indices).cuda()
                    with torch.no_grad():
                        try:
                            pos_value = self._model(pos_data[0], pos_data[2], pos_data[3], pos_data[5], pos_data[6])
                        except:
                            print(batch_idx, len(pos))
                            exit()
                        neg_value = self._model(neg_data[0], neg_data[2], neg_data[3], neg_data[5], neg_data[6])
                        neg_value = neg_value.reshape(bsz, 10, -1)
                    query_v_prime = query_v[indices]
                    # cross_linkage_loss, _, __ = self.model.contrastive_loss_m(query_v_prime, pos_value, neg_value)
                    cross_linkage_loss = self.model.cross_linkage_loss(query_v_prime, pos_value, neg_value)
                    self.writer.add_scalar(join(args.model, 'cross_loss'), cross_linkage_loss.data, self.iteration)

                # total loss
                loss = contrastive_loss #+ args.loss_weight * cross_linkage_loss#+ args.loss_weight * cross_linkage_loss  # + linkage_loss

                # self.add_to_queue(nodes, graph_idx)
                self.iteration += 1
                # self.add_to_queue(key_u, graph_idx)
                # self.add_to_summary_queue(key_s)
                self.writer.add_scalar(join(args.model, 'loss'), loss.data, self.iteration)

                # writer.add_scalar(join(args.model, 'c_prob_times'), c_prob_pos.sum().data / c_prob_neg.sum().data,
                #                  self.iteration)
                # writer.add_scalar(join(args.model, 'linkage_loss'), linkage_loss.data, self.iteration)
                # writer.add_scalar(join(args.model, 'contrastive_loss'), contrastive_loss.data, self.iteration)
                # writer.add_scalar(join(args.model, 'local_loss'), local_loss.data, self.iteration)
                loss.backward(retain_graph=True)
                self.optimizer.step()

                if batch_idx % 100 == 0:
                    print('epoch {} batch {} loss {}'.format(e, batch_idx, tot_loss / 100))
                    tot_loss = 0
                else:
                    tot_loss += loss.detach().cpu().data

                # update
                # if (batch_idx + 1) % args.update_freq == 0:
                self._model.update(self.model)
            # self._model.update(self.model)
            self.save_model(True)


if __name__ == '__main__':
    training = False
    if training:
        trainer = Trainer()
        trainer.train()
    else:
        trainer = Trainer(training=False)
        trainer.fine_tune(join('LinKG_MAG_ENWIKI_v4_2020-02-08 17:12:38.150489'))#, 1e-4)
        #trainer.baseline("DISTMULT")
