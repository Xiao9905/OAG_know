import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class DGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout, concat, bias=False, cuda=False):
        super(DGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a1 = Parameter(torch.FloatTensor(out_features, 1))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)

        self.a2 = Parameter(torch.FloatTensor(out_features, 1))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.concat = concat
        self.cuda = cuda

        # self.bn = nn.BatchNorm1d(out_features, affine = False, eps=1e-6, momentum = 0.1)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features, 1))
            nn.init.xavier_uniform_(self.bias.data, gain=1.414)
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()

        self.special_spmm = SpecialSpmm()
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj):
        support = torch.mm(input, self.W)

        n = input.size(0)
        l_D = F.sigmoid(self.leakyrelu(torch.mm(support, self.a1).squeeze()))
        r_D = F.sigmoid(self.leakyrelu(torch.mm(support, self.a2).squeeze()))

        ind_D = torch.LongTensor([range(n), range(n)])
        # if self.cuda:
        #     ind_D = ind_D.type(torch.cuda.LongTensor)

        # L_D = torch.sparse.FloatTensor(ind_D, l_D, torch.Size([n, n]))
        # R_D = torch.sparse.FloatTensor(ind_D, r_D, torch.size([n, n]))

        # r_sparse_diag = torch.sparse_coo_tensor(ind_D, r_D, torch.Size([n, n]))
        # if self.cuda:
        #     r_sparse_diag = r_sparse_diag.cuda()
        # output = torch.sparse.mm(r_sparse_diag, support)
        output = self.special_spmm(ind_D, r_D, torch.Size([n, n]), support)
        output = torch.spmm(adj, output)
        # output = torch.sparse.mm(adj, output)
        # output = torch.spmm(adj, output)
        # output = torch.spmm(adj, output)
        output = self.special_spmm(ind_D, l_D, torch.Size([n, n]), output)
        # l_sparse_diag = torch.sparse_coo_tensor(ind_D, l_D, torch.Size([n, n]))
        # output = torch.sparse.mm(l_sparse_diag, output)

        alpha = 0.001
        output2 = alpha * torch.spmm(adj, support)
        output = output + output2

        unit = torch.ones(size=(n, 1))
        if self.cuda:
            unit = unit.cuda()
        sumnorm = self.special_spmm(ind_D, r_D, torch.Size([n, n]), unit)
        # sumnorm = torch.sparse.mm(r_sparse_diag, torch.ones(size=(n, 1)))
        sumnorm = torch.spmm(adj, sumnorm)
        # sumnorm = torch.sparse.mm(adj, sumnorm)
        sumnorm = self.special_spmm(ind_D, l_D, torch.Size([n, n]), sumnorm)
        # sumnorm = torch.sparse.mm(l_sparse_diag, sumnorm)

        sumnorm_2 = alpha * torch.spmm(adj, unit)
        sumnorm = sumnorm + sumnorm_2

        output = output.div(sumnorm)

        if self.concat:
            # if this layer is not last layer,
            output = F.elu(output)
            # if this layer is last layer,

        if self.bias is not None:
            return output + self.bias.squeeze()
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SpGraphAttentionLayerGAT(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayerGAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        N = input.size()[0]
        # edge = adj.nonzero().t()
        edge = adj._indices()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda())
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphAttentionLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat, bias=False, cuda=False):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.)

        self.a1 = nn.Parameter(torch.FloatTensor(out_features, 1))
        nn.init.xavier_uniform_(self.a1.data, gain=1.)

        self.a2 = nn.Parameter(torch.FloatTensor(out_features, 1))
        nn.init.xavier_uniform_(self.a2.data, gain=1.)

        self.concat = concat
        self.cuda = cuda

        # self.bn = nn.BatchNorm1d(out_features, affine = False, eps=1e-6, momentum = 0.1)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features, 1))
            nn.init.xavier_uniform_(self.bias.data, gain=1.)
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()

        self.special_spmm = SpecialSpmm()
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, Adj):
        n = input.size(0)

        support = torch.mm(input, self.W)

        adj = torch.sparse.FloatTensor(Adj._indices(), self.dropout(Adj._values()), torch.Size([n, n]))

        # l_D = F.sigmoid(self.leakyrelu(torch.mm(support, self.a1)))
        r_D = torch.sigmoid((torch.mm(support, self.a2)))

        output = r_D * support
        output = torch.spmm(adj, output)
        # output = self.dropout(output)
        # output = l_D * output

        unit = torch.ones(size=(n, 1))
        if self.cuda:
            unit = unit.cuda()

        sumnorm = r_D * unit
        sumnorm = torch.spmm(Adj, sumnorm)

        output = output.div(sumnorm)

        l_D = F.softplus((torch.mm(output, self.a1)))
        l_D = self.dropout(l_D)
        output = torch.clamp(output, min=0.0) + l_D * torch.clamp(output, max=0.0)

        if self.concat:
            # if this layer is not last layer,
            output = F.elu(output)
            # if this layer is last layer,

        if self.bias is not None:
            return output + self.bias.squeeze()
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)

        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        n = h.size(0)  # h is of size n x f_in
        h_prime = torch.matmul(h.unsqueeze(0), self.w)  # n_head x n x f_out
        attn_src = torch.bmm(h_prime, self.a_src)  # n_head x n x 1
        attn_dst = torch.bmm(h_prime, self.a_dst)  # n_head x n x 1
        attn = attn_src.expand(-1, -1, n) + attn_dst.expand(-1, -1, n).permute(0, 2, 1)  # n_head x n x n

        attn = self.leaky_relu(attn)
        attn.data.masked_fill_(1 - adj, float("-inf"))
        attn = self.softmax(attn)  # n_head x n x n
        attn = self.dropout(attn)
        output = torch.bmm(attn, h_prime)  # n_head x n x f_out

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        # self.n_type_nodes = n_type_nodes
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))  # self.n_type_nodes))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))  # self.n_type_nodes))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):  # , v_types):
        bs, n = h.size()[:2]  # h is of size bs x n x f_in
        h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        # v_types = v_types.unsqueeze(1)
        # v_types = v_types.expand(-1, self.n_head, -1, -1)
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        # attn_src = torch.sum(torch.mul(attn_src, v_types),  dim=3, keepdim=True)
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        # attn_dst = torch.sum(torch.mul(attn_dst, v_types), dim=3, keepdim=True)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3,
                                                                                       2)  # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        mask = ~(adj.unsqueeze(1) | torch.eye(adj.shape[-1]).bool().cuda())  # bs x 1 x n x n
        attn.data.masked_fill_(mask, float("-inf"))
        attn = self.softmax(attn)  # bs x n_head x n x n
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)  # bs x n_head x n x f_out
        if self.bias is not None:
            return output + self.bias
        else:
            return output
