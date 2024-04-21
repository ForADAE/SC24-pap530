from torch import nn
import torch
import math

import dgl.function as fn

from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from torch.nn import Identity
from dgl.base import DGLError


def count_degrees(graph):
    degrees = graph.in_degrees().tolist()

    degree_counts = Counter(degrees)

    for degree, count in degree_counts.items():
        print(f"Degree: {degree}, Count: {count}")


def get_memory_usage():
    get_memory_usage.prev_memory_usage = getattr(get_memory_usage,
                                                 'prev_memory_usage', 0)
    current_memory_usage = torch.cuda.memory_allocated()
    memory_diff = current_memory_usage - get_memory_usage.prev_memory_usage
    get_memory_usage.prev_memory_usage = current_memory_usage
    return memory_diff, current_memory_usage


def log_memory_usage(message):
    return
    memory_diff, current_memory_usage = get_memory_usage()
    peak_memory_usage = torch.cuda.max_memory_allocated()
    print(
        f"[MEM_LOG] {memory_diff:>20} {current_memory_usage:>20} {peak_memory_usage:>20} {message}"
    )


class GCNLayer(nn.Module):

    def __init__(self, in_feats, out_feats, bias=True, use_pp=False):
        super(GCNLayer, self).__init__()
        self.use_pp = use_pp
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, graph, feat, in_norm, out_norm):
        log_memory_usage("GCNLayer.forward")
        with graph.local_scope():
            if self.training:
                if self.use_pp:
                    feat = self.linear(feat)
                    log_memory_usage("feat = self.linear(feat)")
                else:
                    in_norm = in_norm.unsqueeze(1)
                    log_memory_usage("in_norm = in_norm.unsqueeze(1)")
                    out_norm = out_norm.unsqueeze(1)
                    log_memory_usage("out_norm = out_norm.unsqueeze(1)")
                    graph.nodes['_U'].data['h'] = feat / out_norm
                    log_memory_usage(
                        "graph.nodes['_U'].data['h'] = feat / out_norm")
                    graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                           fn.sum(msg='m', out='h'),
                                           etype='_E')
                    log_memory_usage(
                        "graph['_E'].update_all(fn.copy_u(u='h', out='m'), fn.sum(msg='m', out='h'), etype='_E')"
                    )
                    feat = self.linear(graph.nodes['_V'].data['h'] / in_norm)
                    log_memory_usage(
                        "feat = self.linear(graph.nodes['_V'].data['h'] / in_norm)"
                    )
            else:
                in_norm = torch.sqrt(graph.in_degrees()).unsqueeze(1)
                log_memory_usage(
                    "in_norm = torch.sqrt(graph.in_degrees()).unsqueeze(1)")
                out_norm = torch.sqrt(graph.out_degrees()).unsqueeze(1)
                log_memory_usage(
                    "out_norm = torch.sqrt(graph.out_degrees()).unsqueeze(1)")
                graph.ndata['h'] = feat / out_norm
                log_memory_usage("graph.ndata['h'] = feat / out_norm")
                graph.update_all(fn.copy_u(u='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                log_memory_usage(
                    "graph.update_all(fn.copy_u(u='h', out='m'), fn.sum(msg='m', out='h'))"
                )
                feat = self.linear(graph.ndata.pop('h') / in_norm)
                log_memory_usage(
                    "feat = self.linear(graph.ndata.pop('h') / in_norm)")
        return feat


class GraphSAGELayer(nn.Module):

    def __init__(self, in_feats, out_feats, bias=True, use_pp=False):
        super(GraphSAGELayer, self).__init__()
        self.use_pp = use_pp
        if self.use_pp:
            self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)
        else:
            self.linear1 = nn.Linear(in_feats, out_feats, bias=bias)
            self.linear2 = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_pp:
            stdv = 1. / math.sqrt(self.linear.weight.size(1))
            self.linear.weight.data.uniform_(-stdv, stdv)
            if self.linear.bias is not None:
                self.linear.bias.data.uniform_(-stdv, stdv)
        else:
            stdv = 1. / math.sqrt(self.linear1.weight.size(1))
            self.linear1.weight.data.uniform_(-stdv, stdv)
            self.linear2.weight.data.uniform_(-stdv, stdv)
            if self.linear1.bias is not None:
                self.linear1.bias.data.uniform_(-stdv, stdv)
                self.linear2.bias.data.uniform_(-stdv, stdv)

    def forward(self, graph, feat, in_norm):
        with graph.local_scope():
            if self.training:
                if self.use_pp:
                    feat = self.linear(feat)
                else:
                    degs = in_norm.unsqueeze(1)
                    num_dst = graph.num_nodes('_V')
                    graph.nodes['_U'].data['h'] = feat
                    graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                           fn.sum(msg='m', out='h'),
                                           etype='_E')
                    ah = graph.nodes['_V'].data['h'] / degs
                    feat = self.linear1(feat[0:num_dst]) + self.linear2(ah)
            else:
                degs = graph.in_degrees().unsqueeze(1)
                graph.ndata['h'] = feat
                graph.update_all(fn.copy_u(u='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                ah = graph.ndata.pop('h') / degs
                if self.use_pp:
                    feat = self.linear(torch.cat((feat, ah), dim=1))
                else:
                    feat = self.linear1(feat) + self.linear2(ah)
        return feat


class GATConv_original(nn.Module):

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(GATConv_original, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats,
                                    out_feats * num_heads,
                                    bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats,
                                    out_feats * num_heads,
                                    bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats,
                                out_feats * num_heads,
                                bias=False)
        self.attn_l = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self._in_dst_feats,
                                        num_heads * out_feats,
                                        bias=bias)
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(
                torch.FloatTensor(size=(num_heads * out_feats, )))
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            A 1D tensor of edge weight values.  Shape: :math:`(|E|,)`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run.")

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])

                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(*src_prefix_shape,
                                                   self._num_heads,
                                                   self._out_feats)
                    feat_dst = self.fc(h_dst).view(*dst_prefix_shape,
                                                   self._num_heads,
                                                   self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(*src_prefix_shape,
                                                       self._num_heads,
                                                       self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape,
                                                       self._num_heads,
                                                       self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)

                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(), ) + dst_prefix_shape[1:]
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            graph.apply_edges(fn.u_add_v("el", "er", "e"))

            e = self.leaky_relu(graph.edata.pop("e"))
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                    1, self._num_heads, 1).transpose(0, 2)
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))

            rst = graph.dstdata["ft"]
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1,
                                                 self._out_feats)
                rst = rst + resval
            if self.has_explicit_bias:
                rst = rst + self.bias.view(*((1, ) * len(dst_prefix_shape)),
                                           self._num_heads, self._out_feats)
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


class GraphSAGELayer_lstm(nn.Module):

    def __init__(self, in_feats, out_feats, bias=True, use_pp=False):
        super(GraphSAGELayer_lstm, self).__init__()
        self.use_pp = use_pp
        if self.use_pp:
            self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)
        else:
            self.linear1 = nn.Linear(in_feats, out_feats, bias=bias)
            self.linear2 = nn.Linear(in_feats, out_feats, bias=bias)
        self.in_feats = in_feats
        if self.use_pp:
            self.lstm = nn.LSTM(in_feats * 2, in_feats, batch_first=True)
        else:
            self.lstm = nn.LSTM(in_feats, in_feats, batch_first=True)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_pp:
            stdv = 1. / math.sqrt(self.linear.weight.size(1))
            self.linear.weight.data.uniform_(-stdv, stdv)
            if self.linear.bias is not None:
                self.linear.bias.data.uniform_(-stdv, stdv)
        else:
            stdv = 1. / math.sqrt(self.linear1.weight.size(1))
            self.linear1.weight.data.uniform_(-stdv, stdv)
            self.linear2.weight.data.uniform_(-stdv, stdv)
            if self.linear1.bias is not None:
                self.linear1.bias.data.uniform_(-stdv, stdv)
                self.linear2.bias.data.uniform_(-stdv, stdv)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self.in_feats)),
            m.new_zeros((1, batch_size, self.in_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"h": rst.squeeze(0)}

    def forward(self, graph, feat, in_norm):
        with graph.local_scope():
            if self.training:
                if self.use_pp:
                    feat = self.linear(feat)
                else:
                    degs = in_norm.unsqueeze(1)
                    num_dst = graph.num_nodes('_V')
                    graph.nodes['_U'].data['h'] = feat
                    graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                           self._lstm_reducer,
                                           etype='_E')
                    ah = graph.nodes['_V'].data['h'] / degs
                    feat = self.linear1(feat[0:num_dst]) + self.linear2(ah)
            else:
                degs = graph.in_degrees().unsqueeze(1)
                graph.ndata['h'] = feat
                graph.update_all(fn.copy_u(u='h', out='m'), self._lstm_reducer)
                ah = graph.ndata.pop('h') / degs
                if self.use_pp:
                    feat = self.linear(torch.cat((feat, ah), dim=1))
                else:
                    feat = self.linear1(feat) + self.linear2(ah)
        return feat
