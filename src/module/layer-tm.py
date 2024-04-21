from torch import nn
import torch
import math

import dgl.function as fn
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from torch.nn import Identity
from dgl.base import DGLError

from .rnn import LSTM

import pytm

from collections import Counter


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


class Linear_with(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with torch.autograd.graph.save_on_cpu():
            return torch.nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


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
        key_feat = pytm.tensor_manage(feat)
        log_memory_usage("key_feat = pytm.tensor_manage(feat)")
        with graph.local_scope():
            if self.training:
                if self.use_pp:
                    feat = self.linear(feat)
                    log_memory_usage("feat = self.linear(feat)")
                    pytm.unlock_managed_tensor(key_feat)
                    log_memory_usage("pytm.unlock_managed_tensor(key_feat)")
                else:
                    in_norm = in_norm.unsqueeze(1)
                    log_memory_usage("in_norm = in_norm.unsqueeze(1)")
                    out_norm = out_norm.unsqueeze(1)
                    log_memory_usage("out_norm = out_norm.unsqueeze(1)")
                    graph.nodes['_U'].data['h'] = feat / out_norm
                    log_memory_usage(
                        "graph.nodes['_U'].data['h'] = feat / out_norm")

                    key_U = pytm.tensor_manage(graph.nodes['_U'].data['h'])
                    log_memory_usage(
                        "key_U = pytm.tensor_manage(graph.nodes['_U'].data['h'])"
                    )

                    graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                           fn.sum(msg='m', out='h'),
                                           etype='_E')
                    log_memory_usage(
                        "graph['_E'].update_all(fn.copy_u(u='h', out='m'), fn.sum(msg='m', out='h'), etype='_E')"
                    )

                    pytm.unlock_managed_tensor(key_U)
                    log_memory_usage("pytm.unlock_managed_tensor(key_U)")

                    feat = self.linear(graph.nodes['_V'].data['h'] / in_norm)
                    log_memory_usage(
                        "feat = self.linear(graph.nodes['_V'].data['h'] / in_norm)"
                    )
                    pytm.unlock_managed_tensor(key_feat)
                    log_memory_usage("pytm.unlock_managed_tensor(key_feat)")

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
                pytm.unlock_managed_tensor(key_feat)
                log_memory_usage("pytm.unlock_managed_tensor(key_feat)")

        return feat


class GCNLayer_with(nn.Module):

    def __init__(self, in_feats, out_feats, bias=True, use_pp=False):
        super(GCNLayer_with, self).__init__()
        self.use_pp = use_pp
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, graph, feat, in_norm, out_norm):
        with torch.autograd.graph.save_on_cpu():
            log_memory_usage("torch.autograd.graph.save_on_cpu()")
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
                        feat = self.linear(graph.nodes['_V'].data['h'] /
                                           in_norm)
                        log_memory_usage(
                            "feat = self.linear(graph.nodes['_V'].data['h'] / in_norm)"
                        )
                else:
                    in_norm = torch.sqrt(graph.in_degrees()).unsqueeze(1)
                    log_memory_usage(
                        "in_norm = torch.sqrt(graph.in_degrees()).unsqueeze(1)"
                    )
                    out_norm = torch.sqrt(graph.out_degrees()).unsqueeze(1)
                    log_memory_usage(
                        "out_norm = torch.sqrt(graph.out_degrees()).unsqueeze(1)"
                    )
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
        key_feat = pytm.tensor_manage(feat)
        with graph.local_scope():
            if self.training:
                if self.use_pp:
                    feat = self.linear(feat)
                    pytm.unlock_managed_tensor(key_feat)
                else:
                    degs = in_norm.unsqueeze(1)
                    num_dst = graph.num_nodes('_V')
                    graph.nodes['_U'].data['h'] = feat
                    key_U = pytm.tensor_manage(graph.nodes['_U'].data['h'])
                    graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                           fn.sum(msg='m', out='h'),
                                           etype='_E')
                    ah = graph.nodes['_V'].data['h'] / degs
                    feat = self.linear1(feat[0:num_dst]) + self.linear2(ah)
                    pytm.unlock_managed_tensor(key_feat)
            else:
                degs = graph.in_degrees().unsqueeze(1)
                graph.ndata['h'] = feat
                graph.update_all(fn.copy_u(u='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                ah = graph.ndata.pop('h') / degs
                if self.use_pp:
                    feat = self.linear(torch.cat((feat, ah), dim=1))
                    pytm.unlock_managed_tensor(key_feat)
                else:
                    feat = self.linear1(feat) + self.linear2(ah)
                    pytm.unlock_managed_tensor(key_feat)
        return feat


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
            self.lstm = LSTM(in_feats * 2, in_feats, batch_first=True)
        else:
            self.lstm = LSTM(in_feats, in_feats, batch_first=True)
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
        if pytm.lstm_swap():
            with torch.autograd.graph.save_on_cpu():
                _, (rst, _) = self.lstm(m, h)
        else:
            _, (rst, _) = self.lstm(m, h)

        return {"h": rst.squeeze(0)}

    def forward(self, graph, feat, in_norm):
        key_feat = pytm.tensor_manage(feat)
        with graph.local_scope():
            if self.training:
                if self.use_pp:
                    feat = self.linear(feat)
                    pytm.unlock_managed_tensor(key_feat)
                else:
                    degs = in_norm.unsqueeze(1)
                    num_dst = graph.num_nodes('_V')
                    graph.nodes['_U'].data['h'] = feat
                    graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                           self._lstm_reducer,
                                           etype='_E')
                    ah = graph.nodes['_V'].data['h'] / degs
                    feat = self.linear1(feat[0:num_dst]) + self.linear2(ah)
                    pytm.unlock_managed_tensor(key_feat)
            else:
                degs = graph.in_degrees().unsqueeze(1)
                graph.ndata['h'] = feat
                graph.update_all(fn.copy_u(u='h', out='m'), self._lstm_reducer)
                ah = graph.ndata.pop('h') / degs
                if self.use_pp:
                    feat = self.linear(torch.cat((feat, ah), dim=1))
                    pytm.unlock_managed_tensor(key_feat)
                else:
                    feat = self.linear1(feat) + self.linear2(ah)
                    pytm.unlock_managed_tensor(key_feat)
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


class GATConv_with(nn.Module):

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
        super(GATConv_with, self).__init__()
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

        log_memory_usage("GATConv.forward")
        with torch.autograd.graph.save_on_cpu():

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
                    log_memory_usage(" src_prefix_shape = feat[0].shape[:-1]")
                    dst_prefix_shape = feat[1].shape[:-1]
                    log_memory_usage(" dst_prefix_shape = feat[1].shape[:-1]")
                    h_src = self.feat_drop(feat[0])
                    log_memory_usage(" h_src = self.feat_drop(feat[0])")
                    h_dst = self.feat_drop(feat[1])
                    log_memory_usage(" h_dst = self.feat_drop(feat[1])")

                    if not hasattr(self, "fc_src"):
                        feat_src = self.fc(h_src).view(*src_prefix_shape,
                                                       self._num_heads,
                                                       self._out_feats)
                        log_memory_usage(" feat_src = self.fc(h_src).view")
                        feat_dst = self.fc(h_dst).view(*dst_prefix_shape,
                                                       self._num_heads,
                                                       self._out_feats)
                        log_memory_usage(" feat_dst = self.fc(h_dst).view")
                    else:
                        feat_src = self.fc_src(h_src).view(
                            *src_prefix_shape, self._num_heads,
                            self._out_feats)
                        feat_dst = self.fc_dst(h_dst).view(
                            *dst_prefix_shape, self._num_heads,
                            self._out_feats)
                else:
                    src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                    h_src = h_dst = self.feat_drop(feat)

                    feat_src = feat_dst = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    if graph.is_block:
                        feat_dst = feat_src[:graph.number_of_dst_nodes()]
                        h_dst = h_dst[:graph.number_of_dst_nodes()]
                        dst_prefix_shape = (graph.number_of_dst_nodes(),
                                            ) + dst_prefix_shape[1:]
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                log_memory_usage(
                    " el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)")
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                log_memory_usage(
                    " er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)")

                graph.srcdata.update({"ft": feat_src, "el": el})
                log_memory_usage(
                    " graph.srcdata.update({'ft': feat_src, 'el': el})")
                graph.dstdata.update({"er": er})
                log_memory_usage(" graph.dstdata.update({'er': er})")
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
                log_memory_usage(
                    " graph.apply_edges(fn.u_add_v('el', 'er', 'e'))")

                e = self.leaky_relu(graph.edata.pop("e"))
                log_memory_usage(" e = self.leaky_relu(graph.edata.pop('e'))")
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
                log_memory_usage(
                    " graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))"
                )

                if edge_weight is not None:
                    graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                        1, self._num_heads, 1).transpose(0, 2)
                graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
                log_memory_usage(
                    " graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))"
                )

                rst = graph.dstdata["ft"]
                log_memory_usage(" rst = graph.dstdata['ft']")
                if self.res_fc is not None:
                    resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1,
                                                     self._out_feats)
                    log_memory_usage(" resval = self.res_fc(h_dst).view")
                    rst = rst + resval
                    log_memory_usage(" rst = rst + resval")
                if self.has_explicit_bias:
                    rst = rst + self.bias.view(
                        *((1, ) * len(dst_prefix_shape)), self._num_heads,
                        self._out_feats)
                    log_memory_usage(" rst = rst + self.bias.view")
                if self.activation:
                    rst = self.activation(rst)
                    log_memory_usage(" rst = self.activation(rst)")

                if get_attention:
                    return rst, graph.edata["a"]
                else:
                    return rst


class GATConv_pytm(nn.Module):

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
        super(GATConv_pytm, self).__init__()
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

        log_memory_usage("GATConv.forward")

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
                key_feat_0 = pytm.tensor_manage(feat[0])
                log_memory_usage("key_feat_0 = pytm.tensor_manage(feat[0])")
                key_feat_1 = pytm.tensor_manage(feat[1])
                log_memory_usage("key_feat_1 = pytm.tensor_manage(feat[1])")
                src_prefix_shape = feat[0].shape[:-1]
                log_memory_usage("src_prefix_shape = feat[0].shape[:-1]")
                dst_prefix_shape = feat[1].shape[:-1]
                log_memory_usage("dst_prefix_shape = feat[1].shape[:-1]")
                h_src = self.feat_drop(feat[0])
                log_memory_usage("h_src = self.feat_drop(feat[0])")
                pytm.unlock_managed_tensor(key_feat_0)
                log_memory_usage("pytm.unlock_managed_tensor(key_feat_0)")

                h_dst = self.feat_drop(feat[1])
                log_memory_usage("h_dst = self.feat_drop(feat[1])")
                pytm.unlock_managed_tensor(key_feat_1)
                log_memory_usage("pytm.unlock_managed_tensor(key_feat_1)")

                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(*src_prefix_shape,
                                                   self._num_heads,
                                                   self._out_feats)
                    log_memory_usage("feat_src = self.fc(h_src).view")
                    feat_dst = self.fc(h_dst).view(*dst_prefix_shape,
                                                   self._num_heads,
                                                   self._out_feats)
                    log_memory_usage("feat_dst = self.fc(h_dst).view")
                else:
                    feat_src = self.fc_src(h_src).view(*src_prefix_shape,
                                                       self._num_heads,
                                                       self._out_feats)
                    log_memory_usage("feat_src = self.fc_src(h_src).view")
                    feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape,
                                                       self._num_heads,
                                                       self._out_feats)
                    log_memory_usage("feat_dst = self.fc_dst(h_dst).view")
            else:
                key_feat = pytm.tensor_manage(feat)
                log_memory_usage("key_feat = pytm.tensor_manage(feat)")
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                log_memory_usage(
                    "src_prefix_shape = dst_prefix_shape = feat.shape[:-1]")
                h_src = h_dst = self.feat_drop(feat)
                log_memory_usage("h_src = h_dst = self.feat_drop(feat)")
                pytm.unlock_managed_tensor(key_feat)
                log_memory_usage("pytm.unlock_managed_tensor(key_feat)")

                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                log_memory_usage("feat_src = feat_dst = self.fc(h_src).view")
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    log_memory_usage(
                        "feat_dst = feat_src[: graph.number_of_dst_nodes()]")
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    log_memory_usage(
                        "h_dst = h_dst[: graph.number_of_dst_nodes()]")
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(), ) + dst_prefix_shape[1:]
                    log_memory_usage(
                        "dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]"
                    )

            key_feat_src = pytm.tensor_manage(feat_src)
            log_memory_usage("key_feat_src = pytm.tensor_manage(feat_src)")
            key_feat_dst = pytm.tensor_manage(feat_dst)
            log_memory_usage("key_feat_dst = pytm.tensor_manage(feat_dst)")
            key_h_dst = pytm.tensor_manage(h_dst)
            log_memory_usage("key_h_dst = pytm.tensor_manage(h_dst)")
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            log_memory_usage(
                "el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)")
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            log_memory_usage(
                "er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)")
            key_el = pytm.tensor_manage(el)
            log_memory_usage("key_el = pytm.tensor_manage(el)")
            key_er = pytm.tensor_manage(er)
            log_memory_usage("key_er = pytm.tensor_manage(er)")

            graph.srcdata.update({"ft": feat_src, "el": el})
            log_memory_usage(
                "graph.srcdata.update({'ft': feat_src, 'el': el})")

            graph.dstdata.update({"er": er})
            log_memory_usage(
                "graph.srcdata.update({'ft': feat_src, 'el': el})")
            pytm.unlock_managed_tensor(key_feat_dst)
            log_memory_usage("pytm.unlock_managed_tensor(key_feat_dst)")
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            log_memory_usage("graph.apply_edges(fn.u_add_v('el', 'er', 'e'))")
            pytm.unlock_managed_tensor(key_el)
            log_memory_usage("pytm.unlock_managed_tensor(key_el)")
            pytm.unlock_managed_tensor(key_er)
            log_memory_usage("pytm.unlock_managed_tensor(key_er)")

            e = self.leaky_relu(graph.edata.pop("e"))
            log_memory_usage("e = self.leaky_relu(graph.edata.pop('e'))")
            key_e = pytm.tensor_manage(e)
            log_memory_usage("key_e = pytm.tensor_manage(e)")
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            log_memory_usage(
                "graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))")
            pytm.unlock_managed_tensor(key_e)
            log_memory_usage("pytm.unlock_managed_tensor(key_e)")

            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                    1, self._num_heads, 1).transpose(0, 2)
            log_memory_usage(
                "graph.edata['a'] = graph.edata['a'] * edge_weight.tile(1, self._num_heads, 1).transpose(0, 2)"
            )
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            log_memory_usage(
                "graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))"
            )

            rst = graph.dstdata["ft"]
            log_memory_usage("rst = graph.dstdata['ft']")
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1,
                                                 self._out_feats)
                log_memory_usage("resval = self.res_fc(h_dst).view")
                pytm.unlock_managed_tensor(key_h_dst)
                log_memory_usage("pytm.unlock_managed_tensor(key_h_dst)")
                rst = rst + resval
                log_memory_usage("rst = rst + resval")
            if self.has_explicit_bias:
                rst = rst + self.bias.view(*((1, ) * len(dst_prefix_shape)),
                                           self._num_heads, self._out_feats)
                log_memory_usage("rst = rst + self.bias.view")
            if self.activation:
                rst = self.activation(rst)
                log_memory_usage("rst = self.activation(rst)")

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


class GATConv(nn.Module):

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
        super(GATConv, self).__init__()
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
            self.fc = Linear_with(self._in_src_feats,
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
        log_memory_usage("GATConv.forward begin")
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
                key_feat_0 = pytm.tensor_manage(feat[0])
                log_memory_usage(" key_feat_0 = pytm.tensor_manage(feat[0]) ")
                key_feat_1 = pytm.tensor_manage(feat[1])
                log_memory_usage(" key_feat_1 = pytm.tensor_manage(feat[1]) ")

                src_prefix_shape = feat[0].shape[:-1]
                log_memory_usage(" src_prefix_shape = feat[0].shape[:-1] ")
                dst_prefix_shape = feat[1].shape[:-1]
                log_memory_usage(" dst_prefix_shape = feat[1].shape[:-1] ")

                h_src = self.feat_drop(feat[0])
                log_memory_usage(" h_src = self.feat_drop(feat[0]) ")
                key_h_src = pytm.tensor_manage(h_src)
                log_memory_usage(" key_h_src = pytm.tensor_manage(h_src) ")

                pytm.swap_to_cpu(key_feat_0)
                log_memory_usage(" >>pytm.swap_to_cpu(key_feat_0) ")

                h_dst = self.feat_drop(feat[1])
                log_memory_usage(" h_dst = self.feat_drop(feat[1]) ")
                key_h_dst = pytm.tensor_manage(h_dst)
                log_memory_usage(" key_h_dst = pytm.tensor_manage(h_dst) ")

                pytm.swap_to_cpu(key_feat_1)
                log_memory_usage(" >>pytm.swap_to_cpu(key_feat_1) ")

                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(*src_prefix_shape,
                                                   self._num_heads,
                                                   self._out_feats)
                    log_memory_usage(
                        " feat_src = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats) "
                    )
                    pytm.swap_to_cpu(key_h_src)
                    log_memory_usage(" >>pytm.swap_to_cpu(key_h_src) ")

                    feat_dst = self.fc(h_dst).view(*dst_prefix_shape,
                                                   self._num_heads,
                                                   self._out_feats)
                    log_memory_usage(
                        " feat_dst = self.fc(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats) "
                    )
                    pytm.swap_to_cpu(key_h_dst)
                    log_memory_usage(" >>pytm.swap_to_cpu(key_h_dst) ")

                    key_feat_dst = pytm.tensor_manage(feat_dst)
                    log_memory_usage(
                        " key_feat_dst = pytm.tensor_manage(feat_dst) ")

                else:
                    feat_src = self.fc_src(h_src).view(*src_prefix_shape,
                                                       self._num_heads,
                                                       self._out_feats)
                    log_memory_usage(
                        " feat_src = self.fc_src(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats) "
                    )

                    feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape,
                                                       self._num_heads,
                                                       self._out_feats)
                    log_memory_usage(
                        " feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats) "
                    )

            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                log_memory_usage(
                    " src_prefix_shape = dst_prefix_shape = feat.shape[:-1] ")
                h_src = h_dst = self.feat_drop(feat)
                log_memory_usage(" h_src = h_dst = self.feat_drop(feat) ")

                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                log_memory_usage(
                    " feat_src = feat_dst = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats) "
                )
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    log_memory_usage(
                        " feat_dst = feat_src[: graph.number_of_dst_nodes()] ")
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    log_memory_usage(
                        " h_dst = h_dst[: graph.number_of_dst_nodes()] ")
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(), ) + dst_prefix_shape[1:]
                    log_memory_usage(
                        " dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:] "
                    )
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            log_memory_usage(
                " el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1) ")
            key_el = pytm.tensor_manage(el)
            log_memory_usage(" key_el = pytm.tensor_manage(el) ")

            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            log_memory_usage(
                " er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1) ")
            key_er = pytm.tensor_manage(er)
            log_memory_usage(" key_er = pytm.tensor_manage(er) ")

            pytm.swap_to_cpu(key_feat_dst)
            log_memory_usage(" >>pytm.swap_to_cpu(key_feat_dst) ")

            graph.srcdata.update({"ft": feat_src, "el": el})
            log_memory_usage(" graph.srcdata.update({ft: feat_src, el: el}) ")

            graph.dstdata.update({"er": er})
            log_memory_usage(" graph.dstdata.update({er: er}) ")
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            log_memory_usage(" graph.apply_edges(fn.u_add_v(el, er, e)) ")

            pytm.swap_to_cpu(key_el)
            log_memory_usage(" >>pytm.swap_to_cpu(key_el) ")
            pytm.swap_to_cpu(key_er)
            log_memory_usage(" >>pytm.swap_to_cpu(key_er) ")

            e = self.leaky_relu(graph.edata.pop("e"))
            log_memory_usage(" e = self.leaky_relu(graph.edata.pop(e)) ")
            key_e = pytm.tensor_manage(e)
            log_memory_usage(" key_e = pytm.tensor_manage(e) ")
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            log_memory_usage(
                " graph.edata[a] = self.attn_drop(edge_softmax(graph, e)) ")

            pytm.swap_to_cpu(key_e)
            log_memory_usage(" >>pytm.swap_to_cpu(key_e) ")

            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                    1, self._num_heads, 1).transpose(0, 2)
                log_memory_usage(
                    " graph.edata[a] = graph.edata[a] * edge_weight.tile(1, self._num_heads, 1).transpose(0, 2) "
                )
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            log_memory_usage(
                " graph.update_all(fn.u_mul_e(ft, a, m), fn.sum(m, ft)) ")

            rst = graph.dstdata["ft"]
            log_memory_usage(" rst = graph.dstdata[ft] ")
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1,
                                                 self._out_feats)
                log_memory_usage(
                    " resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats) "
                )
                rst = rst + resval
                log_memory_usage(" rst = rst + resval ")
            if self.has_explicit_bias:
                rst = rst + self.bias.view(*((1, ) * len(dst_prefix_shape)),
                                           self._num_heads, self._out_feats)
                log_memory_usage(
                    " rst = rst + self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats) "
                )
            if self.activation:
                rst = self.activation(rst)
                log_memory_usage(" rst = self.activation(rst) ")

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


class GCNLayer_lstm(nn.Module):

    def __init__(self, in_feats, out_feats, bias=True, use_pp=False):
        super(GCNLayer_lstm, self).__init__()
        self.use_pp = use_pp
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.in_feats = in_feats
        if self.use_pp:
            self.lstm = LSTM(in_feats * 2, in_feats, batch_first=True)
        else:
            self.lstm = LSTM(in_feats, in_feats, batch_first=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)
        self.lstm.reset_parameters()

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
        if True:
            print("Swapping to CPU")
            with torch.autograd.graph.save_on_cpu():
                _, (rst, _) = self.lstm(m, h)
        else:
            print("Remain on GPU")
            _, (rst, _) = self.lstm(m, h)

        return {"h": rst.squeeze(0)}

    def forward(self, graph, feat, in_norm, out_norm):
        key_feat = pytm.tensor_manage(feat)

        with graph.local_scope():
            if self.training:
                if self.use_pp:
                    feat = self.linear(feat)
                    pytm.unlock_managed_tensor(key_feat)
                else:
                    in_norm = in_norm.unsqueeze(1)
                    out_norm = out_norm.unsqueeze(1)
                    graph.nodes['_U'].data['h'] = feat / out_norm
                    graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                           self._lstm_reducer,
                                           etype='_E')
                    feat = self.linear(graph.nodes['_V'].data['h'] / in_norm)
                    pytm.unlock_managed_tensor(key_feat)
            else:
                in_norm = torch.sqrt(graph.in_degrees()).unsqueeze(1)
                out_norm = torch.sqrt(graph.out_degrees()).unsqueeze(1)
                graph.ndata['h'] = feat / out_norm
                graph.update_all(fn.copy_u(u='h', out='m'), self._lstm_reducer)
                feat = self.linear(graph.ndata.pop('h') / in_norm)
                pytm.unlock_managed_tensor(key_feat)
        return feat
