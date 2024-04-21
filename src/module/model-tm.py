from module.layer import *
import dgl
from torch import nn
from module.sync_bn import SyncBatchNorm
from helper import context as ctx

from .sage_lstm import *

import pytm


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
        f"[MOL_LOG] {memory_diff:>20} {current_memory_usage:>20} {peak_memory_usage:>20} {message}"
    )


class GNNBase(nn.Module):

    def __init__(self,
                 layer_size,
                 activation,
                 use_pp=False,
                 dropout=0.5,
                 norm='layer',
                 n_linear=0):
        super(GNNBase, self).__init__()
        self.n_layers = len(layer_size) - 1
        self.layers = nn.ModuleList()
        self.activation = activation
        self.use_pp = use_pp
        self.n_linear = n_linear

        if norm is None:
            self.use_norm = False
        else:
            self.use_norm = True
            self.norm = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)


class GCN(GNNBase):

    def __init__(self,
                 layer_size,
                 activation,
                 use_pp,
                 dropout=0.5,
                 norm='layer',
                 train_size=None,
                 n_linear=0):
        super(GCN, self).__init__(layer_size, activation, use_pp, dropout,
                                  norm, n_linear)
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                self.layers.append(
                    GCNLayer(layer_size[i], layer_size[i + 1], use_pp=use_pp))
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(
                        nn.LayerNorm(layer_size[i + 1],
                                     elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(
                        SyncBatchNorm(layer_size[i + 1], train_size))
            use_pp = False

    def forward(self, g, feat, in_norm=None, out_norm=None):
        h = feat
        log_memory_usage("h = feat")
        for i in range(self.n_layers):

            key_dropout = pytm.tensor_manage(h)
            log_memory_usage("key_dropout = pytm.tensor_manage(h)")
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                h = self.dropout(h)
                log_memory_usage("h = self.dropout(h)")
            pytm.unlock_managed_tensor(key_dropout)
            log_memory_usage("pytm.unlock_managed_tensor(key_dropout)")

            if i < self.n_layers - self.n_linear:
                if self.training and (i > 0 or not self.use_pp):
                    h = ctx.buffer.update(i, h)
                    log_memory_usage("h = ctx.buffer.update(i, h)")
                h = self.layers[i](g, h, in_norm, out_norm)
                log_memory_usage("h = self.layers[i](g, h, in_norm, out_norm)")
            else:
                h = self.layers[i](h)
                log_memory_usage("h = self.layers[i](h)")

            if i < self.n_layers - 1:
                if self.use_norm:
                    key_norm = pytm.tensor_manage(h)
                    log_memory_usage("key_norm = pytm.tensor_manage(h)")
                    with torch.autograd.graph.save_on_cpu(pin_memory=True):
                        h = self.norm[i](h)
                    log_memory_usage("h = self.norm[i](h)")
                    pytm.unlock_managed_tensor(key_norm)
                    log_memory_usage("pytm.unlock_managed_tensor(key_norm)")
                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                    h = self.activation(h)
                log_memory_usage("h = self.activation(h)")

        return h


class GCN_with(GNNBase):

    def __init__(self,
                 layer_size,
                 activation,
                 use_pp,
                 dropout=0.5,
                 norm='layer',
                 train_size=None,
                 n_linear=0):
        super(GCN_with, self).__init__(layer_size, activation, use_pp, dropout,
                                       norm, n_linear)
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                self.layers.append(
                    GCNLayer_with(layer_size[i],
                                  layer_size[i + 1],
                                  use_pp=use_pp))
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(
                        nn.LayerNorm(layer_size[i + 1],
                                     elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(
                        SyncBatchNorm(layer_size[i + 1], train_size))
            use_pp = False

    def forward(self, g, feat, in_norm=None, out_norm=None):
        with torch.autograd.graph.save_on_cpu():
            h = feat
            log_memory_usage("h = feat")
            for i in range(self.n_layers):
                h = self.dropout(h)
                log_memory_usage("h = self.dropout(h)")
                if i < self.n_layers - self.n_linear:
                    if self.training and (i > 0 or not self.use_pp):
                        h = ctx.buffer.update(i, h)
                        log_memory_usage("h = ctx.buffer.update(i, h)")
                    h = self.layers[i](g, h, in_norm, out_norm)
                    log_memory_usage(
                        "h = self.layers[i](g, h, in_norm, out_norm)")
                else:
                    h = self.layers[i](h)
                    log_memory_usage("h = self.layers[i](h)")

                if i < self.n_layers - 1:
                    if self.use_norm:
                        h = self.norm[i](h)
                        log_memory_usage("h = self.norm[i](h)")
                    h = self.activation(h)
                    log_memory_usage("h = self.activation(h)")

            return h


class GraphSAGE(GNNBase):

    def __init__(self,
                 layer_size,
                 activation,
                 use_pp,
                 dropout=0.5,
                 norm='layer',
                 train_size=None,
                 n_linear=0):
        super(GraphSAGE, self).__init__(layer_size, activation, use_pp,
                                        dropout, norm, n_linear)
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                self.layers.append(
                    GraphSAGELayer(layer_size[i],
                                   layer_size[i + 1],
                                   use_pp=use_pp))
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(
                        nn.LayerNorm(layer_size[i + 1],
                                     elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(
                        SyncBatchNorm(layer_size[i + 1], train_size))
            use_pp = False

    def forward(self, g, feat, in_norm=None):
        h = feat
        log_memory_usage("h = feat")
        for i in range(self.n_layers):
            key_dropout = pytm.tensor_manage(h)
            log_memory_usage("key_dropout = pytm.tensor_manage(h)")
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                h = self.dropout(h)
                log_memory_usage("h = self.dropout(h)")
            pytm.unlock_managed_tensor(key_dropout)
            log_memory_usage("pytm.unlock_managed_tensor(key_dropout)")

            if i < self.n_layers - self.n_linear:
                if self.training and (i > 0 or not self.use_pp):
                    h = ctx.buffer.update(i, h)
                    log_memory_usage("h = ctx.buffer.update(i, h)")
                h = self.layers[i](g, h, in_norm)
                log_memory_usage("h = self.layers[i](g, h, in_norm)")
            else:
                h = self.layers[i](h)
                log_memory_usage("h = self.layers[i](h)")

            if i < self.n_layers - 1:
                if self.use_norm:
                    key_norm = pytm.tensor_manage(h)
                    log_memory_usage("key_norm = pytm.tensor_manage(h)")
                    with torch.autograd.graph.save_on_cpu(pin_memory=True):
                        h = self.norm[i](h)
                        log_memory_usage("h = self.norm[i](h)")
                    pytm.swap_to_cpu(key_norm)
                    log_memory_usage("pytm.swap_to_cpu(key_norm)")
                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                    h = self.activation(h)
                    log_memory_usage("h = self.activation(h)")

        return h


class GraphSAGE_lstm(GNNBase):

    def __init__(self,
                 layer_size,
                 activation,
                 use_pp,
                 dropout=0.5,
                 norm='layer',
                 train_size=None,
                 n_linear=0):
        super(GraphSAGE_lstm, self).__init__(layer_size, activation, use_pp,
                                             dropout, norm, n_linear)
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                self.layers.append(
                    GraphSAGELayer_lstm(layer_size[i],
                                        layer_size[i + 1],
                                        use_pp=use_pp))
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(
                        nn.LayerNorm(layer_size[i + 1],
                                     elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(
                        SyncBatchNorm(layer_size[i + 1], train_size))
            use_pp = False

    def forward(self, g, feat, in_norm=None):
        h = feat
        log_memory_usage("h = feat")
        for i in range(self.n_layers):
            key_dropout = pytm.tensor_manage(h)
            log_memory_usage("key_dropout = pytm.tensor_manage(h)")
            h = self.dropout(h)
            log_memory_usage("h = self.dropout(h)")
            pytm.unlock_managed_tensor(key_dropout)
            log_memory_usage("pytm.unlock_managed_tensor(key_dropout)")

            if i < self.n_layers - self.n_linear:
                if self.training and (i > 0 or not self.use_pp):
                    h = ctx.buffer.update(i, h)
                    log_memory_usage("h = ctx.buffer.update(i, h)")
                h = self.layers[i](g, h, in_norm)
                log_memory_usage("h = self.layers[i](g, h, in_norm)")
            else:
                h = self.layers[i](h)
                log_memory_usage("h = self.layers[i](h)")

            if i < self.n_layers - 1:
                if self.use_norm:
                    key_norm = pytm.tensor_manage(h)
                    log_memory_usage("key_norm = pytm.tensor_manage(h)")
                    h = self.norm[i](h)
                h = self.activation(h)

        _, _, logplan = pytm.get_lstm_plan()
        print(len(logplan))

        return h


class GAT(GNNBase):

    def __init__(self,
                 layer_size,
                 activation,
                 use_pp,
                 heads=1,
                 dropout=0.5,
                 norm='layer',
                 train_size=None,
                 n_linear=0):
        super(GAT, self).__init__(layer_size, activation, use_pp, dropout,
                                  norm, n_linear)
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                self.layers.append(
                    GATConv_pytm(layer_size[i], layer_size[i + 1], heads,
                                 dropout, dropout))
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(
                        nn.LayerNorm(layer_size[i + 1],
                                     elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(
                        SyncBatchNorm(layer_size[i + 1], train_size))

    def forward(self, g, feat):
        h = feat
        log_memory_usage("h = feat")
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                if self.training:
                    if i > 0 or not self.use_pp:
                        h1 = ctx.buffer.update(i, h)
                        log_memory_usage("h1 = ctx.buffer.update(i, h)")
                    else:
                        h1 = h
                        log_memory_usage("h1 = h")
                        h = h[0:g.num_nodes('_V')]
                        log_memory_usage("h = h[0:g.num_nodes('_V')]")
                    h = self.layers[i](g, (h1, h))
                    log_memory_usage("h = self.layers[i](g, (h1, h))")
                else:
                    h = self.layers[i](g, h)
                    log_memory_usage("h = self.layers[i](g, h)")
                h = h.mean(1)
                log_memory_usage("h = h.mean(1)")
            else:
                h = self.dropout(h)
                log_memory_usage("h = self.dropout(h)")
                h = self.layers[i](h)
                log_memory_usage("h = self.layers[i](h)")
            if i < self.n_layers - 1:
                if self.use_norm:
                    h = self.norm[i](h)
                    log_memory_usage("h = self.norm[i](h)")
                h = self.activation(h)
                log_memory_usage("h = self.activation(h)")
        return h


class GAT_with(GNNBase):

    def __init__(self,
                 layer_size,
                 activation,
                 use_pp,
                 heads=1,
                 dropout=0.5,
                 norm='layer',
                 train_size=None,
                 n_linear=0):
        super(GAT_with, self).__init__(layer_size, activation, use_pp, dropout,
                                       norm, n_linear)
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                self.layers.append(
                    GATConv_original(layer_size[i], layer_size[i + 1], heads,
                                     dropout, dropout))
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(
                        nn.LayerNorm(layer_size[i + 1],
                                     elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(
                        SyncBatchNorm(layer_size[i + 1], train_size))

    def forward(self, g, feat):
        with torch.autograd.graph.save_on_cpu():
            h = feat
            log_memory_usage("h = feat")
            for i in range(self.n_layers):
                if i < self.n_layers - self.n_linear:
                    if self.training:
                        if i > 0 or not self.use_pp:
                            h1 = ctx.buffer.update(i, h)
                            log_memory_usage("h1 = ctx.buffer.update(i, h)")
                        else:
                            h1 = h
                            log_memory_usage("h1 = h")
                            h = h[0:g.num_nodes('_V')]
                            log_memory_usage("h = h[0:g.num_nodes('_V')]")
                        h = self.layers[i](g, (h1, h))
                        log_memory_usage("h = self.layers[i](g, (h1, h))")
                    else:
                        h = self.layers[i](g, h)
                        log_memory_usage("h = self.layers[i](g, h)")
                    h = h.mean(1)
                    log_memory_usage("h = h.mean(1)")
                else:
                    h = self.dropout(h)
                    log_memory_usage("h = self.dropout(h)")
                    h = self.layers[i](h)
                    log_memory_usage("h = self.layers[i](h)")
                if i < self.n_layers - 1:
                    if self.use_norm:
                        h = self.norm[i](h)
                        log_memory_usage("h = self.norm[i](h)")
                    h = self.activation(h)
                    log_memory_usage("h = self.activation(h)")
            return h


class GraphSAGE_dgl(GNNBase):

    def __init__(self,
                 layer_size,
                 activation,
                 use_pp,
                 dropout=0.5,
                 norm='layer',
                 train_size=None,
                 n_linear=0):
        super(GraphSAGE_dgl, self).__init__(layer_size, activation, use_pp,
                                            dropout, norm, n_linear)
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                self.layers.append(
                    SAGEConv_dgl(layer_size[i],
                                 layer_size[i + 1],
                                 "lstm",
                                 use_pp=use_pp))
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(
                        nn.LayerNorm(layer_size[i + 1],
                                     elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(
                        SyncBatchNorm(layer_size[i + 1], train_size))
            use_pp = False

    def forward(self, g, feat, in_norm=None):
        h = feat
        for i in range(self.n_layers):
            h = self.dropout(h)
            if i < self.n_layers - self.n_linear:
                if self.training and (i > 0 or not self.use_pp):
                    h = ctx.buffer.update(i, h)
                h = self.layers[i](g, h)
            else:
                h = self.layers[i](h)

            if i < self.n_layers - 1:
                if self.use_norm:
                    h = self.norm[i](h)
                h = self.activation(h)

        return h


class GCN_lstm(GNNBase):

    def __init__(self,
                 layer_size,
                 activation,
                 use_pp,
                 dropout=0.5,
                 norm='layer',
                 train_size=None,
                 n_linear=0):
        super(GCN_lstm, self).__init__(layer_size, activation, use_pp, dropout,
                                       norm, n_linear)
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                self.layers.append(
                    GCNLayer_lstm(layer_size[i],
                                  layer_size[i + 1],
                                  use_pp=use_pp))
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(
                        nn.LayerNorm(layer_size[i + 1],
                                     elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(
                        SyncBatchNorm(layer_size[i + 1], train_size))
            use_pp = False

    def forward(self, g, feat, in_norm=None, out_norm=None):
        h = feat
        for i in range(self.n_layers):
            h = self.dropout(h)
            if i < self.n_layers - self.n_linear:
                if self.training and (i > 0 or not self.use_pp):
                    h = ctx.buffer.update(i, h)
                h = self.layers[i](g, h, in_norm, out_norm)
            else:
                h = self.layers[i](h)

            if i < self.n_layers - 1:
                if self.use_norm:
                    h = self.norm[i](h)
                h = self.activation(h)

        return h
