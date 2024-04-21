import os

import scipy
import dgl
from dgl.data import RedditDataset, YelpDataset
from dgl.distributed import partition_graph
from dgl.distributed.partition import partition_graph_topo
from dgl.distributed.partition import partition_graph_topo_weight
from helper.context import *
from ogb.nodeproppred import DglNodePropPredDataset
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

from dgl.data import AMDataset, FlickrDataset


class TransferTag:
    NODE = 0
    FEAT = 1
    DEG = 2


def sample_sub_graph(g, ratio):
    n = g.num_nodes()
    nodes = np.arange(int(n * ratio))
    sub_g = dgl.node_subgraph(graph=g, nodes=nodes)
    return sub_g


def sub_save_graph(g, path):
    dgl.save_graphs(path, g)


def load_sub_graph(path):
    sub_g = dgl.load_graphs(path)[0][0]
    return sub_g


def load_ogb_papers100m(name, data_path, sub_rate):
    if sub_rate != None:
        print("Subsampling the graph with rate", sub_rate)
        file_path = data_path + 'sub_graph/'
        file_name = 'papers100m_' + str(sub_rate) + '.bin'
        if file_name in os.listdir(file_path):
            print("Load subgraph from file")
            g = load_sub_graph(file_path + file_name)
        else:
            g = load_ogb_dataset(name, data_path)
            g = sample_sub_graph(g, sub_rate)
            sub_save_graph(g, file_path + file_name)
    else:
        g = load_ogb_dataset(name, data_path)

    return g


def load_ogb_dataset(name, data_path):
    print("begin to load ogb dataset", name, "from", data_path)
    dataset = DglNodePropPredDataset(name=name, root=data_path)
    split_idx = dataset.get_idx_split()
    g, label = dataset[0]
    if name == 'ogbn-mag':
        g = dgl.to_homogeneous(g)
        n_node = g.num_nodes()
        num_train = int(n_node * 0.9)
        num_val = int(n_node * 0.05)
        g.ndata['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
        g.ndata['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
        g.ndata['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
        g.ndata['train_mask'][:num_train] = True
        g.ndata['val_mask'][num_train:num_train + num_val] = True
        g.ndata['test_mask'][num_train + num_val:] = True
        label = torch.randint(0, 10, (n_node, ))
        g.ndata['label'] = label
        g.ndata['feat'] = torch.rand(n_node, 128)
        return g

    n_node = g.num_nodes()
    node_data = g.ndata

    if name == 'ogbn-proteins':
        label = torch.randint(0, 10, (n_node, ))
        node_data['label'] = label
        g.ndata['feat'] = torch.rand(n_node, 128)
    else:
        node_data['label'] = label.view(-1).long()

    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][split_idx["train"]] = True
    node_data['val_mask'][split_idx["valid"]] = True
    node_data['test_mask'][split_idx["test"]] = True
    return g


def load_rand_graph(num_nodes, num_edges):
    graph_path = 'dataset/rand/rand_%d_%d' % (num_nodes, num_edges)
    if os.path.exists(graph_path):
        g = dgl.load_graphs(graph_path)[0][0]
    else:
        g = dgl.rand_graph(num_nodes, num_edges)
        g = dgl.add_self_loop(g)
        dgl.save_graphs(graph_path, g)
    return g


def load_data(args):
    if args.dataset == 'reddit':
        data = RedditDataset(raw_dir=args.data_path)
        g = data[0]
    elif args.dataset == 'ogbn-arxiv':
        g = load_ogb_dataset('ogbn-arxiv', args.data_path)
    elif args.dataset == 'ogbn-proteins':
        g = load_ogb_dataset('ogbn-proteins', args.data_path)
    elif args.dataset == 'ogbn-products':
        g = load_ogb_dataset('ogbn-products', args.data_path)
    elif args.dataset == 'ogbn-mag':
        g = load_ogb_dataset('ogbn-mag', args.data_path)
    elif args.dataset == 'ogbn-papers100m':
        g = load_ogb_papers100m('ogbn-papers100M', args.data_path,
                                args.sub_rate)
    elif args.dataset == 'yelp':
        data = YelpDataset(raw_dir=args.data_path)
        g = data[0]
        g.ndata['label'] = g.ndata['label'].float()
        g.ndata['train_mask'] = g.ndata['train_mask'].bool()
        g.ndata['val_mask'] = g.ndata['val_mask'].bool()
        g.ndata['test_mask'] = g.ndata['test_mask'].bool()
        feats = g.ndata['feat']
        scaler = StandardScaler()
        scaler.fit(feats[g.ndata['train_mask']])
        feats = scaler.transform(feats)
        g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)
    elif args.dataset == 'am':
        data = AMDataset(raw_dir=args.data_path)
        g = data[0]
        g = dgl.to_homogeneous(g)

        label = torch.randint(0, data.num_classes, (g.num_nodes(), ))
        g.ndata['label'] = label
        g.ndata['feat'] = torch.rand(g.num_nodes(), 128)
        num_train = int(g.num_nodes() * 0.9)
        num_val = int(g.num_nodes() * 0.05)
        g.ndata['train_mask'] = torch.zeros(g.num_nodes(), dtype=torch.bool)
        g.ndata['val_mask'] = torch.zeros(g.num_nodes(), dtype=torch.bool)
        g.ndata['test_mask'] = torch.zeros(g.num_nodes(), dtype=torch.bool)
        g.ndata['train_mask'][:num_train] = True
        g.ndata['val_mask'][num_train:num_train + num_val] = True
        g.ndata['test_mask'][num_train + num_val:] = True

        feats = g.ndata['feat']
        scaler = StandardScaler()
        scaler.fit(feats[g.ndata['train_mask']])
        feats = scaler.transform(feats)
        g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)
    elif args.dataset == 'flickr':
        data = FlickrDataset(raw_dir=args.data_path)
        g = data[0]
        g.ndata['label'] = g.ndata['label'].long()
        g.ndata['train_mask'] = g.ndata['train_mask'].bool()
        g.ndata['val_mask'] = g.ndata['val_mask'].bool()
        g.ndata['test_mask'] = g.ndata['test_mask'].bool()
        feats = g.ndata['feat']
        scaler = StandardScaler()
        scaler.fit(feats[g.ndata['train_mask']])
        feats = scaler.transform(feats)
        g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)
    elif args.dataset[:4] == 'rand':
        n_nodes = int(args.dataset.split('_')[1])
        n_edges = int(args.dataset.split('_')[2])
        n_feat = int(args.dataset.split('_')[3])
        n_class = int(args.dataset.split('_')[4])
        g = load_rand_graph(n_nodes, n_edges)
        g.ndata['feat'] = torch.rand(n_nodes, n_feat)
        g.ndata['label'] = torch.randint(0, n_class, (n_nodes, ))
        num_train = int(n_nodes * args.train_ratio)
        num_val = int(n_nodes * args.val_ratio)
        g.ndata['train_mask'] = torch.zeros(n_nodes, dtype=torch.bool)
        g.ndata['val_mask'] = torch.zeros(n_nodes, dtype=torch.bool)
        g.ndata['test_mask'] = torch.zeros(n_nodes, dtype=torch.bool)
        g.ndata['train_mask'][:num_train] = True
        g.ndata['val_mask'][num_train:num_train + num_val] = True
        g.ndata['test_mask'][num_train + num_val:] = True
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    n_feat = g.ndata['feat'].shape[1]
    if g.ndata['label'].dim() == 1:
        n_class = g.ndata['label'].max().item() + 1
    else:
        n_class = g.ndata['label'].shape[1]

    g.edata.clear()

    if args.sub_rate != None and args.dataset != 'ogbn-papers100m':
        print("Subsampling the graph with rate", args.sub_rate)
        g = sample_sub_graph(g, args.sub_rate)

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    return g, n_feat, n_class


def graph_partition(args):

    g, n_feat, n_class = load_data(args)

    if args.inductive:
        g = g.subgraph(g.ndata['train_mask'])

    n_class = n_class
    n_feat = n_feat
    n_train = g.ndata['train_mask'].int().sum().item()

    graph_dir = os.path.join(args.part_path, args.graph_name)
    part_config = os.path.join(graph_dir, args.graph_name + '.json')
    if not os.path.exists(part_config):
        with g.local_scope():
            if args.inductive:
                g.ndata.pop('val_mask')
                g.ndata.pop('test_mask')
            g.ndata['in_deg'] = g.in_degrees()
            g.ndata['out_deg'] = g.out_degrees()
            partition_graph(g,
                            args.graph_name,
                            args.n_partitions,
                            graph_dir,
                            part_method=args.partition_method,
                            balance_edges=False,
                            objtype=args.partition_obj)

    with open(os.path.join(graph_dir, 'meta.json'), 'w') as f:
        json.dump({
            'n_feat': n_feat,
            'n_class': n_class,
            'n_train': n_train
        }, f)


def graph_partition_map(args):

    g, n_feat, n_class = load_data(args)
    if args.inductive:
        g = g.subgraph(g.ndata['train_mask'])

    n_class = n_class
    n_feat = n_feat
    n_train = g.ndata['train_mask'].int().sum().item()

    graph_dir = os.path.join(args.part_path, args.graph_name)
    part_config = os.path.join(graph_dir, args.graph_name + '.json')
    if not os.path.exists(part_config):
        with g.local_scope():
            if args.inductive:
                g.ndata.pop('val_mask')
                g.ndata.pop('test_mask')
            g.ndata['in_deg'] = g.in_degrees()
            g.ndata['out_deg'] = g.out_degrees()
            node_map, edge_map = partition_graph(
                g,
                args.graph_name,
                args.n_partitions,
                graph_dir,
                part_method=args.partition_method,
                balance_edges=False,
                objtype=args.partition_obj,
                return_mapping=True,
                num_trainers_per_machine=2)

    with open(os.path.join(graph_dir, 'meta.json'), 'w') as f:
        json.dump({
            'n_feat': n_feat,
            'n_class': n_class,
            'n_train': n_train
        }, f)

    return node_map, edge_map


def graph_partition_sub(g, n_feat, n_class, node_dict, args):

    if args.inductive:
        g = g.subgraph(node_dict['train_mask'])

    n_class = n_class
    n_feat = n_feat
    n_train = node_dict['train_mask'].int().sum().item()

    graph_dir = os.path.join(args.part_path, args.graph_name)
    part_config = os.path.join(graph_dir, args.graph_name + '.json')
    if not os.path.exists(part_config):
        with g.local_scope():
            if args.inductive:
                g.ndata.pop('val_mask')
                g.ndata.pop('test_mask')
            g.ndata['in_deg'] = g.in_degrees()
            g.ndata['out_deg'] = g.out_degrees()
            partition_graph(g,
                            args.graph_name,
                            args.n_partitions,
                            graph_dir,
                            part_method=args.partition_method,
                            balance_edges=False,
                            objtype=args.partition_obj)

    with open(os.path.join(graph_dir, 'meta.json'), 'w') as f:
        json.dump({
            'n_feat': n_feat,
            'n_class': n_class,
            'n_train': n_train
        }, f)


def graph_partition_sub_map(g, n_feat, n_class, node_dict, args):

    if args.inductive:
        g = g.subgraph(node_dict['train_mask'])

    n_class = n_class
    n_feat = n_feat
    n_train = node_dict['train_mask'].int().sum().item()

    graph_dir = os.path.join(args.part_path, args.graph_name)
    part_config = os.path.join(graph_dir, args.graph_name + '.json')
    if not os.path.exists(part_config):
        with g.local_scope():
            if args.inductive:
                g.ndata.pop('val_mask')
                g.ndata.pop('test_mask')
            g.ndata['in_deg'] = g.in_degrees()
            g.ndata['out_deg'] = g.out_degrees()
            node_map, edge_map = partition_graph(
                g,
                args.graph_name,
                args.n_partitions,
                graph_dir,
                part_method=args.partition_method,
                balance_edges=False,
                objtype=args.partition_obj,
                return_mapping=True)

    with open(os.path.join(graph_dir, 'meta.json'), 'w') as f:
        json.dump({
            'n_feat': n_feat,
            'n_class': n_class,
            'n_train': n_train
        }, f)

    return node_map, edge_map


def merge_partitions(part_path_bak, graph_name_bak, args):
    final_path = os.path.join(part_path_bak, graph_name_bak)
    os.makedirs(final_path, exist_ok=True)

    new_part_folder_counter = 0

    for i in range(len(args.topolist)):
        for j in range(args.topolist[i]):
            old_folder_path = os.path.join(args.part_path,
                                           graph_name_bak + '-%d' % i,
                                           'part' + '%d' % j)
            new_folder_path = os.path.join(
                final_path, 'part' + '%d' % new_part_folder_counter)

            os.rename(old_folder_path, new_folder_path)
            new_part_folder_counter += 1


def graph_partition_topo(args):
    g, n_feat, n_class = load_data(args)
    if args.inductive:
        g = g.subgraph(g.ndata['train_mask'])

    n_class = n_class
    n_feat = n_feat
    n_train = g.ndata['train_mask'].int().sum().item()

    graph_dir = os.path.join(args.part_path, args.graph_name)
    part_config = os.path.join(graph_dir, args.graph_name + '.json')
    if not os.path.exists(part_config):
        with g.local_scope():
            if args.inductive:
                g.ndata.pop('val_mask')
                g.ndata.pop('test_mask')
            g.ndata['in_deg'] = g.in_degrees()
            g.ndata['out_deg'] = g.out_degrees()
            print("args.topolist", args.topolist)
            partition_graph_topo(g,
                                 args.graph_name,
                                 args.n_partitions,
                                 graph_dir,
                                 part_method=args.partition_method,
                                 balance_edges=False,
                                 objtype=args.partition_obj,
                                 topo_list=args.topolist)

    with open(os.path.join(graph_dir, 'meta.json'), 'w') as f:
        json.dump({
            'n_feat': n_feat,
            'n_class': n_class,
            'n_train': n_train
        }, f)


def graph_partition_topo_weight(args):
    g, n_feat, n_class = load_data(args)
    if args.inductive:
        g = g.subgraph(g.ndata['train_mask'])

    n_class = n_class
    n_feat = n_feat
    n_train = g.ndata['train_mask'].int().sum().item()

    graph_dir = os.path.join(args.part_path, args.graph_name)
    part_config = os.path.join(graph_dir, args.graph_name + '.json')
    if not os.path.exists(part_config):
        with g.local_scope():
            if args.inductive:
                g.ndata.pop('val_mask')
                g.ndata.pop('test_mask')
            g.ndata['in_deg'] = g.in_degrees()
            g.ndata['out_deg'] = g.out_degrees()
            print("args.topolist", args.topolist)
            partition_graph_topo_weight(g,
                                        args.graph_name,
                                        args.n_partitions,
                                        graph_dir,
                                        part_method=args.partition_method,
                                        balance_edges=False,
                                        objtype=args.partition_obj,
                                        topo_list=args.topolist)

    with open(os.path.join(graph_dir, 'meta.json'), 'w') as f:
        json.dump({
            'n_feat': n_feat,
            'n_class': n_class,
            'n_train': n_train
        }, f)


def load_partition_graph(args, rank):

    graph_dir = os.path.join(args.part_path, args.graph_name)
    part_config = os.path.join(graph_dir, args.graph_name + '.json')

    print('loading partitions graph only', graph_dir)

    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(
        part_config, rank)

    print("node_feat", node_feat)

    node_type = node_type[0]
    node_feat[dgl.NID] = subg.ndata[dgl.NID]
    print("node_feat[dgl.NID]", node_feat)
    print("max", node_feat["_ID"].max())
    print("min", node_feat["_ID"].min())

    if 'part_id' in subg.ndata:
        node_feat['part_id'] = subg.ndata['part_id']
    node_feat['inner_node'] = subg.ndata['inner_node'].bool()
    node_feat['label'] = node_feat[node_type + '/label']
    node_feat['feat'] = node_feat[node_type + '/feat']
    node_feat['in_deg'] = node_feat[node_type + '/in_deg']
    node_feat['out_deg'] = node_feat[node_type + '/out_deg']
    node_feat['train_mask'] = node_feat[node_type + '/train_mask'].bool()
    node_feat.pop(node_type + '/label')
    node_feat.pop(node_type + '/feat')
    node_feat.pop(node_type + '/in_deg')
    node_feat.pop(node_type + '/out_deg')
    node_feat.pop(node_type + '/train_mask')
    if not args.inductive:
        node_feat['val_mask'] = node_feat[node_type + '/val_mask'].bool()
        node_feat['test_mask'] = node_feat[node_type + '/test_mask'].bool()
        node_feat.pop(node_type + '/val_mask')
        node_feat.pop(node_type + '/test_mask')
    if args.dataset == 'ogbn-papers100m':
        node_feat.pop(node_type + '/year')

    with open(os.path.join(graph_dir, 'meta.json'), 'r') as f:
        meta = json.load(f)
        args.n_feat = meta['n_feat']
        args.n_class = meta['n_class']
        args.n_train = meta['n_train']

    return subg, node_feat, gpb


def load_partition(args, rank):

    graph_dir = os.path.join(args.part_path, args.graph_name)
    part_config = os.path.join(graph_dir, args.graph_name + '.json')

    print('loading partitions')

    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(
        part_config, rank)
    isolated_nodes = ((subg.in_degrees() == 0) &
                      (subg.out_degrees() == 0)).nonzero().squeeze(1)
    subg.remove_nodes(isolated_nodes)

    node_type = node_type[0]
    node_feat[dgl.NID] = subg.ndata[dgl.NID]
    if 'part_id' in subg.ndata:
        node_feat['part_id'] = subg.ndata['part_id']
    node_feat['inner_node'] = subg.ndata['inner_node'].bool()
    node_feat['label'] = node_feat[node_type + '/label']
    node_feat['feat'] = node_feat[node_type + '/feat']
    node_feat['in_deg'] = node_feat[node_type + '/in_deg']
    node_feat['out_deg'] = node_feat[node_type + '/out_deg']
    node_feat['train_mask'] = node_feat[node_type + '/train_mask'].bool()
    node_feat.pop(node_type + '/label')
    node_feat.pop(node_type + '/feat')
    node_feat.pop(node_type + '/in_deg')
    node_feat.pop(node_type + '/out_deg')
    node_feat.pop(node_type + '/train_mask')
    if not args.inductive:
        node_feat['val_mask'] = node_feat[node_type + '/val_mask'].bool()
        node_feat['test_mask'] = node_feat[node_type + '/test_mask'].bool()
        node_feat.pop(node_type + '/val_mask')
        node_feat.pop(node_type + '/test_mask')
    if args.dataset == 'ogbn-papers100m':
        node_feat.pop(node_type + '/year')
    subg.ndata.clear()
    subg.edata.clear()

    with open(os.path.join(graph_dir, 'meta.json'), 'r') as f:
        meta = json.load(f)
        args.n_feat = meta['n_feat']
        args.n_class = meta['n_class']
        args.n_train = meta['n_train']

    return subg, node_feat, gpb


def get_layer_size(n_feat, n_hidden, n_class, n_layers):
    layer_size = [n_feat]
    layer_size.extend([n_hidden] * (n_layers - 1))
    layer_size.append(n_class)
    return layer_size


from collections import Counter


def get_boundary(node_dict, gpb, args):
    rank, size = dist.get_rank(), dist.get_world_size()
    device = 'cuda'
    boundary = [None] * size
    boundary_parts = [None] * size

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        belong_right = (node_dict['part_id'] == right)
        num_right = belong_right.sum().view(-1)
        if dist.get_backend() == 'gloo':
            num_right = num_right.cpu()
            num_left = torch.tensor([0])
        else:
            num_left = torch.tensor([0], device=device)
        req = dist.isend(num_right, dst=right)
        dist.recv(num_left, src=left)
        start = gpb.partid2nids(right)[0].item()
        v = node_dict[dgl.NID][belong_right] - start
        if dist.get_backend() == 'gloo':
            v = v.cpu()
            u = torch.zeros(num_left, dtype=torch.long)
        else:
            u = torch.zeros(num_left, dtype=torch.long, device=device)
        req.wait()
        req = dist.isend(v, dst=right)
        dist.recv(u, src=left)
        u, _ = torch.sort(u)
        if dist.get_backend() == 'gloo':
            boundary[left] = u.cuda()
        else:
            boundary[left] = u
        req.wait()
        boundary_parts[left] = node_dict['part_id'][belong_right]

    return boundary


def get_commute(node_dict, gpb):
    rank, size = dist.get_rank(), dist.get_world_size()
    device = 'cuda'
    boundary = [None] * size

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        belong_right = (node_dict['part_id'] == right)
        num_right = belong_right.sum().view(-1)
        if dist.get_backend() == 'gloo':
            num_right = num_right.cpu()
            num_left = torch.tensor([0])
        else:
            num_left = torch.tensor([0], device=device)
        req = dist.isend(num_right, dst=right)
        dist.recv(num_left, src=left)
        start = gpb.partid2nids(right)[0].item()
        v = node_dict[dgl.NID][belong_right] - start
        if dist.get_backend() == 'gloo':
            v = v.cpu()
            u = torch.zeros(num_left, dtype=torch.long)
        else:
            u = torch.zeros(num_left, dtype=torch.long, device=device)
        req.wait()
        req = dist.isend(v, dst=right)
        dist.recv(u, src=left)
        u, _ = torch.sort(u)
        if dist.get_backend() == 'gloo':
            boundary[left] = u.cuda()
        else:
            boundary[left] = u
        req.wait()

    return boundary


_send_cpu, _recv_cpu = {}, {}


def data_transfer(data, recv_shape, tag, dtype=torch.float):

    rank, size = dist.get_rank(), dist.get_world_size()
    msg, res = [None] * size, [None] * size

    for i in range(1, size):
        idx = (rank + i) % size
        key = 'dst%d_tag%d' % (idx, tag)
        if key not in _recv_cpu:
            _send_cpu[key] = torch.zeros_like(data[idx],
                                              dtype=dtype,
                                              device='cpu',
                                              pin_memory=True)
            _recv_cpu[key] = torch.zeros(recv_shape[idx],
                                         dtype=dtype,
                                         pin_memory=True)
        msg[idx] = _send_cpu[key]
        res[idx] = _recv_cpu[key]

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        msg[right].copy_(data[right])
        req = dist.isend(msg[right], dst=right, tag=tag)
        dist.recv(res[left], src=left, tag=tag)
        res[left] = res[left].cuda(non_blocking=True)
        req.wait()

    return res


def merge_feature(feat, recv):
    size = len(recv)
    for i in range(size - 1, 0, -1):
        if recv[i] is None:
            recv[i] = recv[i - 1]
            recv[i - 1] = None
    recv[0] = feat
    return torch.cat(recv)


def inductive_split(g):
    g_train = g.subgraph(g.ndata['train_mask'])
    g_val = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    g_test = g
    return g_train, g_val, g_test


def minus_one_tensor(size, device=None):
    if device is not None:
        return torch.zeros(size, dtype=torch.long, device=device) - 1
    else:
        return torch.zeros(size, dtype=torch.long) - 1


def nonzero_idx(x):
    return torch.nonzero(x, as_tuple=True)[0]


def print_memory(s):
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.cuda.synchronize()
    print('(rank %d) ' % rank + s +
          ': current {:.2f}MB, peak {:.2f}MB, reserved {:.2f}MB'.format(
              torch.cuda.memory_allocated() / 1024 / 1024,
              torch.cuda.max_memory_allocated() / 1024 / 1024,
              torch.cuda.memory_reserved() / 1024 / 1024))


@contextmanager
def timer(s):
    rank, size = dist.get_rank(), dist.get_world_size()
    t = time.time()
    yield
    print('(rank %d) running time of %s: %.3f seconds' %
          (rank, s, time.time() - t))
