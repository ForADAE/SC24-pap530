import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='RCGNN')
    parser.add_argument("--dataset",
                        type=str,
                        default='reddit',
                        help="the input dataset")
    parser.add_argument("--data-path",
                        "--data_path",
                        type=str,
                        default='./dataset/',
                        help="the storage path of datasets")
    parser.add_argument("--part-path",
                        "--part_path",
                        type=str,
                        default='./partition/',
                        help="the storage path of graph partitions")
    parser.add_argument("--graph-name", "--graph_name", type=str, default='')
    parser.add_argument("--model",
                        type=str,
                        default='graphsage',
                        help="model for training")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="dropout probability")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--sampling-rate",
                        "--sampling_rate",
                        type=float,
                        default=1,
                        help="the sampling rate of boundary nodes")
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--n-epochs",
                        "--n_epochs",
                        type=int,
                        default=200,
                        help="the number of training epochs")
    parser.add_argument("--n-partitions",
                        "--n_partitions",
                        type=int,
                        default=2,
                        help="the number of partitions")
    parser.add_argument("--n-hidden",
                        "--n_hidden",
                        type=int,
                        default=16,
                        help="the number of hidden units")
    parser.add_argument("--n-layers",
                        "--n_layers",
                        type=int,
                        default=2,
                        help="the number of GCN layers")
    parser.add_argument("--log-every", "--log_every", type=int, default=10)
    parser.add_argument("--weight-decay",
                        "--weight_decay",
                        type=float,
                        default=0,
                        help="weight for L2 loss")
    parser.add_argument("--norm",
                        choices=['layer', 'batch'],
                        default='layer',
                        help="normalization method")
    parser.add_argument("--partition-obj",
                        "--partition_obj",
                        choices=['vol', 'cut'],
                        default='vol',
                        help="partition objective function ('vol' or 'cut')")
    parser.add_argument(
        "--partition-method",
        "--partition_method",
        choices=['metis', 'random'],
        default='metis',
        help="the method for graph partition ('metis' or 'random')")
    parser.add_argument("--n-linear",
                        "--n_linear",
                        type=int,
                        default=0,
                        help="the number of linear layers")
    parser.add_argument("--use-pp",
                        "--use_pp",
                        action='store_true',
                        help="whether to use precomputation")
    parser.add_argument("--inductive",
                        action='store_true',
                        help="inductive learning setting")
    parser.add_argument("--fix-seed",
                        "--fix_seed",
                        action='store_true',
                        help="fix random seed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", type=str, default='gloo')
    parser.add_argument("--port",
                        type=int,
                        default=18118,
                        help="the network port for communication")
    parser.add_argument("--master-addr",
                        "--master_addr",
                        type=str,
                        default="127.0.0.1")
    parser.add_argument("--node-rank", "--node_rank", type=int, default=0)
    parser.add_argument("--parts-per-node",
                        "--parts_per_node",
                        type=int,
                        default=10)
    parser.add_argument('--skip-partition',
                        action='store_true',
                        help="skip graph partition")
    parser.add_argument('--eval',
                        action='store_true',
                        help="enable evaluation")
    parser.add_argument('--no-eval',
                        action='store_false',
                        dest='eval',
                        help="disable evaluation")
    parser.set_defaults(eval=True)

    parser.add_argument('--weight',
                        action='store_true',
                        help="repartition based on weight")
    parser.set_defaults(weight=False)
    parser.add_argument(
        '--topolist',
        nargs='+',
        type=int,
        help='partitions of each node',
    )

    parser.add_argument('--save-model', action='store_true', help="save model")
    parser.add_argument('--no-save-model',
                        action='store_false',
                        dest='save_model',
                        help="do not save model")
    parser.set_defaults(save_model=False)
    parser.add_argument("--n-nodes",
                        "--n_nodes",
                        type=int,
                        default=1000,
                        help="the number of nodes in the random graph")
    parser.add_argument("--n-edges",
                        "--n_edges",
                        type=int,
                        default=10000,
                        help="the number of edges in the random graph")
    parser.add_argument("--n-feat",
                        "--n_feat",
                        type=int,
                        default=100,
                        help="the number of features in the random graph")
    parser.add_argument("--n-class",
                        "--n_class",
                        type=int,
                        default=10,
                        help="the number of classes in the random graph")
    parser.add_argument("--train-ratio",
                        "--train_ratio",
                        type=float,
                        default=0.9,
                        help="the ratio of training nodes")
    parser.add_argument("--val-ratio",
                        "--val_ratio",
                        type=float,
                        default=0.05,
                        help="the ratio of validation nodes")
    parser.add_argument(
        "--swap-bits",
        "--swap_bits",
        type=int,
        default=0,
        help="the number representing which tensor to swap in bit")
    parser.add_argument(
        "--swap-rate",
        "--swap_rate",
        type=float,
        default=0,
        help="the number representing what ratio of tensor to swap")
    parser.add_argument("--sub-rate",
                        "--sub_rate",
                        type=float,
                        default=None,
                        help="the number representing the ratio of subgraph")

    parser.add_argument('--uvm',
                        action='store_true',
                        help="whether to use UVM in cuda")

    return parser.parse_args()
