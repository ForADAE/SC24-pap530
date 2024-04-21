from helper.parser import *
import random
import torch.multiprocessing as mp
import sys
import subprocess
from helper.utils import *
import train
import warnings

if __name__ == '__main__':

    args = create_parser()
    if args.fix_seed is False:
        if args.parts_per_node < args.n_partitions:
            warnings.warn(
                'Please enable `--fix-seed` for multi-node training.')
        args.seed = random.randint(0, 1 << 31)

    if args.graph_name == '':
        if args.inductive:
            args.graph_name = '%s-%d-%s-%s-induc' % (
                args.dataset, args.n_partitions, args.partition_method,
                args.partition_obj)
        else:
            args.graph_name = '%s-%d-%s-%s-trans' % (
                args.dataset, args.n_partitions, args.partition_method,
                args.partition_obj)

    if args.topolist:
        args.graph_name = args.graph_name + '-' + '-'.join(
            [str(i) for i in args.topolist])

    if args.weight:
        args.graph_name = args.graph_name + '_weight'

    if args.dataset[:4] == 'rand':
        args.part_path = './partition/rand/'

    if args.sub_rate != None:
        args.graph_name = args.graph_name + '_' + "%.2f" % args.sub_rate

    if not args.skip_partition:
        if args.node_rank == 0:
            if args.topolist:
                if args.inductive:
                    graph_partition_topo(args)
                else:
                    if args.weight:
                        print('graph_partition_topo_weight'
                              )
                        graph_partition_topo_weight(args)
                    else:
                        print('graph_partition_topo')
                        graph_partition_topo(args)
            else:
                if args.inductive:
                    graph_partition(args)
                else:
                    graph_partition(args)

    print(args)

    if args.backend == 'gloo':
        processes = []
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        else:
            n = torch.cuda.device_count()
            devices = [f'{i}' for i in range(n)]
        mp.set_start_method('spawn', force=True)
        if args.topolist:
            start_id = sum(args.topolist[:args.node_rank])
            for i in range(start_id, start_id + args.topolist[args.node_rank]):
                os.environ['CUDA_VISIBLE_DEVICES'] = devices[i % len(devices)]
                p = mp.Process(target=train.init_processes,
                               args=(i, args.n_partitions, args))
                p.start()
                print(f'process {i} started')
                processes.append(p)
            for p in processes:
                p.join()
        else:
            start_id = args.node_rank * args.parts_per_node
            for i in range(
                    start_id,
                    min(start_id + args.parts_per_node, args.n_partitions)):
                os.environ['CUDA_VISIBLE_DEVICES'] = devices[i % len(devices)]
                p = mp.Process(target=train.init_processes,
                               args=(i, args.n_partitions, args))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
    elif args.backend == 'mpi':
        gcn_arg = []
        for k, v in vars(args).items():
            if v is True:
                gcn_arg.append(f'--{k}')
            elif v is not False:
                gcn_arg.extend([f'--{k}', f'{v}'])
        mpi_arg = []
        mpi_arg.extend(['-n', f'{args.n_partitions}'])
        command = ['mpirun'] + mpi_arg + ['python', 'train.py'] + gcn_arg
        print(' '.join(command))
        subprocess.run(command, stderr=sys.stderr, stdout=sys.stdout)
    else:
        raise ValueError
