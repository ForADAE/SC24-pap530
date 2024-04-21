import random

datasets = ['ogbn-arxiv', 'ogbn-products', 'yelp', 'reddit']

sub_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

partitions = [4, 6, 7, 8]


def generate_swap_bits():
    # swap_bits = random.sample(range(1, 15), 4)
    swap_bits = random.sample(range(1, 16382), 20)
    swap_bits = [0] + swap_bits + [16383]
    return swap_bits


def generate_partition():
    return random.randint(8, 8)


base_command = "python main.py --dataset {} --dropout 0.3 --lr 0.003 --n-partitions {} --n-epochs 3 --model graphsage_lstm --sampling-rate 1 --n-layers 2 --n-hidden 32 --log-every 1 --use-pp --sub-rate {} --swap-bits 0"

for dataset in datasets:
    swap_bits = generate_swap_bits()
    # partition = generate_partition()

    for partition in partitions:
        for sub_ratio in sub_ratios:
            # for swap_bit in swap_bits:
                command = base_command.format(dataset, partition, sub_ratio)
                print(command)
