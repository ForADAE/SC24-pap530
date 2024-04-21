#!/bin/bash

ips=("node1" "node2")
datasets=("ogbn-products" "yelp" "ogbn-proteins" "reddit")
models=("gat")
n_partitions=(16)
n_layers=(4)
n_hiddens=(128)
swap_bits=(4294967295 2147483647 1073741823 536870911 268435455 134217727 67108863 33554431 16777215 8388607 4194303 2097151 1048575 524287 262143 131071 65535 32767 16383 8191 4095 2047 1023 511 255 127 63 31 15 7 3 1 0)
declare -a pids

log_file="rcgnn_gat_swap_node1.log"

for layer in "${n_layers[@]}"; do
    for hidden in "${n_hiddens[@]}"; do
        for partition in "${n_partitions[@]}"; do
            topolist=$((partition / 2))
            for swap_bit in "${swap_bits[@]}"; do
                for dataset in "${datasets[@]}"; do
                    for model in "${models[@]}"; do

                        search_content="hemeng $dataset $model $partition $layer $hidden 0 $topolist $swap_bit"
                        grep -q "$search_content" $log_file
                        if [ $? -eq 0 ]; then
                            continue
                        fi

                        echo hemeng $dataset $model $partition $layer $hidden $node_rank $topolist $swap_bit

                        for index in "${!ips[@]}"; do
                            ip=${ips[$index]}
                            node_rank=$index
                            if [ $node_rank -eq 0 ]; then
                                lsof -i :18118 | awk 'NR!=1 {print $2}' | xargs kill -9 > /dev/null 2>&1
                            fi

                            nohup ssh $ip <<-EOF >> rcgnn_gat_swap_${ip}.log 2>&1 &
                                cd /root/SC24/pytorch/mytest/comp/RCGNN
                                echo hemeng $dataset $model $partition $layer $hidden $node_rank $topolist $swap_bit

                                GLOO_SOCKET_IFNAME=enp97s0f0 /root/miniconda3/envs/dgl-dev-gpu-121/bin/python main.py \
                                --dataset $dataset \
                                --dropout 0.3 \
                                --lr 0.003 \
                                --n-partitions $partition \
                                --n-epochs 3 \
                                --model $model \
                                --sampling-rate 1 \
                                --n-layers $layer \
                                --n-hidden $hidden \
                                --log-every 10 \
                                --use-pp \
                                --fix-seed \
                                --master-addr 192.168.2.1 \
                                --node-rank $node_rank \
                                --topolist $topolist $topolist \
                                --swap-bits $swap_bit \
                                --port 18118
EOF

                            pid=$!
                            pids+=("$pid")
                        done

                        for pid in "${pids[@]}"; do
                            wait "$pid"
                        done
                        unset pids
                    done
                done
            done
        done
    done
done
