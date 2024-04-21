#!/bin/bash

ips=("node1" "node2" "node3" "node4")
datasets=("ogbn-arxiv" "ogbn-products" "yelp" "reddit" "ogbn-mag" "ogbn-proteins" "am")

declare -a pids

log_file="../logs/rcgnn_node1_other3.log"

models=("gcn" "graphsage" "gat")
n_partitions=(32 16)
n_layers=(4)
n_hiddens=(128)

for layer in "${n_layers[@]}"; do
    for hidden in "${n_hiddens[@]}"; do
        for partition in "${n_partitions[@]}"; do
            topolist=$((partition / 4))
            for dataset in "${datasets[@]}"; do
                for model in "${models[@]}"; do

                    search_content="hemeng $dataset $model $partition $layer $hidden 0 $topolist"
                    grep -q "$search_content" $log_file
                    if [ $? -eq 0 ]; then
                        continue
                    fi

                    echo rcgnn $dataset $model $partition $layer $hidden $node_rank $topolist

                    for index in "${!ips[@]}"; do
                        ip=${ips[$index]}
                        node_rank=$index
                        if [ $node_rank -eq 0 ]; then
                            lsof -i :18119 | awk 'NR!=1 {print $2}' | xargs kill -9 > /dev/null 2>&1
                        fi
                        
                        nohup ssh $ip <<-EOF >> ../logs/rcgnn_${ip}_other3.log 2>&1 &
                            cd /root/SC24/pytorch/mytest/comp/RCGNN
                            echo hemeng $dataset $model $partition $layer $hidden $node_rank $topolist

                            GLOO_SOCKET_IFNAME=enp97s0f0 /root/miniconda3/envs/dgl-dev-gpu-121/bin/python main.py \
                            --dataset $dataset \
                            --dropout 0.3 \
                            --lr 0.003 \
                            --n-partitions $partition \
                            --n-epochs 30 \
                            --model $model \
                            --sampling-rate 1 \
                            --n-layers $layer \
                            --n-hidden $hidden \
                            --log-every 10 \
                            --use-pp \
                            --fix-seed \
                            --master-addr 192.168.2.1 \
                            --node-rank $node_rank \
                            --topolist $topolist $topolist $topolist $topolist \
                            --port 18119 \
                            --partition_obj cut
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


# SAGE-LSTM

log_file="../logs/rcgnn_node1_lstm.log"

models=("graphsage_lstm")
n_partitions=(32 16)
n_layers=(2)
n_hiddens=(32)

for layer in "${n_layers[@]}"; do
    for hidden in "${n_hiddens[@]}"; do
        for partition in "${n_partitions[@]}"; do
            topolist=$((partition / 4))
            for dataset in "${datasets[@]}"; do
                for model in "${models[@]}"; do
                
                    search_content="hemeng $dataset $model $partition $layer $hidden 0 $topolist"
                    grep -q "$search_content" $log_file
                    if [ $? -eq 0 ]; then
                        continue
                    fi

                    for index in "${!ips[@]}"; do
                        ip=${ips[$index]}
                        node_rank=$index
                        if [ $node_rank -eq 0 ]; then
                            lsof -i :18119 | awk 'NR!=1 {print $2}' | xargs kill -9 > /dev/null 2>&1
                        fi

                        echo rcgnn $dataset $model $partition $layer $hidden $node_rank $topolist
                        
                        nohup ssh $ip <<-EOF >> ../logs/rcgnn_${ip}_lstm.log 2>&1 &
                            cd /root/SC24/pytorch/mytest/comp/RCGNN
                            echo hemeng $dataset $model $partition $layer $hidden $node_rank $topolist

                            GLOO_SOCKET_IFNAME=enp97s0f0 /root/miniconda3/envs/dgl-dev-gpu-121/bin/python main.py \
                            --dataset $dataset \
                            --dropout 0.3 \
                            --lr 0.003 \
                            --n-partitions $partition \
                            --n-epochs 30 \
                            --model $model \
                            --sampling-rate 1 \
                            --n-layers $layer \
                            --n-hidden $hidden \
                            --log-every 10 \
                            --use-pp \
                            --fix-seed \
                            --master-addr 192.168.2.1 \
                            --node-rank $node_rank \
                            --topolist $topolist $topolist $topolist $topolist \
                            --port 18119 \
                            --partition_obj cut
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