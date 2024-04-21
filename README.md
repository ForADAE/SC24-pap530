# RCGNN

This is the code repository of SC '24 submission "RCGNN: Efficient Distributed Full-Graph Training on Resource-Constrained Platforms".

## Artifact Setup

### Hardware

We evaluate RCGNN on a 4-node cluster with 8 Intel Xeon Silver 4210 CPUs and 32 NVIDIA GeForce RTX 2060 SUPER GPUs, connected by 10Gbps Ethernet.

### Software

The experiments are conducted on Ubuntu 22.04 with GCC v11.4 and NVCC v12.1. The RCGNN is built on DGL v2.1 and PyTorch v2.2.

### Installation and Deployment

1. Install the dependencies:
```pip install -r requirements.txt```

2. Clone and install the modified PyTorch and DGL from:

https://github.com/ForADAE/pytorch

https://github.com/ForADAE/dgl

3. Clone and install the pytm from:

https://github.com/ForADAE/pytm

4. Run the script in the `scripts` directory to reproduce the experiments.

5. Draw the figures using the notebooks in the `plots` directory.

### Reproducing the Experiments

1. For overall performance evaluation, reproduce the 2 nodes experiment by:

```bash scripts/overall-rcgnn-2.sh```

and the 4 nodes experiment by:

```bash scripts/overall-rcgnn-4.sh```

2. For the communication volumn evaluation, reproduce the results by:

```bash scripts/patition.sh```

3. For the memory footprint estimation, reproduce the results by:

```bash scripts/memory.sh```

4. For the tensor swapping granularity test, reproduce the results of GCN by:

```bash scripts/swap-gcn.sh```

and GAT by:

```bash scripts/swap-gat.sh```