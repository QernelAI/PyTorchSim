#!/bin/bash
python3 ../tests/test_hetro.py --M 1024 --N 1024 --K 1024 --sparsity 0.9 --config stonne_big_c1_simple_noc.json --mode 0 > hetero/big_sparse.log
python3 ../tests/test_hetro.py --M 1024 --N 1024 --K 1024 --sparsity 0.9 --config systolic_ws_128x128_c1_simple_noc_tpuv2_half.json --mode 1 > hetero/big.log
python3 ../tests/test_hetro.py --M 1024 --N 1024 --K 1024 --sparsity 0.9 --config heterogeneous_c2_simple_noc.json --mode 2 > hetero/hetero.log

echo "All processes completed!"
