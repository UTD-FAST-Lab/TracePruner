#!/bin/bash

# cgpruner baseline
# python approach/runners/main.py --dataset xcorp --baseline cgpruner --exp 1
# python approach/runners/main.py --dataset xcorp --baseline cgpruner --exp 2
# python approach/runners/main.py --dataset xcorp --baseline cgpruner --exp 3

# nn baseline
python approach/runners/main.py --dataset xcorp --baseline autopruner --exp 1
python approach/runners/main.py --dataset xcorp --baseline autopruner --exp 2
python approach/runners/main.py --dataset xcorp --baseline autopruner --exp 3

# # svm baseline
# python approach/runners/main.py --dataset xcorp --baseline svm --exp 1
# python approach/runners/main.py --dataset xcorp --baseline svm --exp 2
# python approach/runners/main.py --dataset xcorp --baseline svm --exp 3

# # clustering
# python approach/clustering/main.py  --exp 1
# python approach/clustering/main.py  --exp 2
# python approach/clustering/main.py  --exp 3