# Market1501
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -d market1501 -b 256 --seed 3407 --num-instances 16 --iters 200 --lr-scheduler warmup --eval-step 2 --epochs 70 --use-cm --cm-type wgm --tau-w 0.03 --momentum 0.1 --eps-scheduler expo --eps 0.7 --smooth 0.3 --soft-ce-weight 1.0 --logs-dir logs/dccc/market_resnet50

# DukeMTMC
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -d dukemtmc -b 256 --num-instances 16 --iters 200 --lr-scheduler warmup --eval-step 2 --epochs 70 --use-cm --cm-type wgm --tau-w 0.07 --momentum 0.1 --eps-scheduler expo --eps 0.7 --soft-ce-weight 1.0 --logs-dir logs/dccc/duke_resnet50

# MSMT17
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -d msmt17 -b 256 --num-instances 16 --iters 200 --lr-scheduler warmup --eval-step 2 --epochs 80 --use-cm --cm-type wgm --tau-w 0.6 --momentum 0.1 --eps-scheduler expo --eps 0.8 --soft-ce-weight 1.0 --logs-dir logs/dccc/msmt17_resnet50
