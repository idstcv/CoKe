log='coke_multi_view';
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_view.py \
    -a resnet50 \
    --lr 1.6 \
    --batch-size 1024 \
    --epochs 801 \
    --coke-t 0.05 \
    --coke-k 3000 4000 5000 \
    --coke-dual-lr 20 \
    --coke-ratio 0.4 \
    --coke-alpha 0.2 \
    --coke-beta 0.5 \
    --coke-snum 6 \
    --log ${log} \
    --dist-url 'tcp://localhost:'${RANDOM} --multiprocessing-distributed --world-size 1 --rank 0 \
    /path/to/imagenet/ | tee log/${log}.log;