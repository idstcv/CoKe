log='coke_single_view';
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_single_view.py \
    -a resnet50 \
    --lr 1.6 \
    --batch-size 1024 \
    --epochs 1001 \
    --coke-t 0.1 \
    --coke-k 3000 4000 5000 \
    --coke-stage 801 \
    --coke-tt 0.5 \
    --coke-dual-lr 20 \
    --coke-ratio 0.4 \
    --coke-ls 5 \
    --coke-beta 1 \
    --log ${log} \
    --dist-url 'tcp://localhost:'${RANDOM} --multiprocessing-distributed --world-size 1 --rank 0 \
    /path/to/imagenet/ | tee log/${log}.log;