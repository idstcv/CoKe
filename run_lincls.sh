lr=$1;
model=$2;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python eval_lincls.py \
    -a resnet50 \
    --lr ${lr} \
    --pretrained ${model} \
    --dist-url 'tcp://localhost:'${RANDOM} --multiprocessing-distributed --world-size 1 --rank 0 \
    /path/to/imagenet/