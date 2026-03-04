export CUDA_VISIBLE_DEVICES=$1

corruption='rendition'

backbone="base"
ckpt_dir="/mnt/data1/xiongyizhe/models/vit_base_patch16_224.augreg2_in21k_ft_in1k.bin"
batch_size=32

algorithm="foa_bp"
currenttime=`date +"%Y-%m-%d_%H"`
for num_adapt_iters in 75; do
    tag=$currenttime"_bs64_tsne_iter"$num_adapt_iters

    mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/

    echo $corruption
    python3 main_tsne.py \
        --batch_size $batch_size \
        --workers 8 \
        --backbone $backbone \
        --ckpt_dir $ckpt_dir \
        --data /mnt/data1/xiongyizhe/dataset/imagenet \
        --data_v2 /mnt/data1/xiongyizhe/dataset/imnet-ood \
        --data_sketch /mnt/data1/xiongyizhe/dataset/imnet-ood/sketch \
        --data_corruption /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-c \
        --data_rendition /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-r \
        --corruption $corruption \
        --output ./outputs \
        --algorithm $algorithm \
        --num_adapt_iters $num_adapt_iters --start_validation 312 \
        --tag $tag > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt
done