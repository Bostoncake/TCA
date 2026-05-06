export CUDA_VISIBLE_DEVICES=$1

GPU=$1

if [ "$GPU" -eq 0 ]; then
    corruption='rendition'
    tca_r_large=4
    tca_r_base=6
elif [ "$GPU" -eq 1 ]; then
    corruption='v2'
    tca_r_large=4
    tca_r_base=6
elif [ "$GPU" -eq 2 ]; then
    corruption='sketch'
    tca_r_large=4
    tca_r_base=6
elif [ "$GPU" -eq 3 ]; then
    corruption='original'
    tca_r_large=4
    tca_r_base=6
elif [ "$GPU" -eq 4 ]; then
    corruption='rendition'
    tca_r_large=8
    tca_r_base=12
elif [ "$GPU" -eq 5 ]; then
    corruption='v2'
    tca_r_large=8
    tca_r_base=12
elif [ "$GPU" -eq 6 ]; then
    corruption='sketch'
    tca_r_large=8
    tca_r_base=12
elif [ "$GPU" -eq 7 ]; then
    corruption='original'
    tca_r_large=8
    tca_r_base=12
else
    echo "Invalid GPU number: $GPU. Must be 0-7."
    exit 1
fi

algorithm="foa_bp"
currenttime=`date +"%Y-%m-%d_%H"`
seed=2021

backbone="large"
ckpt_dir="/mnt/data1/xiongyizhe/models/vit_large_patch16_224.augreg_in21k_ft_in1k.bin"
batch_size=32


tag=$currenttime"_large_TCA_r_"$tca_r_large"_seed_"$seed

mkdir -p /home/xiongyizhe/CVPR2026/TCA/MGTTA/outputs/$algorithm"_"$tag/

echo $corruption
python3 main.py \
    --batch_size $batch_size \
    --workers 8 \
    --backbone $backbone \
    --ckpt_dir $ckpt_dir \
    --data /mnt/data1/xiongyizhe/dataset/imagenet \
    --data_v2 /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-v2 \
    --data_sketch /mnt/data1/xiongyizhe/dataset/imnet-ood/sketch \
    --data_corruption /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-c \
    --data_rendition /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-r \
    --corruption $corruption \
    --output ./outputs \
    --algorithm $algorithm \
    --apply_tca --tca_r $tca_r_large \
    --tag $tag --seed $seed > /home/xiongyizhe/CVPR2026/TCA/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt


backbone="base"        # base use tca_r=6,12 GFLOPs:14.81,12.68
ckpt_dir="/mnt/data1/xiongyizhe/models/vit_base_patch16_224.augreg2_in21k_ft_in1k.bin"
batch_size=64


tag=$currenttime"_base_TCA_r_"$tca_r_base"_seed_"$seed

mkdir -p /home/xiongyizhe/CVPR2026/TCA/MGTTA/outputs/$algorithm"_"$tag/

    echo $corruption
    python3 main.py \
        --batch_size $batch_size \
        --workers 8 \
        --backbone $backbone \
        --ckpt_dir $ckpt_dir \
        --data /mnt/data1/xiongyizhe/dataset/imagenet \
        --data_v2 /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-v2 \
        --data_sketch /mnt/data1/xiongyizhe/dataset/imnet-ood/sketch \
        --data_corruption /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-c \
        --data_rendition /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-r \
        --corruption $corruption \
        --output ./outputs \
        --algorithm $algorithm \
        --apply_tca --tca_r $tca_r_base \
        --tag $tag --seed $seed > /home/xiongyizhe/CVPR2026/TCA/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt
