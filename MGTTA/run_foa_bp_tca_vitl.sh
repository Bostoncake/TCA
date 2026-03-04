#!/bin/bash
# Run TCA (Token Condensation as Adaptation, ICCV 2025) as a token pruning
# baseline on ViT-L/16, following the same structure as
# run_foa_bp_token_baselines_vitl.sh (EViT / Tofu).
#
# Usage:
#   bash run_foa_bp_tca_vitl.sh <GPU_ID>
#
# Each GPU handles two corruption types.  tca_r controls how many tier-3
# tokens are removed per pruning layer (layers 3, 6, 9).  The final token
# count per pruning layer is:
#   kept = (n_patch - 3*tca_r) + num_coreset_centers + 1
# where n_patch=196 for ViT-L/16 at 224px.
#
# tca_r=2 : removes 2 tier-3 tokens → keeps 195 tokens/layer
# tca_r=4 : removes 4 tier-3 tokens → keeps 189 tokens/layer

export CUDA_VISIBLE_DEVICES=$1

GPU=$1
if [ "$GPU" -eq 0 ]; then
    corruptions=('gaussian_noise' 'shot_noise')
elif [ "$GPU" -eq 1 ]; then
    corruptions=('impulse_noise' 'defocus_blur')
elif [ "$GPU" -eq 2 ]; then
    corruptions=('glass_blur' 'motion_blur')
elif [ "$GPU" -eq 3 ]; then
    corruptions=('zoom_blur' 'snow')
elif [ "$GPU" -eq 4 ]; then
    corruptions=('frost' 'fog')
elif [ "$GPU" -eq 5 ]; then
    corruptions=('brightness' 'jpeg_compression')
elif [ "$GPU" -eq 6 ]; then
    corruptions=('elastic_transform' 'pixelate')
elif [ "$GPU" -eq 7 ]; then
    corruptions=('contrast')
else
    echo "Invalid GPU number: $GPU. Must be 0-7."
    exit 1
fi

algorithm="foa_bp"
currenttime=`date +"%Y-%m-%d_%H"`
backbone="large"
ckpt_dir="/mnt/data1/xiongyizhe/models/vit_large_patch16_224.augreg_in21k_ft_in1k.bin"
batch_size=32

for seed in 2021 2022 2023; do

    for tca_r in 2 4; do
        tag=$currenttime"_large_TCA_r_"$tca_r"_seed_"$seed

        mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/

        for corruption in ${corruptions[@]}; do
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
                --apply_tca --tca_r $tca_r \
                --tag $tag --seed $seed > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
        done
    done

done
wait
