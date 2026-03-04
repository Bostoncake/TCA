export CUDA_VISIBLE_DEVICES=$1

currenttime=`date +"%Y-%m-%d_%H"`
GPU=$1
if [ "$GPU" -eq 0 ]; then
    backbone="large"
    ckpt_dir="/mnt/data1/xiongyizhe/models/vit_large_patch16_224.augreg_in21k_ft_in1k.bin"
    batch_size=32
    tome_r=4
    load_model_dir="/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-16_12_large_ToMe_r_4_heatmap_model/rendition-large-model.pth"
    heatmap_layer="0,11,22"
elif [ "$GPU" -eq 1 ]; then
    backbone="base"
    ckpt_dir="/mnt/data1/xiongyizhe/models/vit_base_patch16_224.augreg2_in21k_ft_in1k.bin"
    batch_size=64
    tome_r=8
    load_model_dir="/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-16_12_base_ToMe_r_8_heatmap_model/rendition-base-model.pth"
    heatmap_layer="0,5,10"
fi
corruption='rendition'
algorithm="foa_bp"
seed=2020


# tag=$currenttime"_"$backbone"_ToMe_r_"$tome_r"_heatmap_model"
    

# mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/

# echo $corruption
# python3 main.py \
#     --batch_size $batch_size \
#     --workers 8 \
#     --backbone $backbone \
#     --ckpt_dir $ckpt_dir \
#     --data /mnt/data1/xiongyizhe/dataset/imagenet \
#     --data_v2 /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-v2 \
#     --data_sketch /mnt/data1/xiongyizhe/dataset/imnet-ood/sketch \
#     --data_corruption /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-c \
#     --data_rendition /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-r \
#     --corruption $corruption \
#     --output ./outputs \
#     --algorithm $algorithm \
#     --apply_tome --tome_r $tome_r --as_baseline --save_model \
#     --tag $tag --seed $seed > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt

tag=$currenttime"_"$backbone"_ToMe_r_"$tome_r"_heatmap_draw"
    

mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/

test_img_dir="/mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-r/n01847000"
echo $corruption
python3 heatmap.py \
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
    --apply_tome --tome_r $tome_r --as_baseline \
    --tag $tag --seed $seed --load_model_dir $load_model_dir --test_image_dir "$test_img_dir" --heatmap_layer $heatmap_layer # > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt
