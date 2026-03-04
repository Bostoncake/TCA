export CUDA_VISIBLE_DEVICES=$1

currenttime=`date +"%Y-%m-%d_%H"`
backbone="base"
ckpt_dir="/mnt/data1/xiongyizhe/models/vit_base_patch16_224.augreg2_in21k_ft_in1k.bin"

corruption='gaussian_noise'

lr_cls_token=0.08
lr_cls_ssf=0.005
cls_ssf_layer="0,1,2,3,4,5"

algorithm="foa_bp"
# tome_r=8
# tag=$currenttime"_bs64_NAVIA_"$tome_r"_efficiency"
# mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/
# echo $corruption
# python3 main_navia_efficiency.py \
#     --batch_size 64 \
#     --workers 8 \
#     --backbone $backbone \
#     --ckpt_dir $ckpt_dir \
#     --data /mnt/data1/xiongyizhe/dataset/imagenet \
#     --data_v2 /mnt/data1/xiongyizhe/dataset/imnet-ood \
#     --data_sketch /mnt/data1/xiongyizhe/dataset/imnet-ood/sketch \
#     --data_corruption /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-c \
#     --data_rendition /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-r \
#     --corruption $corruption \
#     --output ./outputs \
#     --algorithm $algorithm \
#     --apply_tome --tome_r $tome_r --apply_sep_lr_for_cls --lr_cls_token $lr_cls_token --apply_cls_ssf --cls_ssf_layer $cls_ssf_layer --apply_sep_lr_cls_ssf --lr_cls_ssf $lr_cls_ssf \
#     --tag $tag > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt

tome_r=8
tag=$currenttime"_bs64_ToMe_"$tome_r"_efficiency_no_bp"
mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/
echo $corruption
python3 main_navia_efficiency.py \
    --batch_size 64 \
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
    --apply_tome --tome_r $tome_r --as_baseline \
    --tag $tag > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt

# prune_token_by_layer=8
# tag=$currenttime"_bs64_EViT_prune_token_by_layer_"$prune_token_by_layer"_efficiency"
# mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/
# echo $corruption
# python3 main.py \
#     --batch_size 64 \
#     --workers 8 \
#     --data /nlp_group/xiongyizhe/datasets/ImageNet \
#     --data_v2 /nlp_group/xiongyizhe/datasets/ImageNet-C \
#     --data_sketch /nlp_group/xiongyizhe/datasets/ImageNet-C/sketch \
#     --data_corruption /nlp_group/xiongyizhe/datasets/ImageNet-C/ImageNet-C \
#     --data_rendition /nlp_group/xiongyizhe/datasets/ImageNet-C/imagenet-r \
#     --corruption $corruption \
#     --output ./outputs \
#     --algorithm $algorithm \
#     --apply_evit --prune_token_by_layer $prune_token_by_layer \
#     --tag $tag --seed $seed > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt

# tofu_r=8
# tag=$currenttime"_bs64_Tofu_"$tofu_r"_sep_6_efficiency"
# mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/
# echo $corruption
# python3 main.py \
#     --batch_size 64 \
#     --workers 8 \
#     --data /nlp_group/xiongyizhe/datasets/ImageNet \
#     --data_v2 /nlp_group/xiongyizhe/datasets/ImageNet-C \
#     --data_sketch /nlp_group/xiongyizhe/datasets/ImageNet-C/sketch \
#     --data_corruption /nlp_group/xiongyizhe/datasets/ImageNet-C/ImageNet-C \
#     --data_rendition /nlp_group/xiongyizhe/datasets/ImageNet-C/imagenet-r \
#     --corruption $corruption \
#     --output ./outputs \
#     --algorithm $algorithm \
#     --apply_tofu --tofu_r $tofu_r --tofu_sep 6 \
#     --tag $tag --seed $seed > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt

# tome_r=4
# tag=$currenttime"_bs64_NAVIA_"$tome_r"_efficiency"
# mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/
# echo $corruption
# python3 main_navia_efficiency.py \
#     --batch_size 64 \
#     --workers 8 \
#     --backbone $backbone \
#     --ckpt_dir $ckpt_dir \
#     --data /mnt/data1/xiongyizhe/dataset/imagenet \
#     --data_v2 /mnt/data1/xiongyizhe/dataset/imnet-ood \
#     --data_sketch /mnt/data1/xiongyizhe/dataset/imnet-ood/sketch \
#     --data_corruption /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-c \
#     --data_rendition /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-r \
#     --corruption $corruption \
#     --output ./outputs \
#     --algorithm $algorithm \
#     --apply_tome --tome_r $tome_r --apply_sep_lr_for_cls --lr_cls_token $lr_cls_token --apply_cls_ssf --cls_ssf_layer $cls_ssf_layer --apply_sep_lr_cls_ssf --lr_cls_ssf $lr_cls_ssf \
#     --tag $tag > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt

# tome_r=4
# tag=$currenttime"_bs64_ToMe_"$tome_r"_efficiency"
# mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/
# echo $corruption
# python3 main_navia_efficiency.py \
#     --batch_size 64 \
#     --workers 8 \
#     --backbone $backbone \
#     --ckpt_dir $ckpt_dir \
#     --data /mnt/data1/xiongyizhe/dataset/imagenet \
#     --data_v2 /mnt/data1/xiongyizhe/dataset/imnet-ood \
#     --data_sketch /mnt/data1/xiongyizhe/dataset/imnet-ood/sketch \
#     --data_corruption /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-c \
#     --data_rendition /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-r \
#     --corruption $corruption \
#     --output ./outputs \
#     --algorithm $algorithm \
#     --apply_tome --tome_r $tome_r --as_baseline \
#     --tag $tag > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt

# prune_token_by_layer=4
# tag=$currenttime"_bs64_EViT_prune_token_by_layer_"$prune_token_by_layer"_efficiency"
# mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/
# echo $corruption
# python3 main.py \
#     --batch_size 64 \
#     --workers 8 \
#     --data /nlp_group/xiongyizhe/datasets/ImageNet \
#     --data_v2 /nlp_group/xiongyizhe/datasets/ImageNet-C \
#     --data_sketch /nlp_group/xiongyizhe/datasets/ImageNet-C/sketch \
#     --data_corruption /nlp_group/xiongyizhe/datasets/ImageNet-C/ImageNet-C \
#     --data_rendition /nlp_group/xiongyizhe/datasets/ImageNet-C/imagenet-r \
#     --corruption $corruption \
#     --output ./outputs \
#     --algorithm $algorithm \
#     --apply_evit --prune_token_by_layer $prune_token_by_layer \
#     --tag $tag --seed $seed > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt

# tofu_r=4
# tag=$currenttime"_bs64_Tofu_"$tofu_r"_sep_6_efficiency"
# mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/
# echo $corruption
# python3 main.py \
#     --batch_size 64 \
#     --workers 8 \
#     --data /nlp_group/xiongyizhe/datasets/ImageNet \
#     --data_v2 /nlp_group/xiongyizhe/datasets/ImageNet-C \
#     --data_sketch /nlp_group/xiongyizhe/datasets/ImageNet-C/sketch \
#     --data_corruption /nlp_group/xiongyizhe/datasets/ImageNet-C/ImageNet-C \
#     --data_rendition /nlp_group/xiongyizhe/datasets/ImageNet-C/imagenet-r \
#     --corruption $corruption \
#     --output ./outputs \
#     --algorithm $algorithm \
#     --apply_tofu --tofu_r $tofu_r --tofu_sep 6 \
#     --tag $tag --seed $seed > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt

# for algorithm in "tent" "lame" "cotta" "sar" "deyo" "foa"; do
#     tag=$currenttime"_bs64_efficiency"
#     mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/
#     echo $corruption
#     python3 main_navia_efficiency.py \
#         --batch_size 64 \
#         --workers 8 \
#         --backbone $backbone \
#         --ckpt_dir $ckpt_dir \
#         --data /mnt/data1/xiongyizhe/dataset/imagenet \
#         --data_v2 /mnt/data1/xiongyizhe/dataset/imnet-ood \
#         --data_sketch /mnt/data1/xiongyizhe/dataset/imnet-ood/sketch \
#         --data_corruption /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-c \
#         --data_rendition /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-r \
#         --corruption $corruption \
#         --output ./outputs \
#         --algorithm $algorithm \
#         --as_baseline --num_prompts 3 \
#         --tag $tag > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
# done

# for algorithm in "foa_bp"; do
#     tag=$currenttime"_bs64_efficiency"
#     mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/
#     echo $corruption
#     python3 main_navia_efficiency.py \
#         --batch_size 64 \
#         --workers 8 \
#         --backbone $backbone \
#         --ckpt_dir $ckpt_dir \
#         --data /mnt/data1/xiongyizhe/dataset/imagenet \
#         --data_v2 /mnt/data1/xiongyizhe/dataset/imnet-ood \
#         --data_sketch /mnt/data1/xiongyizhe/dataset/imnet-ood/sketch \
#         --data_corruption /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-c \
#         --data_rendition /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-r \
#         --corruption $corruption \
#         --output ./outputs \
#         --algorithm $algorithm \
#         --as_baseline --num_prompts 3 \
#         --tag $tag > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
# done