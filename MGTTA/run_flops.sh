export CUDA_VISIBLE_DEVICES=$1

currenttime=`date +"%Y-%m-%d_%H"`
GPU=$1
corruption='gaussian_noise'     # use gaussian for FLOPs counting
backbone="large"
ckpt_dir="/mnt/data1/xiongyizhe/models/vit_large_patch16_224.augreg_in21k_ft_in1k.bin"
batch_size=32
algorithm='foa_bp'

tag=$currenttime"_"$backbone"_count_flops"
mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/
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
    --apply_evit --prune_token_by_layer 4 --as_baseline \
    --count_flops \
    --tag $tag > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt


# lr_cls_token=0.05
# lr_cls_ssf=0.05
# cls_ssf_layer="0,1,2,3"     # use the one with higher FLOPs
# algorithm="foa_bp"


# # for tome_r in "10_10_10_9_9_8_8_7_7_6_6_6" "6_5_5_5_4_4_4_4_3_3_3_2"; do
# #     tag=$currenttime"_bs64_count_flops"

# #     mkdir -p /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/
# #     output_log=$corruption"-ToMe_"$tome_r"_cls_ssf_layer_"$cls_ssf_layer"-running-log"

# #     echo $corruption
# #     python3 main.py \
# #         --batch_size 64 \
# #         --workers 8 \
# #         --data /nlp_group/xiongyizhe/datasets/ImageNet \
# #         --data_v2 /nlp_group/xiongyizhe/datasets/ImageNet-C \
# #         --data_sketch /nlp_group/xiongyizhe/datasets/ImageNet-C/sketch \
# #         --data_corruption /nlp_group/xiongyizhe/datasets/ImageNet-C/ImageNet-C \
# #         --data_rendition /nlp_group/xiongyizhe/datasets/ImageNet-C/imagenet-r \
# #         --corruption $corruption \
# #         --output ./outputs \
# #         --algorithm $algorithm \
# #         --apply_tome --tome_r $tome_r --apply_sep_lr_for_cls --lr_cls_token $lr_cls_token --apply_cls_ssf --cls_ssf_layer $cls_ssf_layer --apply_sep_lr_cls_ssf --lr_cls_ssf $lr_cls_ssf \
# #         --count_flops \
# #         --tag $tag > /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/$output_log.txt
# # done

# for prune_token_by_layer in 4 8; do
#     tag=$currenttime"_bs64_count_flops"

#     mkdir -p /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/
#     output_log=$corruption"-EViT_prune_token_by_layer_"$prune_token_by_layer"-running-log"

#     echo $corruption
#     python3 main.py \
#         --batch_size 64 \
#         --workers 8 \
#         --data /nlp_group/xiongyizhe/datasets/ImageNet \
#         --data_v2 /nlp_group/xiongyizhe/datasets/ImageNet-C \
#         --data_sketch /nlp_group/xiongyizhe/datasets/ImageNet-C/sketch \
#         --data_corruption /nlp_group/xiongyizhe/datasets/ImageNet-C/ImageNet-C \
#         --data_rendition /nlp_group/xiongyizhe/datasets/ImageNet-C/imagenet-r \
#         --corruption $corruption \
#         --output ./outputs \
#         --algorithm $algorithm \
#         --apply_evit --prune_token_by_layer $prune_token_by_layer \
#         --count_flops \
#         --tag $tag > /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/$output_log.txt
# done