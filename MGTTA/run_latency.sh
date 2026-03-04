export CUDA_VISIBLE_DEVICES=$1

currenttime=`date +"%Y-%m-%d_%H"`
GPU=$1
corruption='gaussian_noise'     # use gaussian for FLOPs counting
lr_cls_token=0.05
lr_cls_ssf=0.05
cls_ssf_layer="0,1,2,3"     # use the one with higher FLOPs
algorithm="foa_bp"


# for tome_r in "10_10_10_9_9_8_8_7_7_6_6_6" "6_5_5_5_4_4_4_4_3_3_3_2"; do
#     tag=$currenttime"_bs64_count_flops"

#     mkdir -p /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/
#     output_log=$corruption"-ToMe_"$tome_r"_cls_ssf_layer_"$cls_ssf_layer"_latency-running-log"

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
#         --apply_tome --tome_r $tome_r --apply_sep_lr_for_cls --lr_cls_token $lr_cls_token --apply_cls_ssf --cls_ssf_layer $cls_ssf_layer --apply_sep_lr_cls_ssf --lr_cls_ssf $lr_cls_ssf \
#         --count_flops \
#         --tag $tag > /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/$output_log.txt
# done

for prune_token_by_layer in 4 8; do
    tag=$currenttime"_bs64_count_flops"

    mkdir -p /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/
    output_log=$corruption"-EViT_prune_token_by_layer_"$prune_token_by_layer"_latency-running-log"

    echo $corruption
    python3 main.py \
        --batch_size 64 \
        --workers 8 \
        --data /nlp_group/xiongyizhe/datasets/ImageNet \
        --data_v2 /nlp_group/xiongyizhe/datasets/ImageNet-C \
        --data_sketch /nlp_group/xiongyizhe/datasets/ImageNet-C/sketch \
        --data_corruption /nlp_group/xiongyizhe/datasets/ImageNet-C/ImageNet-C \
        --data_rendition /nlp_group/xiongyizhe/datasets/ImageNet-C/imagenet-r \
        --corruption $corruption \
        --output ./outputs \
        --algorithm $algorithm \
        --apply_evit --prune_token_by_layer $prune_token_by_layer \
        --test_batch_time \
        --tag $tag > /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/$output_log.txt
done