export CUDA_VISIBLE_DEVICES=$1

GPU=$1
if [ "$GPU" -eq 0 ]; then
    corruption='rendition'
    tofu_r=4
    prune_token_by_layer=4
elif [ "$GPU" -eq 1 ]; then
    corruption='v2'
    tofu_r=4
    prune_token_by_layer=4
elif [ "$GPU" -eq 2 ]; then
    corruption='sketch'
    tofu_r=4
    prune_token_by_layer=4
elif [ "$GPU" -eq 3 ]; then
    corruption='original'
    tofu_r=4
    prune_token_by_layer=4
elif [ "$GPU" -eq 4 ]; then
    corruption='rendition'
    tofu_r=8
    prune_token_by_layer=8
elif [ "$GPU" -eq 5 ]; then
    corruption='v2'
    tofu_r=8
    prune_token_by_layer=8
elif [ "$GPU" -eq 6 ]; then
    corruption='sketch'
    tofu_r=8
    prune_token_by_layer=8
elif [ "$GPU" -eq 7 ]; then
    corruption='original'
    tofu_r=8
    prune_token_by_layer=8
else
    echo "Invalid GPU number: $GPU. Must be 0-3."
    exit 1
fi

algorithm="foa_bp"
currenttime=`date +"%Y-%m-%d_%H"`

for seed in 2020; do
    tag=$currenttime"_bs64_EViT_prune_token_by_layer_"$prune_token_by_layer
    

    mkdir -p /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/

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
        --tag $tag --seed $seed > /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt


    tag=$currenttime"_bs64_Tofu_"$tofu_r"_sep_6"

    mkdir -p /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/

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
        --apply_tofu --tofu_r $tofu_r --tofu_sep 6 \
        --tag $tag --seed $seed > /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt
done