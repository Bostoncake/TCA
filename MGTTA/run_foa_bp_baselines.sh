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

for seed in 2020; do
    for prune_token_by_layer in 4 8; do
        tag=$currenttime"_bs64_EViT_prune_token_by_layer_"$prune_token_by_layer
        

        mkdir -p /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/

        for corruption in ${corruptions[@]}; do
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
                --tag $tag --seed $seed > /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
        done
    done


    for tofu_r in 4 8; do
        tag=$currenttime"_bs64_Tofu_"$tofu_r"_sep_6_seed_"$seed

        mkdir -p /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/

        for corruption in ${corruptions[@]}; do
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
                --tag $tag --seed $seed > /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
        done
    done
done

# for tome_r in "0_0_0_22_0_0_20_0_0_18_0_0" "0_0_0_12_0_0_11_0_0_11_0_0"; do
#     tag=$currenttime"_bs64_ToMe_"$tome_r"_sched_sweep"

#     mkdir -p /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/

#     for corruption in ${corruptions[@]}; do
#         echo $corruption
#         python3 main.py \
#             --batch_size 64 \
#             --workers 8 \
#             --data /nlp_group/xiongyizhe/datasets/ImageNet \
#             --data_v2 /nlp_group/xiongyizhe/datasets/ImageNet-C \
#             --data_sketch /nlp_group/xiongyizhe/datasets/ImageNet-C/sketch \
#             --data_corruption /nlp_group/xiongyizhe/datasets/ImageNet-C/ImageNet-C \
#             --data_rendition /nlp_group/xiongyizhe/datasets/ImageNet-C/imagenet-r \
#             --corruption $corruption \
#             --output ./outputs \
#             --algorithm $algorithm \
#             --apply_tome --tome_r $tome_r --as_baseline \
#             --tag $tag > /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
#     done
# done