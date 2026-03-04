export CUDA_VISIBLE_DEVICES=$1

corruption='rendition'
tofu_r=8
prune_token_by_layer=8
tome_r=8

algorithm="foa_bp"
currenttime=`date +"%Y-%m-%d_%H"`

for seed in 2020; do
    for num_adapt_iters in 25 50 75 100 125 150 175 200 225 250; do
        tag=$currenttime"_bs64_EViT_prune_token_by_layer_"$prune_token_by_layer"_convergence_iter"$num_adapt_iters
        

        mkdir -p /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/

        echo $corruption
        python3 main_convergence.py \
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
            --num_adapt_iters $num_adapt_iters --start_validation 312 \
            --tag $tag --seed $seed > /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt


        tag=$currenttime"_bs64_Tofu_"$tofu_r"_sep_6_convergence_iter"$num_adapt_iters

        mkdir -p /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/

        echo $corruption
        python3 main_convergence.py \
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
            --num_adapt_iters $num_adapt_iters --start_validation 312 \
            --tag $tag --seed $seed > /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt
        
        tag=$currenttime"_bs64_ToMe_"$tome_r"_convergence_iter"$num_adapt_iters

        mkdir -p /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/

        echo $corruption
        python3 main_convergence.py \
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
            --apply_tome --tome_r $tome_r --as_baseline \
            --num_adapt_iters $num_adapt_iters --start_validation 312 \
            --tag $tag --seed $seed > /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt
    done
done