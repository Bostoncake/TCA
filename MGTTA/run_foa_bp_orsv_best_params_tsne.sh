export CUDA_VISIBLE_DEVICES=$1

corruption='rendition'
tome_r=8
lr_cls_token=0.03
cls_ssf_layer='0,1,2,3,4,5'
lr_cls_ssf=0.05

algorithm="foa_bp"
currenttime=`date +"%Y-%m-%d_%H"`
for num_adapt_iters in 75; do
    tag=$currenttime"_bs64_ToMe_"$tome_r"_best_params_tsne_iter"$num_adapt_iters

    mkdir -p /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/

    echo $corruption
    python3 main_tsne.py \
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
        --apply_tome --tome_r $tome_r --apply_sep_lr_for_cls --lr_cls_token $lr_cls_token --apply_cls_ssf --cls_ssf_layer $cls_ssf_layer --apply_sep_lr_cls_ssf --lr_cls_ssf $lr_cls_ssf \
        --num_adapt_iters $num_adapt_iters --start_validation 312 \
        --tag $tag > /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt
done