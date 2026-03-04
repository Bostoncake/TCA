export CUDA_VISIBLE_DEVICES=$1

GPU=$1
if [ "$GPU" -eq 0 ]; then
    corruption='rendition'
    tome_r=4
    lr_cls_token=0.0008
    cls_ssf_layer='0,1,2,3,4,5'
    lr_cls_ssf=0.05
elif [ "$GPU" -eq 1 ]; then
    corruption='v2'
    tome_r=4
    lr_cls_token=0.003
    cls_ssf_layer='0,1,2,3'
    lr_cls_ssf=0.05
elif [ "$GPU" -eq 2 ]; then
    corruption='sketch'
    tome_r=4
    lr_cls_token=0.005
    cls_ssf_layer='0,1,2,3,4,5'
    lr_cls_ssf=0.05
elif [ "$GPU" -eq 3 ]; then
    corruption='original'
    tome_r=4
    lr_cls_token=0.0008
    cls_ssf_layer='0,1,2,3'
    lr_cls_ssf=0.01
elif [ "$GPU" -eq 4 ]; then
    corruption='rendition'
    tome_r=8
    lr_cls_token=0.03
    cls_ssf_layer='0,1,2,3,4,5'
    lr_cls_ssf=0.05
elif [ "$GPU" -eq 5 ]; then
    corruption='v2'
    tome_r=8
    lr_cls_token=0.005
    cls_ssf_layer='0,1,2,3'
    lr_cls_ssf=0.05
elif [ "$GPU" -eq 6 ]; then
    corruption='sketch'
    tome_r=8
    lr_cls_token=0.03
    cls_ssf_layer='0,1,2,3,4,5'
    lr_cls_ssf=0.05
elif [ "$GPU" -eq 7 ]; then
    corruption='original'
    tome_r=8
    lr_cls_token=0.003
    cls_ssf_layer='0,1,2,3'
    lr_cls_ssf=0.01
else
    echo "Invalid GPU number: $GPU. Must be 0-3."
    exit 1
fi

algorithm="foa_bp"
currenttime=`date +"%Y-%m-%d_%H"`
tag=$currenttime"_bs64_ToMe_"$tome_r"_best_params"

mkdir -p /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/

echo $corruption
python3 main_navia.py \
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
    --tag $tag > /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt