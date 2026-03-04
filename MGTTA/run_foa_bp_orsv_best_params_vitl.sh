export CUDA_VISIBLE_DEVICES=$1

GPU=$1

if [ "$GPU" -eq 0 ]; then
    corruption='rendition'
    tome_r=2
    lr_cls_token=1e-2
    cls_ssf_layer=0,1,2,3,4,5,6,7,8,9,10,11
    lr_cls_ssf=3e-2
elif [ "$GPU" -eq 1 ]; then
    corruption='v2'
    tome_r=2
    lr_cls_token=1e-3
    cls_ssf_layer=0,1,2,3,4,5,6,7
    lr_cls_ssf=3e-4
elif [ "$GPU" -eq 2 ]; then
    corruption='sketch'
    tome_r=2
    lr_cls_token=1e-5
    cls_ssf_layer=0,1,2,3,4,5,6,7,8,9,10,11
    lr_cls_ssf=3e-2
elif [ "$GPU" -eq 3 ]; then
    corruption='original'
    tome_r=2
    lr_cls_token=3e-4
    cls_ssf_layer=0,1,2,3,4,5,6,7
    lr_cls_ssf=3e-3
elif [ "$GPU" -eq 4 ]; then
    corruption='rendition'
    tome_r=4
    lr_cls_token=3e-2
    cls_ssf_layer=0,1,2,3,4,5,6,7
    lr_cls_ssf=3e-2
elif [ "$GPU" -eq 5 ]; then
    corruption='v2'
    tome_r=4
    lr_cls_token=1e-5
    cls_ssf_layer=0,1,2,3,4,5,6,7
    lr_cls_ssf=1e-3
elif [ "$GPU" -eq 6 ]; then
    corruption='sketch'
    tome_r=4
    lr_cls_token=3e-4
    cls_ssf_layer=0,1,2,3,4,5
    lr_cls_ssf=1e-2
elif [ "$GPU" -eq 7 ]; then
    corruption='original'
    tome_r=4
    lr_cls_token=3e-5
    cls_ssf_layer=0,1,2,3,4,5,6,7,8,9,10,11
    lr_cls_ssf=3e-3
else
    echo "Invalid GPU number: $GPU. Must be 0-7."
    exit 1
fi

backbone="large"
ckpt_dir="/mnt/data1/xiongyizhe/models/vit_large_patch16_224.augreg_in21k_ft_in1k.bin"
batch_size=32

algorithm="foa_bp"
currenttime=`date +"%Y-%m-%d_%H"`
run=1
    
tag=$currenttime"_large_ToMe_"$tome_r"_best_params_for_lr_cls_lr_cls_ssf_and_layer_run"$run

mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/


echo $corruption
python3 main_navia.py \
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
    --apply_tome --tome_r $tome_r --apply_sep_lr_for_cls --lr_cls_token $lr_cls_token --apply_cls_ssf --cls_ssf_layer $cls_ssf_layer --apply_sep_lr_cls_ssf --lr_cls_ssf $lr_cls_ssf \
    --tag $tag > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt
