export CUDA_VISIBLE_DEVICES=$1

GPU=$1

for tome_r in 2 4; do
    if [ "$GPU" -eq 0 ]; then
        corruptions=('gaussian_noise' 'shot_noise')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(3e-3 3e-5)
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(1e-2 3e-2)
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 1 ]; then
        corruptions=('impulse_noise' 'defocus_blur')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(3e-3 3e-5)
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(1e-2 1e-2)
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 2 ]; then
        corruptions=('glass_blur' 'motion_blur')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(1e-5 1e-5)
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(3e-5 1e-4)
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 3 ]; then
        corruptions=('zoom_blur' 'snow')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(3e-2 3e-2)
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(1e-5 1e-3)
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 4 ]; then
        corruptions=('frost' 'fog')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(3e-4 1e-3)
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(1e-4 3e-4)
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 5 ]; then
        corruptions=('brightness' 'jpeg_compression')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(1e-5 3e-2)
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(1e-4 1e-2) # actual best: (0.0008 1)
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 6 ]; then
        corruptions=('elastic_transform' 'pixelate')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(3e-2 1e-3)  # actual best: (0.0003 0.5)
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(1e-5 1e-5)
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 7 ]; then
        corruptions=('contrast')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(1e-5)
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(3e-3)
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    else
        echo "Invalid GPU number: $GPU. Must be 0-7."
        exit 1
    fi

    backbone="large"
    ckpt_dir="/mnt/data1/xiongyizhe/models/vit_large_patch16_224.augreg_in21k_ft_in1k.bin"
    batch_size=32

    algorithm="foa_bp"
    currenttime=`date +"%Y-%m-%d_%H"`
    for cls_ssf_layer in "0,1,2,3,4,5" "0,1,2,3,4,5,6,7" "0,1,2,3,4,5,6,7,8,9,10,11"; do
        for lr_cls_ssf in 3e-2 1e-2 3e-3 1e-3 3e-4 1e-4 3e-5 1e-5; do
        
            tag=$currenttime"_bs64_ToMe_"$tome_r"_cls_ssf_layer_"$cls_ssf_layer"_lr_cls_ssf_"$lr_cls_ssf"_param_search_stage2"

            mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/

            for i in ${!corruptions[@]}; do

                corruption=${corruptions[i]}
                lr_cls_token=${lr_cls_tokens[i]}

                echo $corruption
                python3 main.py \
                    --batch_size $batch_size \
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
                    --apply_tome --tome_r $tome_r --apply_sep_lr_for_cls --lr_cls_token $lr_cls_token --apply_cls_ssf --cls_ssf_layer $cls_ssf_layer --apply_sep_lr_cls_ssf --lr_cls_ssf $lr_cls_ssf \
                    --tag $tag > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
            done
        done
    done

done
# for lr_cls_ssf in 0.05 0.01 0.005 0.001; do
#     for cls_ssf_layer in "0,1,2,3" "0,1,2,3,4,5"; do
#         lr_cls_token=0
#         # cls_ssf_layer="3,6,9"
#         for tome_r in 4 8; do
#             tag=$currenttime"_bs64_ToMe_"$tome_r"_train_cls_token_lr_"$lr_cls_token"_apply_cls_ssf_layer_"$cls_ssf_layer"_cls_ssf_lr_"$lr_cls_ssf"_only_bias_for_idea_ensuring"

#             mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/

#             for corruption in ${corruptions[@]}; do
#                 echo $corruption
#                 python3 main.py \
#                     --batch_size $batch_size \
#                     --workers 8 \
#                     --backbone $backbone \
#                     --ckpt_dir $ckpt_dir \
#                     --data /mnt/data1/xiongyizhe/dataset/imagenet \
#                     --data_v2 /mnt/data1/xiongyizhe/dataset/imnet-ood \
#                     --data_sketch /mnt/data1/xiongyizhe/dataset/imnet-ood/sketch \
#                     --data_corruption /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-c \
#                     --data_rendition /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-r \
#                     --corruption $corruption \
#                     --output ./outputs \
#                     --algorithm $algorithm \
#                     --apply_tome --tome_r $tome_r --apply_sep_lr_for_cls --lr_cls_token $lr_cls_token --apply_cls_ssf --cls_ssf_layer $cls_ssf_layer --apply_sep_lr_cls_ssf --lr_cls_ssf $lr_cls_ssf \
#                     --tag $tag > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
#             done
#         done
#     done
# done


# debug vitl
# lr_cls_ssf=0.05
# cls_ssf_layer="0,1,2,3"
# lr_cls_token=0.05
# tome_r=4
# tag=$currenttime"_bs64_ToMe_"$tome_r"_train_cls_token_lr_"$lr_cls_token"_apply_cls_ssf_layer_"$cls_ssf_layer"_cls_ssf_lr_"$lr_cls_ssf"_debug"

# mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/

# for corruption in ${corruptions[@]}; do
#     echo $corruption
#     python3 main.py \
#         --batch_size $batch_size \
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
#         --apply_tome --tome_r $tome_r --apply_sep_lr_for_cls --lr_cls_token $lr_cls_token --apply_cls_ssf --cls_ssf_layer $cls_ssf_layer --apply_sep_lr_cls_ssf --lr_cls_ssf $lr_cls_ssf \
#         --tag $tag > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
# done
