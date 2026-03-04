export CUDA_VISIBLE_DEVICES=$1

currenttime=`date +"%Y-%m-%d_%H"`
backbone="base"
ckpt_dir="/mnt/data1/xiongyizhe/models/vit_base_patch16_224.augreg2_in21k_ft_in1k.bin"
for tome_r in 4 8; do

    GPU=$1
    if [ "$GPU" -eq 0 ]; then
        corruptions=('gaussian_noise' 'shot_noise')
        if [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(0.05 0.003)
            lr_cls_ssfs=(0.01 0.01)
            cls_ssf_layers=("0,1,2,3,4,5" "0,1,2,3")
        elif [ "$tome_r" -eq 8 ]; then
            lr_cls_tokens=(0.08 0.3)
            lr_cls_ssfs=(0.005 0.005)
            cls_ssf_layers=("0,1,2,3,4,5" "0,1,2,3,4,5")
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 1 ]; then
        corruptions=('impulse_noise' 'defocus_blur')
        if [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(0.0008 0.0008)       # actual best: (0.0008 0.0001)
            lr_cls_ssfs=(0.05 0.01)
            cls_ssf_layers=("0,1,2,3,4,5" "0,1,2,3,4,5")
        elif [ "$tome_r" -eq 8 ]; then
            lr_cls_tokens=(0.0008 0.001)
            lr_cls_ssfs=(0.01 0.005)
            cls_ssf_layers=("0,1,2,3,4,5" "0,1,2,3")
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 2 ]; then
        corruptions=('glass_blur' 'motion_blur')
        if [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(0.0008 0.001)
            lr_cls_ssfs=(0.05 0.05)
            cls_ssf_layers=("0,1,2,3,4,5" "0,1,2,3,4,5")
        elif [ "$tome_r" -eq 8 ]; then
            lr_cls_tokens=(0.08 0.05)
            lr_cls_ssfs=(0.05 0.01)
            cls_ssf_layers=("0,1,2,3,4,5" "0,1,2,3")
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 3 ]; then
        corruptions=('zoom_blur' 'snow')
        if [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(0.001 0.003)
            lr_cls_ssfs=(0.01 0.05)
            cls_ssf_layers=("0,1,2,3" "0,1,2,3,4,5")
        elif [ "$tome_r" -eq 8 ]; then
            lr_cls_tokens=(0.0008 0.0008)   # actual best: (0.0001 0.0008)
            lr_cls_ssfs=(0.01 0.05)
            cls_ssf_layers=("0,1,2,3" "0,1,2,3")
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 4 ]; then
        corruptions=('frost' 'fog')
        if [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(0.0008 0.001)
            lr_cls_ssfs=(0.05 0.05)
            cls_ssf_layers=("0,1,2,3" "0,1,2,3")
        elif [ "$tome_r" -eq 8 ]; then
            lr_cls_tokens=(0.001 0.0008)
            lr_cls_ssfs=(0.05 0.05)
            cls_ssf_layers=("0,1,2,3,4,5" "0,1,2,3,4,5")
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 5 ]; then
        corruptions=('brightness' 'jpeg_compression')
        if [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(0.0008 0.05)
            lr_cls_ssfs=(0.001 0.05)
            cls_ssf_layers=("0,1,2,3" "0,1,2,3")
        elif [ "$tome_r" -eq 8 ]; then
            lr_cls_tokens=(0.0008 0.08) # actual best: (0.0008 1)
            lr_cls_ssfs=(0.001 0.05)
            cls_ssf_layers=("0,1,2,3,4,5" "0,1,2,3,4,5")
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 6 ]; then
        corruptions=('elastic_transform' 'pixelate')
        if [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(0.0003 0.1)  # actual best: (0.0003 0.5)
            lr_cls_ssfs=(0.05 0.01)
            cls_ssf_layers=("0,1,2,3" "0,1,2,3")
        elif [ "$tome_r" -eq 8 ]; then
            lr_cls_tokens=(0.0003 0.1)
            lr_cls_ssfs=(0.05 0.001)
            cls_ssf_layers=("0,1,2,3,4,5" "0,1,2,3,4,5")
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 7 ]; then
        corruptions=('contrast')
        if [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(0.001)
            lr_cls_ssfs=(0.01)
            cls_ssf_layers=("0,1,2,3")
        elif [ "$tome_r" -eq 8 ]; then
            lr_cls_tokens=(0.001)
            lr_cls_ssfs=(0.05)
            cls_ssf_layers=("0,1,2,3,4,5")
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    else
        echo "Invalid GPU number: $GPU. Must be 0-7."
        exit 1
    fi

    algorithm="foa_bp"

    # TODO: add super-parameters for tuning here
    for run in 1 2; do

        tag=$currenttime"_bs64_ToMe_"$tome_r"_best_params_for_lr_cls_lr_cls_ssf_and_layer_run"$run"_no_shift"

        mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/

        for i in ${!corruptions[@]}; do
            corruption=${corruptions[i]}
            lr_cls_token=${lr_cls_tokens[i]}
            lr_cls_ssf=${lr_cls_ssfs[i]}
            cls_ssf_layer=${cls_ssf_layers[i]}

            echo $corruption
            python3 main_navia.py \
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
                --apply_tome --tome_r $tome_r --apply_sep_lr_for_cls --lr_cls_token $lr_cls_token --apply_cls_ssf --cls_ssf_layer $cls_ssf_layer --apply_sep_lr_cls_ssf --lr_cls_ssf $lr_cls_ssf \
                --tag $tag > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
        done
    done
done
