export CUDA_VISIBLE_DEVICES=$1

currenttime=`date +"%Y-%m-%d_%H"`
backbone="large"
ckpt_dir="/mnt/data1/xiongyizhe/models/vit_large_patch16_224.augreg_in21k_ft_in1k.bin"
batch_size=32
for tome_r in 2 4; do

    GPU=$1
    if [ "$GPU" -eq 0 ]; then
        corruptions=('gaussian_noise' 'shot_noise')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(3e-3 3e-5)
            lr_cls_ssfs=(1e-2 1e-2)
            cls_ssf_layers=("0,1,2,3,4,5,6,7,8,9,10,11" "0,1,2,3,4,5,6,7,8,9,10,11")
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(1e-2 3e-2)
            lr_cls_ssfs=(3e-2 1e-2)
            cls_ssf_layers=("0,1,2,3,4,5,6,7" "0,1,2,3,4,5,6,7,8,9,10,11")
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 1 ]; then
        corruptions=('impulse_noise' 'defocus_blur')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(3e-3 3e-5)
            lr_cls_ssfs=(1e-2 1e-2)
            cls_ssf_layers=("0,1,2,3,4,5,6,7" "0,1,2,3,4,5")
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(1e-2 1e-2)
            lr_cls_ssfs=(1e-2 1e-2)
            cls_ssf_layers=("0,1,2,3,4,5,6,7,8,9,10,11" "0,1,2,3,4,5,6,7,8,9,10,11")
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 2 ]; then
        corruptions=('glass_blur' 'motion_blur')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(1e-5 1e-5)
            lr_cls_ssfs=(3e-2 1e-2)
            cls_ssf_layers=("0,1,2,3,4,5" "0,1,2,3,4,5,6,7")
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(3e-5 1e-4)
            lr_cls_ssfs=(1e-4 1e-2)
            cls_ssf_layers=("0,1,2,3,4,5,6,7" "0,1,2,3,4,5,6,7")
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 3 ]; then
        corruptions=('zoom_blur' 'snow')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(3e-2 3e-2)
            lr_cls_ssfs=(1e-2 1e-2)
            cls_ssf_layers=("0,1,2,3,4,5" "0,1,2,3,4,5,6,7")
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(1e-5 1e-3)
            lr_cls_ssfs=(3e-3 3e-2)
            cls_ssf_layers=("0,1,2,3,4,5" "0,1,2,3,4,5,6,7")
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 4 ]; then
        corruptions=('frost' 'fog')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(3e-4 1e-3)
            lr_cls_ssfs=(3e-2 1e-2)
            cls_ssf_layers=("0,1,2,3,4,5,6,7,8,9,10,11" "0,1,2,3,4,5")
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(1e-4 3e-4)
            lr_cls_ssfs=(1e-2 3e-2)
            cls_ssf_layers=("0,1,2,3,4,5,6,7,8,9,10,11" "0,1,2,3,4,5,6,7,8,9,10,11")
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 5 ]; then
        corruptions=('brightness' 'jpeg_compression')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(1e-5 3e-2)
            lr_cls_ssfs=(1e-2 3e-2)
            cls_ssf_layers=("0,1,2,3,4,5,6,7,8,9,10,11" "0,1,2,3,4,5,6,7,8,9,10,11")
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(1e-4 1e-2) # actual best: (0.0008 1)
            lr_cls_ssfs=(1e-2 1e-2)
            cls_ssf_layers=("0,1,2,3,4,5,6,7" "0,1,2,3,4,5,6,7,8,9,10,11")
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 6 ]; then
        corruptions=('elastic_transform' 'pixelate')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(3e-2 1e-3)  # actual best: (0.0003 0.5)
            lr_cls_ssfs=(3e-2 1e-5)
            cls_ssf_layers=("0,1,2,3,4,5" "0,1,2,3,4,5")
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(1e-5 1e-5)
            lr_cls_ssfs=(1e-2 1e-2)
            cls_ssf_layers=("0,1,2,3,4,5" "0,1,2,3,4,5")
        else
            echo "Not implemented tome_r:"$tome_r", exit."
            exit 1
        fi
    elif [ "$GPU" -eq 7 ]; then
        corruptions=('contrast')
        if [ "$tome_r" -eq 2 ]; then
            lr_cls_tokens=(1e-5)
            lr_cls_ssfs=(3e-5)
            cls_ssf_layers=("0,1,2,3,4,5")
        elif [ "$tome_r" -eq 4 ]; then
            lr_cls_tokens=(3e-3)
            lr_cls_ssfs=(3e-5)
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
