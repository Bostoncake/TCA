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
backbone="large"
ckpt_dir="/mnt/data1/xiongyizhe/models/vit_large_patch16_224.augreg_in21k_ft_in1k.bin"
batch_size=32

for seed in 2021 2022 2023; do

    # for tome_r in 2 4; do
    #     tag=$currenttime"_large_ToMe_r_"$tome_r
        

    #     mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/

    #     for corruption in ${corruptions[@]}; do
    #         echo $corruption
    #         python3 main.py \
    #             --batch_size $batch_size \
    #             --workers 8 \
    #             --backbone $backbone \
    #             --ckpt_dir $ckpt_dir \
    #             --data /mnt/data1/xiongyizhe/dataset/imagenet \
    #             --data_v2 /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-v2 \
    #             --data_sketch /mnt/data1/xiongyizhe/dataset/imnet-ood/sketch \
    #             --data_corruption /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-c \
    #             --data_rendition /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-r \
    #             --corruption $corruption \
    #             --output ./outputs \
    #             --algorithm $algorithm \
    #             --apply_tome --tome_r $tome_r --as_baseline \
    #             --tag $tag --seed $seed > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
    #     done
    # done

    for prune_token_by_layer in 2 4; do
        tag=$currenttime"_large_EViT_prune_token_by_layer_"$prune_token_by_layer"_seed_"$seed
        

        mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/

        for corruption in ${corruptions[@]}; do
            echo $corruption
            python3 main.py \
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
                --apply_evit --prune_token_by_layer $prune_token_by_layer \
                --tag $tag --seed $seed > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
        done
    done


    # for tofu_r in 2 4; do
    #     tag=$currenttime"_large_Tofu_"$tofu_r"_sep_12_seed_"$seed

    #     mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/

    #     for corruption in ${corruptions[@]}; do
    #         echo $corruption
    #         python3 main.py \
    #             --batch_size $batch_size \
    #             --workers 8 \
    #             --backbone $backbone \
    #             --ckpt_dir $ckpt_dir \
    #             --data /mnt/data1/xiongyizhe/dataset/imagenet \
    #             --data_v2 /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-v2 \
    #             --data_sketch /mnt/data1/xiongyizhe/dataset/imnet-ood/sketch \
    #             --data_corruption /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-c \
    #             --data_rendition /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-r \
    #             --corruption $corruption \
    #             --output ./outputs \
    #             --algorithm $algorithm \
    #             --apply_tofu --tofu_r $tofu_r --tofu_sep 12 \
    #             --tag $tag --seed $seed > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
    #     done
    # done
done
wait
