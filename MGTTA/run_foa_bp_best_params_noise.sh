export CUDA_VISIBLE_DEVICES=$1

currenttime=`date +"%Y-%m-%d_%H"`
backbone="base"
ckpt_dir="/mnt/data1/xiongyizhe/models/vit_base_patch16_224.augreg2_in21k_ft_in1k.bin"
tome_r=8

corruption='gaussian_noise'

for gaussian_noise in 0.25 0.5 0.75; do
    GPU=$1
    if [ "$GPU" -eq 0 ]; then
        lr_cls_token=0.08
        lr_cls_ssf=0.005
        cls_ssf_layer="0,1,2,3,4,5"

        algorithm="foa_bp"

        # TODO: add super-parameters for tuning here
        tag=$currenttime"_bs64_ToMe_"$tome_r"_best_params_noise_"${gaussian_noise}

        mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/
        echo $corruption
        python3 main_navia_noise.py \
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
            --tag $tag --gaussian_noise ${gaussian_noise} > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
    elif [ "$GPU" -eq 1 ]; then
        algorithm="foa_bp"
        tag=$currenttime"_bs64_noise_"${gaussian_noise}

        mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/
        echo $corruption
        python3 main_navia_noise.py \
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
            --algorithm $algorithm --as_baseline\
            --tag $tag --gaussian_noise ${gaussian_noise} > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
    elif [ "$GPU" -eq 2 ]; then
        algorithm="foa"
        tag=$currenttime"_bs64_noise_"${gaussian_noise}

        mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/
        echo $corruption
        python3 main_navia_noise.py \
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
            --algorithm $algorithm --as_baseline --num_prompts 3 \
            --tag $tag --gaussian_noise ${gaussian_noise} > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
    elif [ "$GPU" -eq 3 ]; then
        algorithm="cotta"
        tag=$currenttime"_bs64_noise_"${gaussian_noise}

        mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/
        echo $corruption
        python3 main_navia_noise.py \
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
            --algorithm $algorithm --as_baseline\
            --tag $tag --gaussian_noise ${gaussian_noise} > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
    elif [ "$GPU" -eq 4 ]; then
        algorithm="eata"
        tag=$currenttime"_bs64_noise_"${gaussian_noise}

        mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/
        echo $corruption
        python3 main_navia_noise.py \
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
            --algorithm $algorithm --as_baseline\
            --tag $tag --gaussian_noise ${gaussian_noise} > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
    elif [ "$GPU" -eq 5 ]; then
        algorithm="tent"
        tag=$currenttime"_bs64_noise_"${gaussian_noise}

        mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/
        echo $corruption
        python3 main_navia_noise.py \
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
            --algorithm $algorithm --as_baseline\
            --tag $tag --gaussian_noise ${gaussian_noise} > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
    elif [ "$GPU" -eq 6 ]; then
        algorithm="tent"
        tag=$currenttime"_bs64_noise_"${gaussian_noise}

        mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/
        echo $corruption
        python3 main_navia_noise.py \
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
            --algorithm $algorithm --as_baseline\
            --tag $tag --gaussian_noise ${gaussian_noise} > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
    fi
done