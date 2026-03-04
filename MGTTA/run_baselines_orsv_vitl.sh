export CUDA_VISIBLE_DEVICES=$1

GPU=$1
if [ "$GPU" -eq 0 ]; then
    corruptions=('rendition')
elif [ "$GPU" -eq 1 ]; then
    corruptions=('v2')
elif [ "$GPU" -eq 2 ]; then
    corruptions=('sketch')
elif [ "$GPU" -eq 3 ]; then
    corruptions=('original')
elif [ "$GPU" -eq 4 ]; then
    corruptions=('rendition')
elif [ "$GPU" -eq 5 ]; then
    corruptions=('v2')
elif [ "$GPU" -eq 6 ]; then
    corruptions=('sketch')
elif [ "$GPU" -eq 7 ]; then
    corruptions=('original')
else
    echo "Invalid GPU number: $GPU. Must be 0-7."
    exit 1
fi

currenttime=`date +"%Y-%m-%d_%H"`
backbone="large"
ckpt_dir="/mnt/data1/xiongyizhe/models/vit_large_patch16_224.augreg_in21k_ft_in1k.bin"
batch_size=32

# algorithms=('tent' 'eata' 'sar' 'cotta' 'lame' 'deyo' 'foa' 'foa_bp')
# algorithms=('eata' 'deyo')
algorithms=('noadapt')
num_prompts=3
# seeds=(2020 2021 2022)
seed=2020

# for seed in ${seeds[@]}; do
for algorithm in ${algorithms[@]}; do
    if [ "$algorithm" = "cotta" ]; then
        batch_size=16
    fi

    tag=$currenttime"_"$backbone"_orsv_baselines"

    mkdir -p /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/

    for corruption in ${corruptions[@]}; do
        echo $corruption
        python3 main.py \
            --batch_size $batch_size \
            --workers 8 \
            --backbone $backbone \
            --num_prompts $num_prompts \
            --ckpt_dir $ckpt_dir \
            --data /mnt/data1/xiongyizhe/dataset/imagenet \
            --data_v2 /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-v2 \
            --data_sketch /mnt/data1/xiongyizhe/dataset/imnet-ood/sketch \
            --data_corruption /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-c \
            --data_rendition /mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-r \
            --corruption $corruption \
            --output ./outputs \
            --algorithm $algorithm --as_baseline \
            --tag $tag --seed $seed > /home/xiongyizhe/CVPR2026/MGTTA/outputs/$algorithm"_"$tag/$corruption-running-log.txt
    done
done
# done
# wait
