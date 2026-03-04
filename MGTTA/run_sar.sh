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
    corruptions=('brightness' 'contrast')
elif [ "$GPU" -eq 6 ]; then
    corruptions=('elastic_transform' 'pixelate')
elif [ "$GPU" -eq 7 ]; then
    corruptions=('jpeg_compression')
else
    echo "Invalid GPU number: $GPU. Must be 0-7."
    exit 1
fi

algorithm="sar"
currenttime=`date +"%Y-%m-%d_%H"`
tag=$currenttime"_bs64"

mkdir -p /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/

for corruption in ${corruptions[@]}; do
    echo $corruption
    python3 main.py \
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
        --tag $tag > /nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/$algorithm"_"$tag/C/$corruption-running-log.txt
done