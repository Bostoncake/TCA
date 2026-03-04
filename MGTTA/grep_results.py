import os
import json

result_dir_list = [
    # f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-19_19_bs64_ToMe_8_img_feats_loss_{lambda_img_feat}/C" for lambda_img_feat in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    # f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-20_19_bs64_ToMe_8_img_feats_loss_15_train_cls_token_lr_{lr}/C" for lr in [0.5, 1.0]
    # f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-20_20_bs64_ToMe_8_img_feats_loss_15_train_cls_token_lr_{lr}_init_cls_token/C" for lr in [0.1, 0.01, 0.05, 0.005]
    # f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-22_14_bs64_ToMe_8_train_cls_token_lr_{lr}no_img_feat_loss/C" for lr in [0.1, 0.01, 0.05, 0.005]
    # f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-23_00_bs64_ToMe_8_train_cls_token_lr_0.05_apply_protect_prompt_protect_{protect_ratio}/C" for protect_ratio in [0.1, 0.2, 0.3, 0.4]
    # "/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-22_11_bs64_plain_foabp_cls_token_ft_run3/C"
    # f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-23_14_bs64_ToMe_8_train_cls_token_lr_0.05_apply_cls_ssf_layer_{layer}/C" for layer in ["3,6,9", "1,4,7,10", "1,3,5,7,9", "0,2,4,6,8,10", "0,1,2,3,4,5,6,7,8,9,10,11"]
    # f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-27_19_bs64_ToMe_8_best_params_for_cls_lr_ssf_and_layer_NOW_tune_lr_cls_token_{lr}/C" for lr in [1, 0.8, 0.5, 0.3, 0.1, 0.08, 0.05, 0.03, 0.01, 0.008, 0.005, 0.003, 0.001, 0.0008, 0.0005, 0.0003, 0.0001]
    # f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-28_10_bs64_ToMe_8_best_params_for_lr_cls_lr_cls_ssf_and_layer_run{run}/C" for run in [1, 2]
    # f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_hist_2025-05-28_21_bs64_ToMe_8_best_params_for_lr_cls_lr_cls_ssf_and_layer_NOW_learnable_shift_lr_{lr}_tune_hist/C" for lr in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, "0.00005"]
    # f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-29_15_bs64_ToMe_8_best_params_for_lr_cls_lr_cls_ssf_and_layer_run{run}_no_shift/C" for run in [1,2]
    # "/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-23_10_bs64_ToMe_8_train_cls_token_lr_0.05_rerun/C"
    # "/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-20_15_bs64_ToMe_4_img_feats_loss_15_train_cls_token/C",
    # "/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-20_15_bs64_ToMe_8_img_feats_loss_15_train_cls_token/C"
    # "/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-30_09_bs64_ToMe_8"
    # "/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-30_10_bs64"
    # f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-06-02_18_bs64_ToMe_{tome_sched}_sched_sweep/C" for tome_sched in ["4", "10_10_10", "8_8_8_8", "6_6_6_6_6_6"]
    # "/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-06-03_20_bs64_EViT_prune_token_by_layer_4",
    # "/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-06-03_20_bs64_EViT_prune_token_by_layer_8",
    # "/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-06-03_20_bs64_Tofu_4_sep_6",
    # "/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-06-03_20_bs64_Tofu_8_sep_6"
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-10-29_19_bs64_ToMe_4_best_params_for_lr_cls_lr_cls_ssf_and_layer_run1_no_shift/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-10-29_19_bs64_ToMe_4_best_params_for_lr_cls_lr_cls_ssf_and_layer_run2_no_shift/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-10-29_19_bs64_ToMe_8_best_params_for_lr_cls_lr_cls_ssf_and_layer_run1_no_shift/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-10-29_19_bs64_ToMe_8_best_params_for_lr_cls_lr_cls_ssf_and_layer_run2_no_shift/C'
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/tent_2025-10-28_17_large_bs64/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/lame_2025-10-28_17_large_bs64/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/cotta_2025-10-29_09_large_bs64/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/sar_2025-10-28_17_large_bs64/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/eata_2025-10-28_17_large_bs64/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/deyo_2025-10-28_22_large_bs64/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_2025-10-30_11_large_bs64/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-03_19_large_bs64/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-05_00_large_ToMe_r_2/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-05_00_large_ToMe_r_4/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-05_00_large_EViT_prune_token_by_layer_2/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-05_00_large_EViT_prune_token_by_layer_4/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-05_00_large_Tofu_2_sep_12_seed_2020/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-05_00_large_Tofu_4_sep_12_seed_2020/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/tent_2025-11-03_23_large_orsv_baselines',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/lame_2025-11-03_23_large_orsv_baselines',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/cotta_2025-11-03_23_large_orsv_baselines',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/sar_2025-11-03_23_large_orsv_baselines',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/eata_2025-11-03_23_large_orsv_baselines',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/deyo_2025-11-03_23_large_orsv_baselines',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_2025-11-03_23_large_orsv_baselines',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-03_23_large_orsv_baselines',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-05_22_large_ToMe_r_2',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-05_22_large_ToMe_r_4',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-05_22_large_EViT_prune_token_by_layer_2',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-05_22_large_EViT_prune_token_by_layer_4',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-05_22_large_Tofu_2_sep_12',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-05_22_large_Tofu_4_sep_12'
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/noadapt_2025-11-13_17_large_bs64/C',
    # '/home/xiongyizhe/CVPR2026/MGTTA/outputs/noadapt_2025-11-13_18_large_orsv_baselines'
    '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-13_00_large_EViT_prune_token_by_layer_2_seed_2021/C',
    '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-13_00_large_EViT_prune_token_by_layer_2_seed_2022/C',
    '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-13_00_large_EViT_prune_token_by_layer_2_seed_2023/C',
    '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-13_00_large_EViT_prune_token_by_layer_4_seed_2021/C',
    '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-13_00_large_EViT_prune_token_by_layer_4_seed_2022/C',
    '/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2025-11-13_00_large_EViT_prune_token_by_layer_4_seed_2023/C'
]

# for lr_cls_token in [1, 0.8, 0.5, 0.3, 0.1, 0.08, 0.05, 0.03, 0.01, 0.008, 0.005, 0.003, 0.001, 0.0008, 0.0005, 0.0003, 0.0001]:
#     for lr_cls_ssf in [0.05, 0.01, 0.005, 0.001]:
#         for cls_ssf_layer in ["0,1,2,3", "0,1,2,3,4,5"]:
#             result_dir_list.append(f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-29_21_bs64_ToMe_8_train_cls_token_lr_{lr_cls_token}_apply_cls_ssf_layer_{cls_ssf_layer}_cls_ssf_lr_{lr_cls_ssf}_only_bias_on_orsv")

# result_dir_list=[]
# for ssf_lr in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
#     result_dir_list.extend([
#         f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-23_16_bs64_ToMe_8_train_cls_token_lr_0.05_apply_cls_ssf_layer_{layer}_cls_ssf_lr_{ssf_lr}/C" for layer in ["3,6,9", "1,4,7,10", "1,3,5,7,9", "0,2,4,6,8,10", "0,1,2,3,4,5,6,7,8,9,10,11"]
#     ])

# result_dir_list=[]
# for ssf_lr in [0.05, 0.01, 0.005, 0.001]:
#     result_dir_list.extend([
#         f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-27_15_bs64_ToMe_8_train_cls_token_lr_0.05_apply_cls_ssf_layer_{layer}_cls_ssf_lr_{ssf_lr}_only_bias_for_idea_ensuring/C" for layer in ["0,1,2,3", "0,1,2,3,4,5"]
#     ])


# result_dir_list = [
#     "/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-05-26_11_bs64_ToMe_4_train_cls_token_lr_0.05_apply_cls_ssf_layer_0,2,4,6,8,10_cls_ssf_lr_0.05_only_bias/C"
# ]

C_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
datasets = ['original', 'rendition', 'v2', 'sketch']

top1_list_for_print = []
ece_list_for_print = []
for result_dir in result_dir_list:
    if result_dir[-2:] == "/C":
        top1_list = []
        ece_list = []
        for corruption in C_corruptions:

            file_dir = os.path.join(result_dir, f"{corruption}-result.json")

            try:
                with open(file_dir, "r") as f:
                    result = json.load(f)
                
                top1_list.append(result["top1"])
                ece_list.append(result["ECE"])
            except:
                top1_list.append(0)
                ece_list.append(0)

        # print("*******************Corruption*******************")
        # print(" ".join(f"{top1:.1f}" for top1 in top1_list))
        # print(" ".join(f"{ece:.1f}" for ece in ece_list))
        top1_list_for_print.append(" ".join(f"{top1:.3f}" for top1 in top1_list))
        ece_list_for_print.append(" ".join(f"{ece:.1f}" for ece in ece_list))

    else:
        top1_list = []
        ece_list = []
        for corruption in datasets:

            file_dir = os.path.join(result_dir, f"{corruption}-result.json")

            try:
                with open(file_dir, "r") as f:
                    result = json.load(f)
                
                top1_list.append(result["top1"])
                ece_list.append(result["ECE"])
            except:
                top1_list.append(0)
                ece_list.append(0)

        # print("*******************ImageNet*******************")
        # print(" ".join(f"{top1:.1f}" for top1 in top1_list))
        # print(" ".join(f"{ece:.1f}" for ece in ece_list))
        top1_list_for_print.append(" ".join(f"{top1:.1f}" for top1 in top1_list))
        ece_list_for_print.append(" ".join(f"{ece:.1f}" for ece in ece_list))

print("*******************Top-1 Acc*******************")
for str_item in top1_list_for_print:
    print(str_item)
print("**********************ECE**********************")
for str_item in ece_list_for_print:
    print(str_item)
