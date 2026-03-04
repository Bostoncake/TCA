import os
import json

convergence_iter = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
result_dir_list = [
    #f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-07-09_00_bs64_EViT_prune_token_by_layer_8_convergence_iter{conv_iter}" for conv_iter in convergence_iter
    f"/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-07-09_00_bs64_ToMe_8_convergence_iter{conv_iter}" for conv_iter in convergence_iter
]
# C_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
# datasets = ['original', 'rendition', 'v2', 'sketch']
datasets = ['rendition']

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
        # ece_list = []
        for corruption in datasets:

            file_dir = os.path.join(result_dir, f"{corruption}-result.json")

            try:
                with open(file_dir, "r") as f:
                    result = json.load(f)
                
                top1_list.append(result["top1_val"])
                # ece_list.append(result["ECE"])
            except:
                top1_list.append(0)
                # ece_list.append(0)

        # print("*******************ImageNet*******************")
        # print(" ".join(f"{top1:.1f}" for top1 in top1_list))
        # print(" ".join(f"{ece:.1f}" for ece in ece_list))
        top1_list_for_print.append(" ".join(f"{top1:.1f}" for top1 in top1_list))
        # ece_list_for_print.append(" ".join(f"{ece:.1f}" for ece in ece_list))

print("*******************Top-1 Acc*******************")
print(top1_list_for_print)
# for str_item in top1_list_for_print:
#     print(str_item)
# print("**********************ECE**********************")
# for str_item in ece_list_for_print:
#     print(str_item)
