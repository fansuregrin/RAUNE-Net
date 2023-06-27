###########################################################
# This script is for testing multiple epochs automaticlly
###########################################################
read -p "name:" name
read -p "checkpoint_dir:" checkpoint_dir
read -p "num_down:" num_down
read -p "num_blocks:" num_blocks
read -p "use_att_up:" use_att_up  # 'true' or 'false'

model_v=RAUNENet

py_use_att_up=""
if [ ${use_att_up} == "true" ];
then
    py_use_att_up="--use_att_up"
fi

#########################################################################################
# You should change these dataset path to your own. 
declare -A ds_dict
ds_dict=([euvp_test515]="/DataA/pwz/workshop/Datasets/EUVP_Dataset/test_samples/Inp"
         [ocean_ex]="/DataA/pwz/workshop/Datasets/ocean_ex/poor"
         [UIEB100]="/DataA/pwz/workshop/Datasets/UIEB100/raw"
         [LSUI400]="/DataA/pwz/workshop/Datasets/LSUI400/input")
declare -A refer_dict
refer_dict=([euvp_test515]="/DataA/pwz/workshop/Datasets/EUVP_Dataset/test_samples/GTr"
            [ocean_ex]="/DataA/pwz/workshop/Datasets/ocean_ex/good"
            [UIEB100]="/DataA/pwz/workshop/Datasets/UIEB100/reference"
            [LSUI400]="/DataA/pwz/workshop/Datasets/LSUI400/GT")
#########################################################################################

# Your should change these epochs to those you want to test.
# epochs=(50 55 60 65 70 75 80 85 90 95)
epochs=(85 90 95)

for ds_name in ${!ds_dict[@]};
do
    for epoch in ${epochs[@]};
    do
        python ./test_raunenet.py \
            --name ${name} \
            --test_name ${ds_name} \
            --data_dir ${ds_dict[${ds_name}]} \
            --checkpoint_dir ${checkpoint_dir} \
            --result_dir results/${model_v}/${name}/epoch_${epoch} \
            --epoch ${epoch} \
            --num_down ${num_down} \
            --num_blocks ${num_blocks} \
            ${py_use_att_up}
        python ./calc_psnr_ssim.py \
            --input_dir results/${model_v}/${name}/epoch_${epoch}/${ds_name}/single/predicted \
            --refer_dir ${refer_dict[${ds_name}]} \
            --output_dir results/${model_v}/${name}/epoch_${epoch}/${ds_name} \
            --resize --width 256 --height 256
    done
done