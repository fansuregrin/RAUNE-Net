###########################################################
# This script is for testing multiple epochs automaticlly
###########################################################
read -p "name:" name
read -p "checkpoint_dir:" checkpoint_dir

model_v=FUnIE_GAN

#########################################################################################
# You should change these dataset path to your own.
declare -A ds_dict
ds_dict=([U45]="/DataA/pwz/workshop/Datasets/U45/U45"
         [RUIE_Color90]="/DataA/pwz/workshop/Datasets/RUIE_Color90"
         [UPoor200]="/DataA/pwz/workshop/Datasets/UPoor200_256x256")
#########################################################################################

# Your should change these epochs to those you want to test.
# epochs=(50 55 60 65 70 75 80 85 90 95)
epochs=(50 85 95)

for ds_name in ${!ds_dict[@]};
do
    for epoch in ${epochs[@]};
    do
        python ./test_funiegan.py \
            --name ${name} \
            --test_name ${ds_name} \
            --checkpoint_dir ${checkpoint_dir} \
            --data_dir ${ds_dict[${ds_name}]} \
            --result_dir results/${model_v}/${name}/epoch_${epoch} \
            --epoch ${epoch}
    done
done