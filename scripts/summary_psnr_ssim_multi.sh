model_v=RAUNENet
epochs=(50 55 60 65 70 75 80 85 90 95)
output_dir=psnr_ssim_summary
train_set=LSUI3879
net_list=("3blocks2down")
loss_weights=PCONT1_SSIM1_SCONT1


summary(){
    touch "${1}"
    echo epoch,psnr,ssim > "${1}"
    for epoch in ${epochs[@]};
    do
        target_file="results/${model_v}/${2}/epoch_${epoch}/${3}/quantitive_eval.csv"
        if [ -f "${target_file}" ]; then
            psnr=`tail "${target_file}" -n 1 | awk -F, '{print $2}'`
            ssim=`tail "${target_file}" -n 1 | awk -F, '{print $3}'`
            echo ${epoch},${psnr},${ssim} >> "${1}"
        fi
    done
}

# create output directory if it not exist
if [ ! -d ${output_dir} ]
then
    mkdir -p ${output_dir}
fi

if [ ${#net_list[@]} -eq 0 ]
then
    name=${train_set}_${loss_weight}
    testset_list=("LSUI400" "UIEB100" "ocean_ex" "euvp_test515")
    for test_name in ${testset_list[@]}
    do
        output_file=${output_dir}/${train_set}_${model_v}_${loss_weight}_${test_name}.csv
        summary $output_file $name $test_name   
    done
else
    for net in ${net_list[@]}
    do
        testset_list=("LSUI400" "UIEB100" "ocean_ex" "euvp_test515")
        for test_name in ${testset_list[@]}
        do
            name=${train_set}_${net}_${loss_weights}
            output_file=${output_dir}/${train_set}_${model_v}_${net}_${loss_weight}_${test_name}.csv
            summary $output_file $name $test_name
        done
    done
fi