read -p "name:" name
read -p "test_set:" test_set 
read -p "model_v:" model
read -p "output filename:" out_file

epochs=(50 55 60 65 70 75 80 85 90 95)

touch $out_file
echo epoch,psnr,ssim >> $out_file
for epoch in ${epochs[@]};
do
    psnr=`tail results/$model/$name/epoch_$epoch/$test_set/quantitive_eval.csv -n 1 \
        | awk -F, '{print $2}'`
    ssim=`tail results/$model/$name/epoch_$epoch/$test_set/quantitive_eval.csv -n 1 \
        | awk -F, '{print $3}'`
    echo $epoch,$psnr,$ssim >> $out_file
done