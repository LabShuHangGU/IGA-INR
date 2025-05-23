export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

# sin, verified.
for k in $( seq 1 8)
do
    CUDA_VISIBLE_DEVICES=2 python fitting.py \
    --photo_name kodim0${k} \
    --photo_address /data0/home/shikexuan/kodim_photo/ \
    --mode sin \
    --patch_size 32 \
    --xuhao 20 \
    --replace_start 0 \
    --epochs 10000 \
    --replace_end 20 \
    --learning_rate 1e-3 \
    --hidden_layers 2 \
    --hidden_features 256 \
    --saving \
    --step_if 
done

# # sin+iga, verified.
for k in $( seq 3 3)
do
    CUDA_VISIBLE_DEVICES=4 python fitting.py \
    --photo_name kodim0${k} \
    --photo_address /data0/home/shikexuan/kodim_photo/ \
    --mode sin \
    --patch_size 32 \
    --xuhao 20 \
    --replace_start 0 \
    --epochs 10000 \
    --replace_end 20 \
    --learning_rate 1e-3 \
    --hidden_layers 2 \
    --hidden_features 256 \
    --gradient_adjust \
    --step_if 
done


# sin+fr, verified. Due to the complex interaction among the three hyperparameters of the model, 
# we directly adopt the open-source settings for F and P, while tuning alpha for each image.
for k in $( seq 8 8)
do
    CUDA_VISIBLE_DEVICES=2 python fitting.py \
    --photo_name kodim0${k} \
    --photo_address /data0/home/shikexuan/kodim_photo/ \
    --mode sin+fr \
    --patch_size 32 \
    --xuhao 20 \
    --replace_start 0 \
    --epochs 10000 \
    --replace_end 20 \
    --learning_rate 1e-3 \
    --hidden_layers 2 \
    --hidden_features 256 \
    --alpha 0.004 \
    --saving \
    --step_if 
done


