export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

# finer, verified.
for k in $( seq 1 8)
do
    CUDA_VISIBLE_DEVICES=2 python fitting.py \
    --photo_name kodim0${k} \
    --photo_address /data0/home/shikexuan/kodim_photo/ \
    --mode finer \
    --patch_size 32 \
    --xuhao 20 \
    --replace_start 0 \
    --epochs 10000 \
    --replace_end 20 \
    --learning_rate 1e-3 \
    --hidden_layers 2 \
    --hidden_features 256 \
    --step_if 
done


# finer+iga
for k in $( seq 1 8)
do
    CUDA_VISIBLE_DEVICES=2 python fitting.py \
    --photo_name kodim0${k} \
    --photo_address /data0/home/shikexuan/kodim_photo/ \
    --mode finer \
    --patch_size 32 \
    --xuhao 15 \
    --replace_start 0 \
    --epochs 10000 \
    --replace_end 15 \
    --learning_rate 1e-3 \
    --hidden_layers 2 \
    --hidden_features 256 \
    --gradient_adjust \
    --step_if 
done