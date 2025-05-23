export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

#relu, verified
for k in $( seq 1 8)
do
    CUDA_VISIBLE_DEVICES=0 python fitting.py \
    --photo_name kodim0${k} \
    --photo_address /data0/home/shikexuan/kodim_photo/ \
    --mode relu \
    --patch_size 32 \
    --xuhao 25 \
    --replace_start 0 \
    --epochs 10000 \
    --replace_end 25 \
    --learning_rate 1e-3 \
    --hidden_layers 2 \
    --hidden_features 256 \
    --step_if \
    --saving
done

# # relu+iga, verified
for k in $( seq 1 8)
do
    CUDA_VISIBLE_DEVICES=0 python fitting.py \
    --photo_name kodim0${k} \
    --photo_address /data0/home/shikexuan/kodim_photo/ \
    --mode relu \
    --patch_size 32 \
    --xuhao 25 \
    --replace_start 0 \
    --epochs 10000 \
    --replace_end 25 \
    --learning_rate 5e-3 \
    --hidden_layers 2 \
    --hidden_features 256 \
    --gradient_adjust \
    --step_if \
    --saving
done

# relu+fr, verified
for k in $( seq 1 8)
do
    CUDA_VISIBLE_DEVICES=0 python fitting.py \
    --photo_name kodim0${k} \
    --photo_address /data0/home/shikexuan/kodim_photo/ \
    --mode relu+fr \
    --patch_size 32 \
    --xuhao 25 \
    --replace_start 0 \
    --epochs 10000 \
    --replace_end 25 \
    --learning_rate 5e-3 \
    --hidden_layers 2 \
    --hidden_features 256 \
    --alpha 0.001 \
    --step_if \
    --saving
done

# relu+bn, verified
for k in $( seq 1 8)
do
    CUDA_VISIBLE_DEVICES=0 python fitting.py \
    --photo_name kodim0${k} \
    --photo_address /data0/home/shikexuan/kodim_photo/ \
    --mode relu+bn \
    --patch_size 32 \
    --xuhao 25 \
    --replace_start 0 \
    --epochs 10000 \
    --replace_end 25 \
    --learning_rate 1e-3 \
    --hidden_layers 2 \
    --hidden_features 256 \
    --step_if
done