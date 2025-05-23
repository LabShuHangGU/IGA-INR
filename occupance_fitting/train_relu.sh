export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

GPU_ID=2

# # In a single iteration, more data points do not provide the baseline model with significant gains. 
# # relu, verified
for name in 'thai' 'arma' 'dragon' 'bun' 'lucy'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode relu \
  --xuhao 8 \
  --replace_start 0 \
  --replace_end 8 \
  --random_index \
  --learning_rate 2e-3 \
  --epochs 200 \
  --group_num 512 \
  --alpha 0.01 \
  --group_size 256 \
  --saving
done

# # relu+iga
for name in 'thai' 'arma' 'dragon' 'bun' 'lucy'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode relu \
  --xuhao 7 \
  --replace_start 0 \
  --replace_end 7 \
  --random_index \
  --learning_rate 5e-3 \
  --epochs 200 \
  --group_num 512 \
  --alpha 0.01 \
  --gradient_adjust \
  --group_size 256
done

# relu+fr
for name in 'thai' 'arma' 'dragon' 'bun' 'lucy'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode relu+fr \
  --xuhao 8 \
  --replace_start 0 \
  --replace_end 8 \
  --random_index \
  --learning_rate 2e-3 \
  --epochs 200 \
  --group_num 512 \
  --alpha 0.01 \
  --group_size 256 \
  --saving
done


# relu+bn
# BN requires larger batch size (=group_num * group_size) and large learning rate.
for name in 'thai' 'arma' 'dragon' 'bun' 'lucy'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode relu+bn \
  --xuhao 6 \
  --replace_start 0 \
  --replace_end 6 \
  --random_index \
  --learning_rate 5e-3 \
  --epochs 200 \
  --group_num 100000 \
  --alpha 0.01 \
  --group_size 2
done

# for name in  'thai' 'arma' 'dragon' 'bun' 'lucy'
# do
# CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
#   --name ${name} \
#   --mode relu+bn \
#   --xuhao 6 \
#   --replace_start 0 \
#   --replace_end 6 \
#   --random_index \
#   --learning_rate 2e-3 \
#   --epochs 200 \
#   --group_num 100000 \
#   --alpha 0.01 \
#   --group_size 2
# done




