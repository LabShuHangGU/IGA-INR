export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2


GPU_ID=5

# PE
for name in  'thai' 'arma' 'dragon' 'bun' 'lucy'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode relu+pe \
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

# PE+IGA
for name in  'thai' 'arma' 'dragon'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode relu+pe \
  --xuhao 6 \
  --replace_start 0 \
  --replace_end 6 \
  --random_index \
  --learning_rate 2e-3 \
  --epochs 200 \
  --group_num 512 \
  --alpha 0.01 \
  --gradient_adjust \
  --group_size 256 \
  --saving
done

# PE+IGA
for name in  'bun'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode relu+pe \
  --xuhao 7 \
  --replace_start 0 \
  --replace_end 7 \
  --random_index \
  --learning_rate 2e-3 \
  --epochs 200 \
  --group_num 512 \
  --alpha 0.01 \
  --gradient_adjust \
  --group_size 256 \
  --saving
done

# PE+IGA
for name in  'lucy'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode relu+pe \
  --xuhao 13 \
  --replace_start 0 \
  --replace_end 13 \
  --random_index \
  --learning_rate 2e-3 \
  --epochs 200 \
  --group_num 512 \
  --alpha 0.01 \
  --gradient_adjust \
  --group_size 256 \
  --saving
done

#PE+FR
for name in 'thai' 'arma' 'dragon' 'bun' 'lucy'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode relu+pe+fr \
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

# PE+BN
for name in  'thai' 'arma' 'dragon' 'bun' 'lucy'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode relu+pe+bn \
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

# PE+BN
for name in  'thai' 'arma' 'dragon' 'bun' 'lucy'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode relu+pe+bn \
  --xuhao 6 \
  --replace_start 0 \
  --replace_end 6 \
  --random_index \
  --learning_rate 2e-3 \
  --epochs 200 \
  --group_num 100000 \
  --alpha 0.01 \
  --group_size 2
done

