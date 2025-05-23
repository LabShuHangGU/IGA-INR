export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

GPU_ID=2

# SIREN
for name in 'thai' 'arma' 'dragon' 'bun' 'lucy'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode sin \
  --xuhao 8 \
  --replace_start 0 \
  --replace_end 8 \
  --random_index \
  --learning_rate 5e-4 \
  --epochs 200 \
  --group_num 512 \
  --alpha 0.01 \
  --group_size 256 \
  --saving
done


# SIREN+IGA
for name in  'thai' 'arma' 
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode sin \
  --xuhao 13 \
  --replace_start 0 \
  --replace_end 13 \
  --random_index \
  --learning_rate 5e-4 \
  --epochs 200 \
  --group_num 512 \
  --alpha 0.01 \
  --gradient_adjust \
  --group_size 256 \
  --saving
done

# SIREN+IGA
for name in  'dragon'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode sin \
  --xuhao 11 \
  --replace_start 0 \
  --replace_end 11 \
  --random_index \
  --learning_rate 5e-4 \
  --epochs 200 \
  --group_num 512 \
  --alpha 0.01 \
  --gradient_adjust \
  --group_size 256 \
  --saving
done

# SIREN+IGA
for name in  'bun'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode sin \
  --xuhao 8 \
  --replace_start 0 \
  --replace_end 8 \
  --random_index \
  --learning_rate 5e-4 \
  --epochs 200 \
  --group_num 512 \
  --alpha 0.01 \
  --gradient_adjust \
  --group_size 256 \
  --saving
done

# SIREN+IGA
for name in  'lucy'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode sin \
  --xuhao 14 \
  --replace_start 0 \
  --replace_end 14 \
  --random_index \
  --learning_rate 5e-4 \
  --epochs 200 \
  --group_num 512 \
  --alpha 0.01 \
  --gradient_adjust \
  --group_size 256 \
  --saving
done

# SIREN+FR
for name in  'thai' 'arma' 'dragon' 'bun' 'lucy'
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python 3D_occupance.py \
  --name ${name} \
  --mode sin+fr \
  --xuhao 8 \
  --replace_start 0 \
  --replace_end 8 \
  --random_index \
  --learning_rate 5e-4 \
  --epochs 200 \
  --group_num 512 \
  --alpha 0.001 \
  --group_size 256 \
  --saving
done

