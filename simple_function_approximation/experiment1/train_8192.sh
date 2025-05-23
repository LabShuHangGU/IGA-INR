export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2


CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name sgd_8192_baseline \
    --k_of_target 0.2 0.4 0.8 1.6  \
    --width 8192 \
    --optimizer sgd \
    --epochs 20000 \
    --lr 1e-1 \
    --estimation 'ntk' \
    --xuhao 0 \
    --replace_start 0 \
    --replace_end 10 \
    --i_freq 100 \
    --IGA 8 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name sgd_8192_K_10 \
    --k_of_target 0.2 0.4 0.8 1.6  \
    --width 8192 \
    --optimizer sgd \
    --epochs 20000 \
    --lr 1e-1 \
    --estimation 'ntk' \
    --enable-adjust \
    --xuhao 0 \
    --replace_start 0 \
    --replace_end 10 \
    --i_freq 100 \
    --IGA 8 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name sgd_8192_K_12 \
    --k_of_target 0.2 0.4 0.8 1.6  \
    --width 8192 \
    --optimizer sgd \
    --epochs 20000 \
    --lr 1e-1 \
    --estimation 'ntk' \
    --enable-adjust \
    --xuhao 0 \
    --replace_start 0 \
    --replace_end 12 \
    --i_freq 100 \
    --IGA 8 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name sgd_8192_K_14 \
    --k_of_target 0.2 0.4 0.8 1.6  \
    --width 8192 \
    --optimizer sgd \
    --epochs 20000 \
    --lr 1e-1 \
    --estimation 'ntk' \
    --enable-adjust \
    --xuhao 0 \
    --replace_start 0 \
    --replace_end 14 \
    --i_freq 100 \
    --IGA 8 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name sgd_8192_Kt_10 \
    --k_of_target 0.2 0.4 0.8 1.6  \
    --width 8192 \
    --optimizer sgd \
    --epochs 20000 \
    --lr 1e-1 \
    --estimation 'entk' \
    --enable-adjust \
    --xuhao 0 \
    --replace_start 0 \
    --replace_end 10 \
    --i_freq 100 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name sgd_8192_Kt_12 \
    --k_of_target 0.2 0.4 0.8 1.6  \
    --width 8192 \
    --optimizer sgd \
    --epochs 20000 \
    --lr 1e-1 \
    --estimation 'entk' \
    --enable-adjust \
    --xuhao 0 \
    --replace_start 0 \
    --replace_end 12 \
    --i_freq 100 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name sgd_8192_Kt_14 \
    --k_of_target 0.2 0.4 0.8 1.6  \
    --width 8192 \
    --optimizer sgd \
    --epochs 20000 \
    --lr 1e-1 \
    --estimation 'entk' \
    --enable-adjust \
    --xuhao 0 \
    --replace_start 0 \
    --replace_end 14 \
    --i_freq 100 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name sgd_8192_Kt_10_IGA_8 \
    --k_of_target 0.2 0.4 0.8 1.6  \
    --width 8192 \
    --optimizer sgd \
    --epochs 20000 \
    --lr 1e-1 \
    --estimation 'entk' \
    --enable-adjust \
    --enable-IGA \
    --IGA 8 \
    --xuhao 0 \
    --replace_start 0 \
    --replace_end 10 \
    --i_freq 100 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name sgd_8192_Kt_12_IGA_8 \
    --k_of_target 0.2 0.4 0.8 1.6  \
    --width 8192 \
    --optimizer sgd \
    --epochs 20000 \
    --lr 1e-1 \
    --estimation 'entk' \
    --enable-adjust \
    --enable-IGA \
    --IGA 8 \
    --xuhao 0 \
    --replace_start 0 \
    --replace_end 12 \
    --i_freq 100 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name sgd_8192_Kt_14_IGA_8 \
    --k_of_target 0.2 0.4 0.8 1.6  \
    --width 8192 \
    --optimizer sgd \
    --epochs 20000 \
    --lr 1e-1 \
    --estimation 'entk' \
    --enable-adjust \
    --enable-IGA \
    --IGA 8 \
    --xuhao 0 \
    --replace_start 0 \
    --replace_end 14 \
    --i_freq 100