export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

CUDA_VISIBLE_DEVICES=5 python exp.py \
    --name relu_6_frequency_baseline \
    --k_of_target 5 10 15 20 25 30 \
    --optimizer adam \
    --mode relu \
    --in_features 1 \
    --hidden_features 256 \
    --hidden_layers 3 \
    --out_features 1 \
    --epochs 25000 \
    --lr 5e-5 \
    --N 2048 \
    --xuhao 1 \
    --replace_start 0 \
    --replace_end 1 \
    --min_val -1 \
    --max_val 1 \
    --i_freq 1 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name relu_6_frequency_balance_2_IGA_1 \
    --k_of_target 5 10 15 20 25 30 \
    --optimizer adam \
    --mode relu \
    --in_features 1 \
    --hidden_features 256 \
    --hidden_layers 3 \
    --out_features 1 \
    --epochs 25000 \
    --lr 5e-5 \
    --N 2048 \
    --enable_IGA \
    --IGA 1 \
    --gradient_adjust \
    --xuhao 1 \
    --replace_start 0 \
    --replace_end 1 \
    --min_val -1 \
    --max_val 1 \
    --i_freq 1 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name relu_6_frequency_balance_4_IGA_1 \
    --k_of_target 5 10 15 20 25 30 \
    --optimizer adam \
    --mode relu \
    --in_features 1 \
    --hidden_features 256 \
    --hidden_layers 3 \
    --out_features 1 \
    --epochs 25000 \
    --lr 5e-5 \
    --N 2048 \
    --enable_IGA \
    --IGA 1 \
    --gradient_adjust \
    --xuhao 3 \
    --replace_start 0 \
    --replace_end 3 \
    --min_val -1 \
    --max_val 1 \
    --i_freq 1 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name relu_6_frequency_balance_6_IGA_1 \
    --k_of_target 5 10 15 20 25 30 \
    --optimizer adam \
    --mode relu \
    --in_features 1 \
    --hidden_features 256 \
    --hidden_layers 3 \
    --out_features 1 \
    --epochs 25000 \
    --lr 5e-5 \
    --N 2048 \
    --enable_IGA \
    --IGA 1 \
    --gradient_adjust \
    --xuhao 5 \
    --replace_start 0 \
    --replace_end 5 \
    --min_val -1 \
    --max_val 1 \
    --i_freq 1 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name relu_6_frequency_balance_8_IGA_1 \
    --k_of_target 5 10 15 20 25 30 \
    --optimizer adam \
    --mode relu \
    --in_features 1 \
    --hidden_features 256 \
    --hidden_layers 3 \
    --out_features 1 \
    --epochs 25000 \
    --lr 5e-5 \
    --N 2048 \
    --enable_IGA \
    --IGA 1 \
    --gradient_adjust \
    --xuhao 7 \
    --replace_start 0 \
    --replace_end 7 \
    --min_val -1 \
    --max_val 1 \
    --i_freq 1 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name relu_6_frequency_balance_4_IGA_4 \
    --k_of_target 5 10 15 20 25 30 \
    --optimizer adam \
    --mode relu \
    --in_features 1 \
    --hidden_features 256 \
    --hidden_layers 3 \
    --out_features 1 \
    --epochs 25000 \
    --lr 5e-5 \
    --N 2048 \
    --enable_IGA \
    --IGA 4 \
    --gradient_adjust \
    --xuhao 3 \
    --replace_start 0 \
    --replace_end 3 \
    --min_val -1 \
    --max_val 1 \
    --i_freq 1 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name relu_6_frequency_balance_2_IGA_4 \
    --k_of_target 5 10 15 20 25 30 \
    --optimizer adam \
    --mode relu \
    --in_features 1 \
    --hidden_features 256 \
    --hidden_layers 3 \
    --out_features 1 \
    --epochs 25000 \
    --lr 5e-5 \
    --N 2048 \
    --enable_IGA \
    --IGA 4 \
    --gradient_adjust \
    --xuhao 1 \
    --replace_start 0 \
    --replace_end 1 \
    --min_val -1 \
    --max_val 1 \
    --i_freq 1 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name relu_6_frequency_balance_4_IGA_4 \
    --k_of_target 5 10 15 20 25 30 \
    --optimizer adam \
    --mode relu \
    --in_features 1 \
    --hidden_features 256 \
    --hidden_layers 3 \
    --out_features 1 \
    --epochs 25000 \
    --lr 5e-5 \
    --N 2048 \
    --enable_IGA \
    --IGA 4 \
    --gradient_adjust \
    --xuhao 3 \
    --replace_start 0 \
    --replace_end 3 \
    --min_val -1 \
    --max_val 1 \
    --i_freq 1 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name relu_6_frequency_balance_6_IGA_4 \
    --k_of_target 5 10 15 20 25 30 \
    --optimizer adam \
    --mode relu \
    --in_features 1 \
    --hidden_features 256 \
    --hidden_layers 3 \
    --out_features 1 \
    --epochs 25000 \
    --lr 5e-5 \
    --N 2048 \
    --enable_IGA \
    --IGA 4 \
    --gradient_adjust \
    --xuhao 5 \
    --replace_start 0 \
    --replace_end 5 \
    --min_val -1 \
    --max_val 1 \
    --i_freq 1 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name relu_6_frequency_balance_8_IGA_4 \
    --k_of_target 5 10 15 20 25 30 \
    --optimizer adam \
    --mode relu \
    --in_features 1 \
    --hidden_features 256 \
    --hidden_layers 3 \
    --out_features 1 \
    --epochs 25000 \
    --lr 5e-5 \
    --N 2048 \
    --enable_IGA \
    --IGA 4 \
    --gradient_adjust \
    --xuhao 7 \
    --replace_start 0 \
    --replace_end 7 \
    --min_val -1 \
    --max_val 1 \
    --i_freq 1 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name relu_6_frequency_balance_2_IGA_8 \
    --k_of_target 5 10 15 20 25 30 \
    --optimizer adam \
    --mode relu \
    --in_features 1 \
    --hidden_features 256 \
    --hidden_layers 3 \
    --out_features 1 \
    --epochs 25000 \
    --lr 5e-5 \
    --N 2048 \
    --enable_IGA \
    --IGA 8 \
    --gradient_adjust \
    --xuhao 1 \
    --replace_start 0 \
    --replace_end 1 \
    --min_val -1 \
    --max_val 1 \
    --i_freq 1 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name relu_6_frequency_balance_4_IGA_8 \
    --k_of_target 5 10 15 20 25 30 \
    --optimizer adam \
    --mode relu \
    --in_features 1 \
    --hidden_features 256 \
    --hidden_layers 3 \
    --out_features 1 \
    --epochs 25000 \
    --lr 5e-5 \
    --N 2048 \
    --enable_IGA \
    --IGA 8 \
    --gradient_adjust \
    --xuhao 3 \
    --replace_start 0 \
    --replace_end 3 \
    --min_val -1 \
    --max_val 1 \
    --i_freq 1 \

CUDA_VISIBLE_DEVICES=9 python exp.py \
    --name relu_6_frequency_balance_6_IGA_8 \
    --k_of_target 5 10 15 20 25 30 \
    --optimizer adam \
    --mode relu \
    --in_features 1 \
    --hidden_features 256 \
    --hidden_layers 3 \
    --out_features 1 \
    --epochs 25000 \
    --lr 5e-5 \
    --N 2048 \
    --enable_IGA \
    --IGA 8 \
    --gradient_adjust \
    --xuhao 5 \
    --replace_start 0 \
    --replace_end 5 \
    --min_val -1 \
    --max_val 1 \
    --i_freq 1 \

CUDA_VISIBLE_DEVICES=5 python exp.py \
    --name relu_6_frequency_balance_8_IGA_8 \
    --k_of_target 5 10 15 20 25 30 \
    --optimizer adam \
    --mode relu \
    --in_features 1 \
    --hidden_features 256 \
    --hidden_layers 3 \
    --out_features 1 \
    --epochs 25000 \
    --lr 5e-5 \
    --N 2048 \
    --enable_IGA \
    --IGA 8 \
    --gradient_adjust \
    --xuhao 7 \
    --replace_start 0 \
    --replace_end 7 \
    --min_val -1 \
    --max_val 1 \
    --i_freq 1