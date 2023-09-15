# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# DSADS
# 100% training data
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'dsads' --target 0 --n_act_class 19 --n_aug_class 8 --auglossweight 1.0 --conweight 1.0 --dp 'dis' --dpweight 10.0 --n_feature 64 --remain_data_rate 1.0 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'dsads' --target 1 --n_act_class 19 --n_aug_class 8 --auglossweight 0.1 --conweight 0.1 --dp 'dis' --dpweight 10.0 --n_feature 64 --remain_data_rate 1.0 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'dsads' --target 2 --n_act_class 19 --n_aug_class 8 --auglossweight 0.01 --conweight 0.1 --dp 'dis' --dpweight 0.1 --n_feature 64 --remain_data_rate 1.0 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'dsads' --target 3 --n_act_class 19 --n_aug_class 8 --auglossweight 1.0 --conweight 5.0 --dp 'dis' --dpweight 1.0 --n_feature 64 --remain_data_rate 1.0 --save_path results/

# 20% training data
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'dsads' --target 0 --n_act_class 19 --n_aug_class 8 --auglossweight 1.0 --conweight 0.5 --dp 'dis' --dpweight 0.1 --n_feature 64 --remain_data_rate 0.2 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'dsads' --target 1 --n_act_class 19 --n_aug_class 8 --auglossweight 1.0 --conweight 2.0 --dp 'dis' --dpweight 10.0 --n_feature 64 --remain_data_rate 0.2 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'dsads' --target 2 --n_act_class 19 --n_aug_class 8 --auglossweight 0.1 --conweight 0.2 --dp 'dis' --dpweight 0.1 --n_feature 64 --remain_data_rate 0.2 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'dsads' --target 3 --n_act_class 19 --n_aug_class 8 --auglossweight 0.1 --conweight 1.0 --dp 'dis' --dpweight 0.01 --n_feature 64 --remain_data_rate 0.2 --save_path results/


# PAMAP2
# 100% training data
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'pamap' --target 0 --n_act_class 8 --n_aug_class 8 --auglossweight 10.0 --conweight 10.0 --dp 'dis' --dpweight 0.1 --n_feature 64 --remain_data_rate 1.0 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'pamap' --target 1 --n_act_class 8 --n_aug_class 8 --auglossweight 0.01 --conweight 2.0 --dp 'dis' --dpweight 1.0 --n_feature 64 --remain_data_rate 1.0 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'pamap' --target 2 --n_act_class 8 --n_aug_class 8 --auglossweight 1.0 --conweight 10.0 --dp 'dis' --dpweight 1.0 --n_feature 64 --remain_data_rate 1.0 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'pamap' --target 3 --n_act_class 8 --n_aug_class 8 --auglossweight 10.0 --conweight 0.2 --dp 'dis' --dpweight 10.0 --n_feature 64 --remain_data_rate 1.0 --save_path results/

# 20% training data
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'pamap' --target 0 --n_act_class 8 --n_aug_class 8 --auglossweight 10.0 --conweight 2.0 --dp 'dis' --dpweight 0.1 --n_feature 64 --remain_data_rate 0.2 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'pamap' --target 1 --n_act_class 8 --n_aug_class 8 --auglossweight 0.01 --conweight 0.4 --dp 'dis' --dpweight 1.0 --n_feature 64 --remain_data_rate 0.2 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'pamap' --target 2 --n_act_class 8 --n_aug_class 8 --auglossweight 0.01 --conweight 1.0 --dp 'dis' --dpweight 10.0 --n_feature 64 --remain_data_rate 0.2 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'pamap' --target 3 --n_act_class 8 --n_aug_class 8 --auglossweight 1.0 --conweight 2.0 --dp 'dis' --dpweight 0.1 --n_feature 64 --remain_data_rate 0.2 --save_path results/


# USC-HAD
# 100% training data
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'uschad' --target 0 --n_act_class 12 --n_aug_class 8 --auglossweight 0.1 --conweight 10.0 --dp 'dis' --dpweight 0.8 --n_feature 128 --remain_data_rate 1.0 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'uschad' --target 1 --n_act_class 12 --n_aug_class 8 --auglossweight 0.1 --conweight 10.0 --dp 'dis' --dpweight 4.0 --n_feature 128 --remain_data_rate 1.0 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'uschad' --target 2 --n_act_class 12 --n_aug_class 8 --auglossweight 0.1 --conweight 10.0 --dp 'dis' --dpweight 10.0 --n_feature 128 --remain_data_rate 1.0 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'uschad' --target 3 --n_act_class 12 --n_aug_class 8 --auglossweight 0.01 --conweight 2.0 --dp 'dis' --dpweight 2.0 --n_feature 128 --remain_data_rate 1.0 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'uschad' --target 4 --n_act_class 12 --n_aug_class 8 --auglossweight 10.0 --conweight 1.0 --dp 'dis' --dpweight 0.3 --n_feature 128 --remain_data_rate 1.0 --save_path results/

# 20% training data
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'uschad' --target 0 --n_act_class 12 --n_aug_class 8 --auglossweight 0.01 --conweight 1.0 --dp 'dis' --dpweight 1.0 --n_feature 128 --remain_data_rate 0.2 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'uschad' --target 1 --n_act_class 12 --n_aug_class 8 --auglossweight 0.1 --conweight 2.0 --dp 'dis' --dpweight 1.0 --n_feature 128 --remain_data_rate 0.2 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'uschad' --target 2 --n_act_class 12 --n_aug_class 8 --auglossweight 1.0 --conweight 0.3 --dp 'dis' --dpweight 10.0 --n_feature 128 --remain_data_rate 0.2 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'uschad' --target 3 --n_act_class 12 --n_aug_class 8 --auglossweight 0.1 --conweight 2.0 --dp 'dis' --dpweight 1.0 --n_feature 128 --remain_data_rate 0.2 --save_path results/
python main.py --num_workers 1 --root_path /home/qinxin/data/process_0702/process/ --seed 1 --dataset 'uschad' --target 4 --n_act_class 12 --n_aug_class 8 --auglossweight 0.01 --conweight 1.0 --dp 'dis' --dpweight 10.0 --n_feature 128 --remain_data_rate 0.2 --save_path results/
