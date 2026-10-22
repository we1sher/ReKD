
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
run = print
if torch.cuda.is_available():
    run = os.system

data_path = 'path/to/your/data/'
teacher_ckpts_path = 'path/to/your/teacher/checkpoints/'

configs = [
        {'T': 'resnet32x4',
         'S': 'resnet8x4', 
         'A': [0.05], 'B': [50.0]},
        {'T': 'vgg13',
         'S': 'vgg8',
         'A': [0.05], 'B': [50.0]},
        {'T': 'wrn_40_2',           
         'S': 'wrn_16_2',
         'A': [0.05], 'B': [30.0]},
        {'T': 'resnet56',
         'S': 'resnet20',
         'A': [0.50], 'B': [15.0]},
        {'T': 'resnet110',
         'S': 'resnet32', 
         'A': [0.50], 'B': [20.0]},
        {'T': 'vgg13',
         'S': 'mobilenetv2',
         'A': [0.01], 'B': [50.0]},
        {'T': 'resnet32x4',
         'S': 'shufflenetv1',
         'A': [0.01], 'B': [50.0]},
        {'T': 'wrn_40_2',
         'S': 'shufflenetv1',
         'A': [0.01], 'B': [50.0]},
        {'T': 'resnet50',
         'S': 'mobilenetv2',
         'A': [0.01], 'B': [50.0]},
        {'T': 'resnet32x4',
         'S': 'shufflenetv2',
         'A': [0.01], 'B': [50.0]},
        ]

seeds = [68]

for config in configs:
    for alpha in config['A']:
        for beta in config['B']:
            for seed in seeds:
                orders = (f"python train.py --teacher {config['T']} --model {config['S']} "
                          f"--pr-loss-weight {alpha} --kd-loss-weight {beta} --seed {seed} "
                          f"--data_path {data_path} --teacher-weight {teacher_ckpts_path}")
                run(orders)
