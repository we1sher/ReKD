
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
run = print
if torch.cuda.is_available():
    run = os.system

data_path = 'path/to/imagenet/'

configs = [
        {'T': 'resnet34',
         'S': 'resnet18', 
         'A': [0.50], 'B': [20.0]},
        {'T': 'resnet50',
         'S': 'mobilenet', 
         'A': [0.05], 'B': [20.0]},
        ]

seeds = [68]

for config in configs:
    for alpha in config['A']:
        for beta in config['B']:
            for seed in seeds:
                orders = (f"python imagenet_amp.py --teacher {config['T']} --arch {config['S']} "
                          f"--re-pr-loss-weight {alpha} --re-kd-loss-weight {beta} --seed {seed} "
                          f"{data_path}")
                run(orders)
