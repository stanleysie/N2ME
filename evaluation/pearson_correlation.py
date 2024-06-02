import os
import argparse
import math
import glob
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings('error')

def handle_windows_path(path):
    drive = path.split(':')[0]
    path = path.replace('\\', '/')
    return path.replace(f'{drive}:/', f'/mnt/{drive.lower()}/')

project_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('--expr', type=str, help='experiment name')
parser.add_argument('--root_dir', type=str, help='root directory')
parser.add_argument('--subject', type=str, help='subject name')
parser.add_argument('--samm_aus', type=str, help='path to directory containing aus file from SAMM database')
parser.add_argument('--mmew_aus', type=str, help='path to directory containing aus file from MMEW database')
parser.add_argument('--casme_ii_aus', type=str, help='path to directory containing aus file from CASME II database')
parser.add_argument('--use_zero', type=int, default=0, help='use aus with zero values')
args = parser.parse_args()
args.use_zero = bool(args.use_zero)

def z_to_r(z):
    return (math.exp(2*z)-1)/(math.exp(2*z)+1)

# directories
samm_dir = handle_windows_path(args.samm_aus)
mmew_dir = handle_windows_path(args.mmew_aus)
casme_ii_dir = handle_windows_path(args.casme_ii_aus)
root_dir = handle_windows_path(args.root_dir)

emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
expr_name = args.expr
subject = args.subject if args.subject != 'all' else '**'
save_dir = f"{root_dir}/pearson_corr/{expr_name}"
os.makedirs(save_dir, exist_ok=True)
columns = []
aus_z_scores = {}

for emo in tqdm(emotions):
    data_dir = glob.glob(f"{root_dir}/{emo}/{expr_name}/**/{subject}")

    for dir in data_dir:
        subject_name = dir.split('/')[-1]
        name = dir.split('/')[-2]
        
        db_dir = samm_dir
        if 'm_' in name:
            db_dir = mmew_dir
        elif 'c_' in name:
            db_dir = casme_ii_dir
            
        original = pd.read_csv(f'{db_dir}/{"_".join(name.split("_")[1:])}.csv')
        generated = pd.read_csv(f'{dir}/{subject_name}.csv')

        if len(columns) == 0:
            columns = [col for col in original.columns if '_r' in col]
            aus_z_scores['skipped'] = []
        
        original = original[columns]
        generated = generated[columns]

        for col in columns:
            if not col in aus_z_scores:
                aus_z_scores[col] = []

            real = list(original[col])
            gen = list(generated[col])

            if args.use_zero:
                if sum(real) == 0 and sum(gen) == 0:
                    corr = 0.99
                elif sum(real) == 0 or sum(gen) == 0:
                    aus_z_scores['skipped'].append((name, col))
                    continue
                else:
                    corr = np.corrcoef(real, gen)[0, 1]
            else:
                if sum(real) == 0 or sum(gen) == 0:
                    aus_z_scores['skipped'].append((name, col))
                    continue
                else:
                    corr = np.corrcoef(real, gen)[0, 1]

            if corr == 1:
                corr = 0.99
            z_score = np.log((1 + corr) / (1 - corr)) / 2
            aus_z_scores[col].append(z_score)

name = 'z_score_v2' if args.use_zero else 'z_score'
with open(f'{save_dir}/{name}.json', 'w') as j:
    json.dump(aus_z_scores, j, indent=4)

name = 'pearson_v2' if args.use_zero else 'pearson'
with open(f'{save_dir}/{name}.txt', 'w') as f:
    for key, values in aus_z_scores.items():
        if key == 'skipped':
            continue
        f.write(f"{key} - {z_to_r(np.mean(values))}\n")