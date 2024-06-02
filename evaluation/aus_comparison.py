import os
import argparse
import glob
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.signal import savgol_filter

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
args = parser.parse_args()

# directories
samm_dir = handle_windows_path(args.samm_aus)
mmew_dir = handle_windows_path(args.mmew_aus)
casme_ii_dir = handle_windows_path(args.casme_ii_aus)
root_dir = handle_windows_path(args.root_dir)
openface_dir = 'path to OpenFace /build/bin folder'

emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
expr_name = args.expr
columns = []

for emo in emotions:
    data_dir = glob.glob(f"{root_dir}/{emo}/{expr_name}/**/{args.subject}")
    print('Emotion: ', emo)

    for dir in tqdm(data_dir):
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
        
        original = original[columns]
        generated = generated[columns]

        save_dir = f"{dir}/evaluation/au_analysis"
        os.makedirs(save_dir, exist_ok=True)

        for col in columns:
            X = range(len(original))
            plt.plot(X, savgol_filter(original[col], 5, 3), label="Original")
            plt.plot(X, savgol_filter(generated[col], 5, 3), label="Generated")
            plt.title(col)
            plt.legend()
            plt.grid()
            plt.savefig(f"{save_dir}/{col}.png")
            plt.close()