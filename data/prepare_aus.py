import os
import argparse
import glob
import pickle
import pandas as pd
from tqdm import tqdm

def handle_windows_path(path):
    drive = path.split(':')[0]
    path = path.replace('\\', '/')
    return path.replace(f'{drive}:/', f'/mnt/{drive.lower()}/')

project_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('--aus_dir', type=str, default='aus', help='directory to folder containing extracted action units')
parser.add_argument('--output_dir', type=str, help='output directory')
args = parser.parse_args()

args.aus_dir = handle_windows_path(args.aus_dir)
args.output_dir = handle_windows_path(args.output_dir)

def get_data(paths):
    data = {}
    columns = []
    for filename in tqdm(paths):
        content = pd.read_csv(filename)

        if len(columns) == 0:
            # extract only columns for AUs
            columns = [col for col in content.columns if col.startswith('AU') and '_r' in col]
            
        content = content[columns].to_numpy()
        data[os.path.basename(filename[:-4])] = content[0]
    
    return data


def save_dict(data, name):
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def main():
    filepaths = glob.glob(f'{args.aus_dir}/*.csv')
    filepaths.sort()

    data = get_data(filepaths)
    save_dict(data, f'{args.output_dir}/aus')


if __name__ == '__main__':
    main()