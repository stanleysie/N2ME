import os
import argparse
import random
import glob
import json
from data_extractor import DataExtractor
from optical_flow import OpticalFlow
from tqdm import tqdm

random.seed(0)

def handle_windows_path(path):
    drive = path.split(':')[0]
    path = path.replace('\\', '/')
    return path.replace(f'{drive}:/', f'/mnt/{drive.lower()}/')

project_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('--expr', type=str, help='experiment name')
parser.add_argument('--data_dir', type=str, help='root directory')
parser.add_argument('--out_dir', type=str, help='output directory')
parser.add_argument('--n_samples', type=int, default=10, help='number of samples')
args = parser.parse_args()
args.data_dir = handle_windows_path(args.data_dir)
args.out_dir = handle_windows_path(args.out_dir)

emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
facs_dir = {
    "SAMM": handle_windows_path("path to SAMM FACS annotation (.csv)"),
    "MMEW": handle_windows_path("path to MMEW FACS annotation (.csv)"),
    "CASME_II": handle_windows_path("path to CASME II FACS annotation (.csv)")
}

def extract_onset_apex():
    extractor = DataExtractor(emotions, facs_dir)
    data_dict = {}

    for emo in tqdm(emotions):
        data = glob.glob(f'{args.data_dir}/{emo}/{args.expr}/**/**')
        data_dict[emo] = extractor.get_data(data, args.n_samples)

    with open(f'{args.out_dir}/onset_apex_synthetic.json', 'w') as f:
        json.dump(data_dict, f, indent=4)

def generate_features():
    optical_flow = OpticalFlow(emotions)

    with open(f'{args.out_dir}/onset_apex_synthetic.json', 'r') as f:
        data_dict = json.load(f)

    save_dir = args.out_dir + '/SYNTHETIC'
    os.makedirs(save_dir, exist_ok=True)

    saved_imgs = glob.glob(f'{save_dir}/*.jpg')

    for emo, data in data_dict.items():
        imgs = [img for img in saved_imgs if emo in img]
        if len(imgs) == args.n_samples * 4:
            continue

        start_index = len(imgs) // 4
        onset_data = data['onset']
        apex_data = data['apex']

        for i, (onset, apex) in enumerate(zip(onset_data, apex_data)):
            if i < start_index:
                continue

            subject = f"{'_'.join(onset[1].split('/')[-2].split('_')[:-1])}_{onset[0] or apex[0]}"
            onset = onset[1]
            apex = apex[1]

            print(f"{emo} - ({i+1}/{len(onset_data)})", end=' | ')
            optical_flow.generate_optical_flow(i+1, save_dir, emo, subject, onset, apex)

    dataset = optical_flow.get_dataset()
    with open(f'{args.out_dir}/dataset_synthetic.json', 'w') as f:
        json.dump(dataset, f, indent=4)
    
def generate_hybrid_dataset():
    with open(f'{args.out_dir}/dataset.json', 'r') as f:
        data = json.load(f)

    with open(f'{args.out_dir}/dataset_synthetic.json', 'r') as f:
        data_synthetic = json.load(f)

    X = []
    Y = []
    emo_count = {}

    for emo in emotions:
        ids = [i for i, img in enumerate(data['X']) if emo in img[0]]
        if len(ids) < args.n_samples:
            x = [data['X'][i] for i in ids]
            y = [data['y'][i] for i in ids]
            emo_count[emo] = len(ids)
        else:
            ids = random.sample(ids, args.n_samples)
            x = [data['X'][i] for i in ids]
            y = [data['y'][i] for i in ids]
            emo_count[emo] = args.n_samples

        X.extend(x)
        Y.extend(y)

    for emo in emotions:
        ids = [i for i, img in enumerate(data_synthetic['X']) if emo in img[0]]
        if emo_count[emo] < args.n_samples:
            ids = random.sample(ids, args.n_samples - emo_count[emo])
            x = [data_synthetic['X'][i] for i in ids]
            y = [data_synthetic['y'][i] for i in ids]
            emo_count[emo] += len(ids)
        
            X.extend(x)
            Y.extend(y)
    
    with open(f'{args.out_dir}/dataset_hybrid.json', 'w') as f:
        json.dump({
            'X': X,
            'y': Y
        }, f, indent=4)



def main():
    print('S3S-CNN Preparation')
    print('1. Extract onset and apex frames')
    print('2. Generate features')
    print('3. Generate hybrid dataset')
    option = int(input('Enter option: '))

    if option == 1:
        extract_onset_apex()
    elif option == 2:
        generate_features()
    elif option == 3:
        generate_hybrid_dataset()

if __name__ == '__main__':
    main()

