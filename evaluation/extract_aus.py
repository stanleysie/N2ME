import os
import argparse
import glob
import subprocess

def handle_windows_path(path):
    drive = path.split(':')[0]
    path = path.replace('\\', '/')
    return path.replace(f'{drive}:/', f'/mnt/{drive.lower()}/')

project_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('--expr', type=str, help='experiment name')
parser.add_argument('--root_dir', type=str, help='root directory')
parser.add_argument('--subject', type=str, help='subject name')
args = parser.parse_args()

# directories
root_dir = handle_windows_path(args.root_dir)
openface_dir = 'path to OpenFace /build/bin folder'
emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
expr_name = args.expr
subject = args.subject if args.subject != 'all' else '**'

for emo in emotions:
    data_dir = glob.glob(f"{root_dir}/{emo}/{expr_name}/**/{subject}")
    
    for dir in data_dir:
        name = dir.split('/')[-1]
        openface_command = f'{openface_dir}/FeatureExtraction -fdir {dir} -out_dir {dir} -of {name} -aus'
        subprocess.run(openface_command, shell=True)
        subprocess.run(f'rm {dir}/{name}_of_details.txt', shell=True)