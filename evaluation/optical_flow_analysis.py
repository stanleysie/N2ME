import os
import argparse
import cv2
import glob
import subprocess
import face_recognition
import numpy as np
import pandas as pd
from tqdm import tqdm

def handle_windows_path(path):
    drive = path.split(':')[0]
    path = path.replace('\\', '/')
    return path.replace(f'{drive}:/', f'/mnt/{drive.lower()}/')

project_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('--expr', type=str, help='experiment name')
parser.add_argument('--root_dir', type=str, help='root directory')
parser.add_argument('--subject', type=str, help='subject name')
parser.add_argument('--samm_dir', type=str, help='path to SAMM database directory')
parser.add_argument('--mmew_dir', type=str, help='path to MMEW database directory')
parser.add_argument('--casme_ii_dir', type=str, help='path to CASME II database directory')
args = parser.parse_args()
args.samm_dir = handle_windows_path(args.samm_dir)
args.mmew_dir = handle_windows_path(args.mmew_dir)
args.casme_ii_dir = handle_windows_path(args.casme_ii_dir)

def read_image(path):
    img = cv2.imread(path)
    return img

def generate_optical_strain(flow):
    u = flow[...,0]
    v = flow[...,1]

    ux, uy = np.gradient(u)
    vx, vy = np.gradient(v)

    e_xy = 0.5*(uy + vx)
    e_xx = ux
    e_yy = vy
    e_m = e_xx ** 2 + 2 * e_xy ** 2 + e_yy ** 2
    e_m = np.sqrt(e_m)
    e_m = cv2.normalize(e_m, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    e_m = e_m.astype(np.uint8)
    
    return e_m

def generate_optical_flow(onset, apex):   
    # compute optical flow (TVL1Flow)
    subprocess.run(f"tvl1flow/tvl1flow '{onset}' '{apex}'", shell=True)

    onset_img = read_image(onset)

    # creating mask based on the image
    hsv = np.zeros_like(onset_img)
    hsv[..., 1] = 255

    # reading optical flow
    flow = cv2.readOpticalFlow("flow.flo")

    # compute for the optical flow's magnitude and direction from the flow vector
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # generate optical strain
    optical_strain = generate_optical_strain(flow)

    if onset_img.shape[0] > 128 or onset_img.shape[1] > 128:
        face_locations = face_recognition.face_locations(onset_img)
        top, right, bottom, left = face_locations[0]
        rgb = rgb[top:bottom, left:right]
        optical_strain = optical_strain[top:bottom, left:right]
        rgb = cv2.resize(rgb, (128, 128))
        optical_strain = cv2.resize(optical_strain, (128, 128))

    return rgb, optical_strain

# directories
root_dir = handle_windows_path(args.root_dir)
emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
expr_name = args.expr
columns = []

# FACS Annotations
samm_facs = f"{args.samm_dir}/SAMM_FACS.csv"
mmew_facs = f"{args.mmew_dir}/MMEW_FACS.csv"
casme_ii_facs = f"{args.casme_ii_dir}/CASME_II_FACS.csv"
samm_facs = pd.read_csv(samm_facs)
mmew_facs = pd.read_csv(mmew_facs)
casme_ii_facs = pd.read_csv(casme_ii_facs)

# formatting columns
samm_facs['ref'] = samm_facs['Filename']
mmew_facs['ref'] = mmew_facs['Filename']
casme_ii_facs['Subject'] = casme_ii_facs['Subject'].apply(lambda x: f'{int(x):02d}')
casme_ii_facs['ref'] = 'sub' + casme_ii_facs['Subject'] + '_' + casme_ii_facs['Filename']

samm_facs = samm_facs.to_dict('records')
mmew_facs = mmew_facs.to_dict('records')
casme_ii_facs = casme_ii_facs.to_dict('records')

# ME Databases
samm_db = f"{args.samm_dir}/data"
mmew_db = f"{args.mmew_dir}/data"
casme_ii_db = f"{args.casme_ii_dir}/data"

# Generating optical flow and optical strain
for emo in emotions:
    data_dir = glob.glob(f"{root_dir}/{emo}/{expr_name}/**/{args.subject}")
    print('Emotion: ', emo)

    for dir in tqdm(data_dir):
        name = dir.split('/')[-2]
        ref = None
        imgs_dir = None
        
        if 's_' in name:
            ref = [r for r in samm_facs if r['ref'] == name[2:]][0]
            imgs_dir = f"{samm_db}/{int(ref['Subject']):03d}/{ref['Filename']}"
            imgs = glob.glob(f"{imgs_dir}/*.jpg")
            imgs = sorted(imgs, key=lambda x: int(x.split('/')[-1].split('_')[-1][:-4]))
        elif 'm_' in name:
            ref = [r for r in mmew_facs if r['ref'] == name[2:]][0]
            imgs_dir = f"{mmew_db}/{ref['Estimated Emotion']}/{ref['Filename']}"
            imgs = glob.glob(f"{imgs_dir}/*.jpg")
            imgs = sorted(imgs, key=lambda x: int(x.split('/')[-1][:-4]))
        elif 'c_' in name:
            ref = [r for r in casme_ii_facs if r['ref'] == name[2:]][0]
            imgs_dir = f"{casme_ii_db}/sub{int(ref['Subject']):02d}/{ref['Filename']}"
            imgs = glob.glob(f"{imgs_dir}/*.jpg")
            imgs = sorted(imgs, key=lambda x: int(x.split('/')[-1][3:-4]))

        apex = ref['ApexFrame'] - ref['OnsetFrame']
        # original images
        original_onset = imgs[0]
        original_apex = imgs[apex]
        
        # generated images
        imgs = glob.glob(f"{dir}/*.jpg")
        imgs = sorted(imgs, key=lambda x: int(x.split('/')[-1][:-4]))
        me_onset = imgs[0]
        me_apex = imgs[apex-1]

        save_dir = f"{dir}/evaluation/optical_flow_analysis"
        os.makedirs(save_dir, exist_ok=True)

        original_of, original_os = generate_optical_flow(original_onset, original_apex)
        generated_of, generated_os = generate_optical_flow(me_onset, me_apex)

        cv2.imwrite(f"{save_dir}/original_of.jpg", original_of)
        cv2.imwrite(f"{save_dir}/original_os.jpg", original_os)
        cv2.imwrite(f"{save_dir}/generated_of.jpg", generated_of)
        cv2.imwrite(f"{save_dir}/generated_os.jpg", generated_os)