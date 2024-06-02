import os
import math
import glob
import subprocess
import multiprocessing
import cv2
import face_recognition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def handle_windows_path(path):
    drive = path.split(':')[0]
    path = path.replace('\\', '/')
    return path.replace(f'{drive}:/', f'/mnt/{drive.lower()}/')

# ====================================================================================================
# 1. Extraction of information and facial cropping and aligning with OpenFace
#    Information:
#       - confidence level
#       - aus
# ====================================================================================================
# imgs_dir = handle_windows_path("path to CelebA images")
# imgs_save_dir = handle_windows_path("path to preprocess_1 folder")
# aus_save_dir = handle_windows_path("path to aus_1 folder")
# openface_dir = 'path to OpenFace /build/bin folder'

# data = glob.glob(f"{imgs_dir}/*.jpg")
# data = [f.split('/')[-1].split('.')[0] for f in data]
# aus_files = glob.glob(f"{aus_save_dir}/*.csv")
# aus_files = [f.split('/')[-1].split('.')[0] for f in aus_files]
# data = list(set(data) - set(aus_files))

# def openface_command(image, out_dir, name):
#     command = f'{openface_dir}/FaceLandmarkImg -f {image} -out_dir {out_dir} -of {name} -nomask -nobadaligned -simsize 128 -format_aligned jpg -wild'
#     return command

# def extract_info(name):
#     img_path = f'{imgs_dir}/{name}.jpg'
#     command = openface_command(img_path, aus_save_dir, name)
#     subprocess.run(command, shell=True)

#     subprocess.run(f'rm {aus_save_dir}/{name}.jpg', shell=True)
#     if os.path.exists(f'{aus_save_dir}/{name}_aligned/face_det_000000.jpg'):
#         subprocess.run(f'mv {aus_save_dir}/{name}_aligned/face_det_000000.jpg {imgs_save_dir}/{name}.jpg', shell=True)
#     subprocess.run(f'rm {aus_save_dir}/{name}.hog', shell=True)
#     subprocess.run(f'rm {aus_save_dir}/{name}_of_details.txt', shell=True)
#     subprocess.run(f'rm -rf {aus_save_dir}/{name}_aligned', shell=True)

# def extract_info_parallel(imgs):
#     pool = multiprocessing.Pool(processes=8)
#     pool.map(extract_info, imgs)
#     pool.close()
#     pool.join()

# extract_info_parallel(data)

# ====================================================================================================
# 2. Face detection and filtering out non-frontal images
#
#    Calculating distance between:
#       - D1 - nose tip and chin
#       - D2 - corner left eye and nose tip
#       - D3 - corner right eye and nose tip
#    -> D2 and D3 should have a relatively small difference
#
#    Max difference between D2 and D3: 20
#    Max difference between x coordinates of nose tip and chin: 6
# ====================================================================================================
# raw_imgs_dir = handle_windows_path("path to CelebA images")
# imgs_dir = handle_windows_path("path to preprocess_1 folder")
# aus_dir = handle_windows_path("path to aus_1 folder")
# imgs_save_dir = handle_windows_path("path to preprocess_2 folder")
# aus_save_dir = handle_windows_path("path to aus_2 folder")

# imgs = glob.glob(f'{imgs_dir}/*.jpg')
# imgs = [i.split('/')[-1].split('.')[0] for i in imgs]
# aus = glob.glob(f'{aus_dir}/*.csv')
# aus = [i.split('/')[-1].split('.')[0] for i in aus]

# save_files = glob.glob(f'{aus_save_dir}/*.csv')
# save_files = [i.split('/')[-1].split('.')[0] for i in save_files]

# ids = list(set(imgs).intersection(set(aus)) - set(save_files))
# ids.sort()

# # calculate distance between two points
# def distance(p1, p2):
#     return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# for id in tqdm(ids):
#     img = face_recognition.load_image_file(f'{raw_imgs_dir}/{id}.jpg')
#     landmarks = face_recognition.face_landmarks(img)

#     if len(landmarks) == 0:
#         continue
    
#     landmarks = landmarks[0]
#     nose_tip = landmarks['nose_bridge'][-1]
#     chin = landmarks['chin'][8]
#     left_eye = landmarks['right_eye'][3]
#     right_eye = landmarks['left_eye'][0]

#     right_eye_nose = distance(right_eye, nose_tip)
#     left_eye_nose = distance(left_eye, nose_tip)
#     eye_nose = abs(right_eye_nose - left_eye_nose)
#     nose_chin = abs(nose_tip[0] - chin[0])

#     if eye_nose <= 20:
#         if nose_chin <= 6:
#             subprocess.run(f'mv {imgs_dir}/{id}.jpg {imgs_save_dir}/{id}.jpg', shell=True)
#             subprocess.run(f'mv {aus_dir}/{id}.csv {aus_save_dir}/{id}.csv', shell=True)

# ====================================================================================================
# 3. Filtering out images based on confidence level is at least 0.75
# ====================================================================================================
# imgs_dir = handle_windows_path("path to preprocess_2 folder")
# aus_dir = handle_windows_path("path to aus_2 folder")
# imgs_save_dir = handle_windows_path("path to preprocess_3 folder")
# aus_save_dir = handle_windows_path("path to aus_3 folder")

# imgs = glob.glob(f'{imgs_dir}/*.jpg')
# imgs = [i.split('/')[-1].split('.')[0] for i in imgs]
# aus = glob.glob(f'{aus_dir}/*.csv')
# aus = [i.split('/')[-1].split('.')[0] for i in aus]

# save_files = glob.glob(f'{aus_save_dir}/*.csv')
# save_files = [i.split('/')[-1].split('.')[0] for i in save_files]

# ids = list(set(imgs).intersection(set(aus)) - set(save_files))
# ids.sort()

# for id in tqdm(ids):
#     df = pd.read_csv(f'{aus_dir}/{id}.csv')
#     c = df['confidence'].mean()
#     if c >= 0.75:
#         subprocess.run(f'mv {imgs_dir}/{id}.jpg {imgs_save_dir}/{id}.jpg', shell=True)
#         subprocess.run(f'mv {aus_dir}/{id}.csv {aus_save_dir}/{id}.csv', shell=True)

# ====================================================================================================
# 4. Filtering of the images based on the blurriness level
#
#    Blurriness level is calculated using the Laplacian variance
#    -> the higher the value, the less blurry the image is
#    -> threshold: 70
#    -> https://theailearner.com/2021/10/30/blur-detection-using-the-variance-of-the-laplacian-method/
# ====================================================================================================
# imgs_dir = handle_windows_path("path to preprocess_3 folder")
# aus_dir = handle_windows_path("path to aus_3 folder")
# imgs_save_dir = handle_windows_path("path to preprocess_final folder")
# aus_save_dir = handle_windows_path("path to aus_final folder")

# imgs = glob.glob(f'{imgs_dir}/*.jpg')
# imgs = [i.split('/')[-1].split('.')[0] for i in imgs]
# aus = glob.glob(f'{aus_dir}/*.csv')
# aus = [i.split('/')[-1].split('.')[0] for i in aus]

# save_files = glob.glob(f'{aus_save_dir}/*.csv')
# save_files = [i.split('/')[-1].split('.')[0] for i in save_files]

# ids = list(set(imgs).intersection(set(aus)) - set(save_files))
# ids.sort()

# threshold = 70
# for id in tqdm(ids):
#     img = cv2.imread(f'{imgs_dir}/{id}.jpg')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     laplace = cv2.Laplacian(gray, cv2.CV_64F).var()
#     if laplace >= threshold:
#         subprocess.run(f'mv {imgs_dir}/{id}.jpg {imgs_save_dir}/{id}.jpg', shell=True)
#         subprocess.run(f'mv {aus_dir}/{id}.csv {aus_save_dir}/{id}.csv', shell=True)

# ====================================================================================================
# 5. Generate train and test ids
# ====================================================================================================
# save_dir = handle_windows_path("path to preprocess folder")
# imgs_dir = handle_windows_path("path to preprocess_final folder")
# imgs = glob.glob(f'{imgs_dir}/*.jpg')
# imgs = [i.split('/')[-1] for i in imgs]
# imgs.sort()

# ratio = 0.9
# train_ids = imgs[:int(len(imgs)*ratio)]
# test_ids = imgs[int(len(imgs)*ratio):]

# with open(f'{save_dir}/train_ids.txt', 'w') as f:
#     f.write('\n'.join(train_ids))

# with open(f'{save_dir}/test_ids.txt', 'w') as f:
#     f.write('\n'.join(test_ids))