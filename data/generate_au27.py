import math
import glob
import pickle
import face_recognition
import numpy as np
from tqdm import tqdm

def handle_windows_path(path):
    drive = path.split(':')[0]
    path = path.replace('\\', '/')
    return path.replace(f'{drive}:/', f'/mnt/{drive.lower()}/')

def get_mouth_height(top_lip, bottom_lip):
    sum = 0
    for i in [8,9,10]:
        # distance between two near points up and down
        distance = math.sqrt( (top_lip[i][0] - bottom_lip[18-i][0])**2 + 
                              (top_lip[i][1] - bottom_lip[18-i][1])**2   )
        sum += distance
    return sum / 3

imgs_dir = handle_windows_path("path to CelebA images")
aus_dir = handle_windows_path("path to CelebA extracted aus")
aus_file = handle_windows_path("path to CelebA saved aus.pkl")
ids = glob.glob(f'{aus_dir}/*.csv')
ids = [a.split('/')[-1][:-4] for a in ids]
ids.sort()

with open(aus_file, 'rb') as f:
    aus = pickle.load(f)

mouth = {}
max_height = 0      # 33.015
min_height = 1000   # 0.0

for id in tqdm(ids):
    # mouth height
    img = face_recognition.load_image_file(f"{imgs_dir}/{id}.jpg")
    landmarks = face_recognition.face_landmarks(img)
    top_lip = landmarks[0]['top_lip']
    bottom_lip = landmarks[0]['bottom_lip']
    mouth_height = get_mouth_height(top_lip, bottom_lip)
    max_height = max(max_height, mouth_height)
    min_height = min(min_height, mouth_height)
    mouth[id] = mouth_height

print(f"Max height: {max_height}")
print(f"Min height: {min_height}")

# normalize mouth height
for id in tqdm(ids):
    mouth[id] = (mouth[id] - min_height) / (max_height - min_height)
    aus[id] = np.insert(aus[id], -1, mouth[id]*5.0)

# save labels
path = handle_windows_path("path to CelebA aus folder]")
with open(f"{path}/aus_mouth.pkl", 'wb') as f:
    pickle.dump(aus, f, pickle.HIGHEST_PROTOCOL)