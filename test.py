import os
import cv2
import glob
import json
import random
import math
import subprocess
import torch
import face_recognition
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from models.model_factory import build_model
from evaluation.preparation import prepare_me_summary
from PIL import Image
from tqdm import tqdm

from utils.opencv import read_img, save_img
from config import Config

class Test(Config):
    def __init__(self):
        super().__init__(mode="test")

        self.test_mode = self.config.test_mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.neutral_G = None
        self.me_G = None
        self.neutral_c_dim = self.config.c_dim + 1 if self.config.use_18_aus else self.config.c_dim
        self.me_c_dim = self.config.c_dim
        self.emotions_list = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
        self.info = {}

        # build models
        ckpts_dir = self.config.ckpts_dir
        expr_name = self.config.expr_name
        self.config.ckpts_dir = '/'.join(self.config.ckpts_dir.split('/')[:-1]) + '/neutral'
        self.config.expr_name = self.config.expr_name if self.config.test_mode == 'neutral' else self.config.neutral_expr_name
        self.config.model_name = self.config.neutral_model
        self.config.c_dim = self.neutral_c_dim
        self.neutral_G = build_model(self.config)
        self.neutral_G.set_eval()
        
        if self.test_mode in ['me', 'all']:
            self.config.ckpts_dir = ckpts_dir
            self.config.expr_name = expr_name
            self.config.model_name = self.config.me_model
            self.config.c_dim = self.me_c_dim
            self.me_G = build_model(self.config)
            self.me_G.set_eval()
        
        # prepare me summary
        if self.test_mode == 'me':
            if not os.path.exists(f"{self.config.test_dir}/me_summary.json"):
                prepare_me_summary(self.config.test_dir)

        self.test()


    def test(self):
        imgs = self.get_imgs()
        subject = 'all'
        passed = []

        for img in imgs:
            name = img.split('/')[-1].split('.')[0]
            print(f'\nSubject: {name}', end='')
            if (name != subject and subject != 'all') or name in passed:
                continue
            self.info['name'] = name

            if self.me_G is not None:
                self.generate_me(img, name)
            elif self.neutral_G is not None:
                self.generate_neutral(img, name, f'{self.config.test_dir}/neutral/{self.config.expr_name}')
            


    def generate_neutral(self, img, name, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        if os.path.exists(f'{save_dir}/{name}.jpg'):
            return

        target_au = np.zeros(self.neutral_c_dim)

        img = read_img(img)
        self.info['rgb_img'] = img
        self.preprocess_img(img)

        # save image
        neutral_img = self.generate(self.neutral_G, img, target_au, target_au)
        self.postprocess_img(neutral_img, f"{save_dir}/{name}.jpg")

        # extract aus
        self.extract_aus(f"{save_dir}/{name}.jpg", name, return_aus=False)


    def generate_me(self, img, name):
        # get me summary
        me_summary = f"{self.config.test_dir}/{self.config.me_summary}"
        with open(me_summary, 'r') as j:
            me_summary = json.load(j)

        # determine emotions
        emotions = self.config.me_emotions
        me_samples = {}
        for emo in self.emotions_list:
            if emotions == 'all' or emo in emotions:
                if self.config.me_samples == -1:
                    me_samples[emo] = me_summary[emo]
                else:
                    me_samples[emo] = random.sample(me_summary[emo], self.config.me_samples)

        # select input image
        neutral_img = f'{self.config.test_dir}/neutral/{self.config.neutral_expr_name}/{name}.jpg'
        if not os.path.exists(neutral_img):
            self.generate_neutral(img, name, f'{self.config.test_dir}/neutral/{self.config.neutral_expr_name}')
        
        img = read_img(neutral_img)
        self.info['rgb_img'] = img
        source_au = self.extract_aus(neutral_img, name)
        self.preprocess_img(img, crop=False)

        # generate me
        for emo in self.emotions_list:
            if emo in me_samples:
                print(f'\nGenerating {emo} micro-expressions...')
                self.info['emo'] = emo
                root = ''
                for sample in tqdm(me_samples[emo]):
                    if sample.startswith('s_'):
                        root = self.config.samm_aus_dir
                    elif sample.startswith('m_'):
                        root = self.config.mmew_aus_dir
                    elif sample.startswith('c_'):
                        root = self.config.casme_ii_aus_dir
                    
                    target_aus = self.get_aus(f'{root}/{"_".join(sample.split("_")[1:])}.csv')
                    for i, target_au in enumerate(target_aus):
                        me_img = self.generate(self.me_G, img, source_au, target_au)

                        # save image
                        save_dir = f'{self.config.test_dir}/{emo}/{self.config.expr_name}/{sample}/{name}'
                        os.makedirs(save_dir, exist_ok=True)
                        self.postprocess_img(me_img, f"{save_dir}/{i:03d}.jpg")


    def generate(self, model, img, source_au, target_au):
        # Returns the generated image (image tensor)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        face = self.info['X_real']
        face = torch.unsqueeze(transform(Image.fromarray(face)), 0)
        source_au = torch.unsqueeze(torch.from_numpy(source_au), 0)
        target_au = torch.unsqueeze(torch.from_numpy(target_au), 0)
        
        X_real = face.to(device=self.device, dtype=torch.float)
        y_real = source_au.to(device=self.device, dtype=torch.float)
        y_target = target_au.to(device=self.device, dtype=torch.float)
        model.set_batch_input(X_real, y_real, y_target)
        out = model.inference()

        return out


    def preprocess_img(self, img, crop=True):
        # detect image
        bbox = face_recognition.face_locations(img)
        if crop and len(bbox) > 0:
            top, right, bottom, left = bbox[0]
            face = img[top:bottom, left:right]
            original_shape = (face.shape[1], face.shape[0])
            face =  cv2.resize(face, (self.config.img_size, self.config.img_size))
            self.info['original_shape'] = original_shape
        else:
            if img.shape[0] != 128:
                face =  cv2.resize(img, (self.config.img_size, self.config.img_size))
            else:
                face = img
        
        self.info['X_real'] = face
        self.info['bbox'] = bbox

    
    def postprocess_img(self, img, path):
        # img: tensor image
        if len(self.info['bbox']) > 0 and self.config.keep_size:
            real_img = self.info['rgb_img']
            top, right, bottom, left = self.info['bbox'][0]
            dim = self.info['original_shape']
            real_img[top:bottom, left:right] = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            save_img(real_img, path)
        else:
            save_img(img, path)


    def get_imgs(self):
        dir = f'{self.config.test_dir}/{self.config.input_dir}'
        return glob.glob(f'{dir}/*.jpg')


    def get_aus(self, file):
        aus = []
        content = pd.read_csv(file)
        cols = [col for col in content.columns if 'AU' in col and '_r' in col]
        content = content[cols].to_numpy()
        for c in content:
            aus.append(c / 5.0)

        return aus
    

    def extract_aus(self, img, name, return_aus=True):
        # img: image path (string)
        # name: name of the image (string)

        save_dir = f"{'/'.join(img.split('/')[:-1])}"
        if not os.path.exists(f'{save_dir}/{name}.csv'):
            openface_dir = 'path to Openface /build/bin folder'
            openface_command = f'{openface_dir}/FaceLandmarkImg -f {img} -out_dir {save_dir} -of {name} -aus -wild'
            subprocess.run(openface_command, shell=True)
            subprocess.run(f'rm {save_dir}/{name}_of_details.txt', shell=True)

        if return_aus:
            # read aus
            df = pd.read_csv(f'{save_dir}/{name}.csv')
            col = [c for c in df.columns if 'AU' in c and '_r' in c]
            aus = df[col].values[0]
            return aus / 5.0
    

    def get_mouth_height(self, img):
        img = face_recognition.load_image_file(img)
        landmarks = face_recognition.face_landmarks(img)
        if len(landmarks) == 0:
            return None
        top_lip = landmarks[0]['top_lip']
        bottom_lip = landmarks[0]['bottom_lip']

        sum=0
        for i in [8,9,10]:
            # distance between two near points up and down
            distance = math.sqrt((top_lip[i][0] - bottom_lip[18-i][0])**2 + 
                                (top_lip[i][1] - bottom_lip[18-i][1])**2   )
            sum += distance
        return sum / 3



if __name__ == "__main__":
    Test()