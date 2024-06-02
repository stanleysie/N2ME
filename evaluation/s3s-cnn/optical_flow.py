import subprocess
import cv2
import numpy as np

class OpticalFlow:
    def __init__(self, emotions):
        self.emotions = emotions
        self.dataset = {
            "X": [],
            "y": []
        }
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')    


    def read_image(self, path):
        img = cv2.imread(path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray_img
    

    def generate_optical_strain(self, flow):
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
    

    def generate_optical_flow(self, index, path, emotion, subject, onset, apex):   
        # compute optical flow (TVL1Flow)
        subprocess.run(f"tvl1flow/tvl1flow '{onset}' '{apex}'", shell=True)
        print(f'synthetic {emotion} {subject}')

        onset_img, onset_img_gray = self.read_image(onset)

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

        # compute for the horizontal and vertical components of the optical flow
        horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
        vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
        horz = horz.astype('uint8')
        vert = vert.astype('uint8')

        # generate optical strain
        optical_strain = self.generate_optical_strain(flow)

        of_path = f"{path}/{index} {emotion} {subject} OF.jpg"
        os_path = f"{path}/{index} {emotion} {subject} OS.jpg"
        horz_path = f"{path}/{index} {emotion} {subject} H.jpg"
        vert_path = f"{path}/{index} {emotion} {subject} V.jpg"

        self.dataset['X'].append((of_path, os_path, horz_path, vert_path))
        self.dataset['y'].append(self.emotions.index(emotion))

        cv2.imwrite(of_path, rgb)
        cv2.imwrite(os_path, optical_strain)
        cv2.imwrite(horz_path, horz)
        cv2.imwrite(vert_path, vert)
    

    def get_dataset(self):
        return self.dataset