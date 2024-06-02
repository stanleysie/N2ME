import cv2

def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is not None:
        # removing grayscale and low res images
        if len(img.shape) != 3:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def save_img(img, path):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)