import math
import torchvision
import numpy as np
import matplotlib.pyplot as plt

def tensor2im(img, imtype=np.uint8, unnormalize=True, idx=0, nrows=None):
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)

    img = img.cpu().float()
    if unnormalize:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        for i, m, s in zip(img, mean, std):
            i.mul_(s).add_(m)

    image_numpy = img.numpy()
    image_numpy_t = np.transpose(image_numpy, (1, 2, 0))
    image_numpy_t = image_numpy_t*254.0

    return image_numpy_t.astype(imtype)

def tensor2maskim(mask, imtype=np.uint8, idx=0, nrows=1):
    im = tensor2im(mask, imtype=imtype, unnormalize=False, idx=idx, nrows=nrows)
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=-1)
    return im

def plot_au(img, aus, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    fig.subplots_adjust(0, 0, 0.8, 1)  # get rid of margins

    # display img
    ax.imshow(img)

    if len(aus) == 11:
        au_ids = ['1','2','4','5','6','9','12','17','20','25','26']
        x = 0.1
        y = 0.39
        i = 0
        for au, id in zip(aus, au_ids):
            if id == '9':
                x = 0.5
                y -= .15
                i = 0
            elif id == '12':
                x = 0.1
                y -= .15
                i = 0

            ax.text(x + i * 0.2, y, id, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='r', fontsize=20)
            ax.text((x-0.001)+i*0.2, y-0.07, au, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='b', fontsize=20)
            i+=1

    else:
        au_ids = ['1', '2', '4', '5', '6', '7', '9', '10', '12', '14', '15', '17', '20', '23', '25', '26', '45']
        if len(aus) == 18:
            au_ids.insert(-1, '27*')
        x = 0.1
        y = 0.39
        i = 0
        for au, id in zip(aus, au_ids):
            if id == '9' or id == '20':
                x = 0.1
                y -= .15
                i = 0

            ax.text(x + i * 0.2, y, id, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='r', fontsize=20)
            ax.text((x-0.001)+i*0.2, y-0.07, round(au, 4), horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='b', fontsize=15)
            i+=1

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data