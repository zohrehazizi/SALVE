from skimage.color import hsv2rgb,rgb2hsv
from PIL import Image
import numpy as np
import os


alpha_darken = 1#np.random.uniform(0.9,1,1)
beta_darken = 1#np.random.uniform(0.5,1,1)
gamma_darken = 2.2#np.random.uniform(1.5,5,1)


def darken_frames(input_path, output_path, alpha, beta, gamma):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_dirs = [os.path.join(input_path,d) for d in os.listdir(input_path) if '.jpg' in d]
    for d in image_dirs:
        img = Image.open(d)
        frame = np.array(img)
        frame = frame*1.0/255.0
        frame_hsv = rgb2hsv(frame)
        frame_v = frame_hsv[:,:,2]
        dark_frame_v = beta*(alpha*frame_v)**gamma
        frame_hsv[:,:,2] = dark_frame_v
        frame_rgb = hsv2rgb(frame_hsv)
        dark_frame = (np.clip(frame_rgb, a_min=0, a_max=1)*255).astype(np.uint8)

        img = Image.fromarray(dark_frame)
        img.save(f"{output_path}/{d.split('/')[-1]}")


def noisy_dark_frames(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_dirs = [os.path.join(input_path,d) for d in os.listdir(input_path) if '.jpg' in d]
    for d in sorted(image_dirs):
        img = Image.open(d)
        frame = np.array(img)*1.0/255.0
        noisy1 = noisy("poisson", frame)
        noisy_frame = noisy("gauss", noisy1)

        noisy_dark_frame = (np.clip(noisy_frame, a_min=0, a_max=1)*255).astype(np.uint8)

        img = Image.fromarray(noisy_dark_frame)
        img.save(f"{output_path}/{d.split('/')[-1]}")
        print(d)

# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def noisy(noise_typ,image):
    row,col,ch= image.shape
    if noise_typ == "gauss":
        mean = 0
        #var = 0.1
        sigma = np.random.uniform(0.01,0.04,1) #var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "poisson":
        #vals = len(np.unique(image))
        #print(vals)
        #vals = 2 ** np.ceil(np.log2(vals))
        #print(vals)
        #noise = np.random.poisson(image * vals) / float(vals)
        noise = np.random.poisson(lam=np.random.uniform(0.01,0.04,1), size=image.shape)/10.0
        noise = noise.reshape(row,col,ch)

        #print(noise)

        noisy = image + noise
        return noisy



if __name__=='__main__':

    input_path = f'input'
    dark_path = f'dark'
    noisy_path = f'noisy'

    add_noise=False

    for d in os.listdir(input_path):
        darken_frames(os.path.join(input_path,d), os.path.join(dark_path,d), alpha_darken, beta_darken, gamma_darken)
    if add_noise:
        for d in os.listdir(dark_path):
            noisy_dark_frames(os.path.join(dark_path,d), os.path.join(noisy_path,d))
