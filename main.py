from PIL import Image
import numpy as np
import os
from NATLE import image_enhancer, natleRegressor
import time


alpha = 0.015
epsA = 0.001
epsR = 0.001
epsG = 0.05
sigma = 10
lambdaG = 1.5
beta = 3
gamma = 2.2

frame_rate=24

if __name__=='__main__':

    dark_path = f'dark'
    enhanced_path = f'enhanced'

    regressor = natleRegressor(epsR, gamma, regressor_kernel_size=5, medianfilter_kernel_size=3)
    regressor.cuda()

    for d in sorted(os.listdir(dark_path)):
        enhancer=None
        if not os.path.exists(os.path.join(enhanced_path,d)):
            os.makedirs(os.path.join(enhanced_path,d))
        for d2 in sorted(os.listdir(os.path.join(dark_path,d))):
            img = Image.open(os.path.join(dark_path,d,d2))
            frame = np.array(img)
            frame_num = int(d2[0:-4])
            print('frame number:', frame_num)
            if frame_num%frame_rate==0:
                t0 =time.time()
                if enhancer is None:
                    enhancer = image_enhancer((frame.shape[0],frame.shape[1]), alpha,epsA,epsR, epsG,sigma,lambdaG,beta, gamma)
                enhanced, L_reg, R_reg = enhancer.forward(frame, find_reg=True)
                regressor.L_regressor.set_weights(L_reg)
                regressor.R_regressor.set_weights(R_reg)
                print('time for NATLE and reg fit:', time.time()-t0)
            else:
                t0 =time.time()
                enhanced = regressor(frame)
                print('time for predict:', time.time()-t0)
            img = Image.fromarray(enhanced)
            img.save(f"{enhanced_path}/{d}/{d2}")

