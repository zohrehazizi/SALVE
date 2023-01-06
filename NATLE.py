import numpy as np
from skimage.color import hsv2rgb, rgb2hsv
import os
import time
from cv2 import fastNlMeansDenoising, fastNlMeansDenoisingColored, bilateralFilter
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab
from scipy.signal import medfilt2d
from sklearn.linear_model import Ridge
from torch import nn

from torch_helper_utils.RGB_HSV_HSL import hsv2rgb_torch, rgb2hsv_torch
from torch_helper_utils.Filters import LConvRegressor, RConvRegressor, MedianPool2d


alpha = 0.015
epsA = 0.001
epsR = 0.001
epsG = 0.05
sigma = 10
lambdaG = 1.5
beta = 3
gamma = 2.2


class natleRegressor(nn.Module):
    def __init__(self, epsR, gamma, regressor_kernel_size=5, medianfilter_kernel_size=3):
        super().__init__()
        self.gamma = gamma
        self.L_regressor = LConvRegressor(kernel_size=regressor_kernel_size, eps=epsR)
        self.median_filter = MedianPool2d(kernel_size=medianfilter_kernel_size)
        self.R_regressor = RConvRegressor(kernel_size=regressor_kernel_size)
    def denoise_rgb_np(self, rgb):
        max_val = np.max(rgb, axis=(0,1), keepdims=True)
        uint_rgb = np.uint8(rgb/max_val*255)
        # uint_denoised_rgb = fastNlMeansDenoisingColored(uint_rgb)
        uint_denoised_rgb = bilateralFilter(uint_rgb, 15, 75, 75)
        denoised_rgb = uint_denoised_rgb.astype(np.float32)/255.0*max_val
        return np.expand_dims(denoised_rgb, axis=0)
    def denoise_rgb(self, x):
        t0 = time.time()
        denoised_rgb = [self.denoise_rgb_np(x[i].cpu().permute(1,2,0).numpy()) for i in range(x.size(0))]
        denoised_rgb = np.concatenate(denoised_rgb, axis=0)
        denoised_rgb = torch.tensor(denoised_rgb).permute(0,3,1,2)
        t1 = time.time()
        #print(f"time inside denoise_rgb: {t1-t0}")
        return denoised_rgb
    def forward(self, x):
        if len(x.shape)==3:
            x = self.single_img_to_torch(x)
        else:
            assert (len(x.shape)==4), "image should be 3-d or 4-d tensor"
            x = self.multiple_images_to_torch(x)
        with torch.no_grad():
            hsv_torch = rgb2hsv_torch(x)
            L_torch = self.L_regressor(x)
            hsv_torch[:, 2:3, :, :]/=L_torch
            Rhat_rgb_torch =  hsv2rgb_torch(hsv_torch)
            rgbB4_torch = self.median_filter(Rhat_rgb_torch)
            RGBden_torch = self.denoise_rgb(rgbB4_torch).to(x.device)
            hsvnew_torch = rgb2hsv_torch(RGBden_torch)
            R_torch = self.R_regressor(hsvnew_torch[:, 2:3, :, :])
            L_new_torch = L_torch**(1/self.gamma)
            S_real_torch = R_torch*L_new_torch
            HSV_n_torch = torch.cat([hsvnew_torch[:, 0:2, :, :], S_real_torch] , dim=1)
            RGB_torch = hsv2rgb_torch(HSV_n_torch)
            RGB_torch = (torch.clip(RGB_torch, min=0, max=1)*255).to(torch.uint8)
            return self.torch_to_np(RGB_torch)

    def single_img_to_torch(self, imgArray):
        device = self.L_regressor.weight.device
        return (torch.tensor(imgArray, dtype=torch.float32, device=device)/255.0).unsqueeze(0).permute(0,3,1,2)
    def multiple_images_to_torch(self, imgArray):
        device = self.L_regressor.weight.device
        return (torch.tensor(imgArray, dtype=torch.float32, device=device)/255.0).permute(0,3,1,2)
    def torch_to_np(self, x):
        x = x.cpu().permute(0,2,3,1).numpy()
        if x.shape[0]==1:
            x = x.squeeze(0)
        return x

class image_enhancer():
    def __init__(self, img_size, alpha,epsA,epsR, epsG,sigma,lambdaG,beta, gamma):
        self.alpha = alpha
        self.epsA = epsA
        self.epsR = epsR
        self.epsG = epsG
        self.sigma = sigma
        self.lambdaG = lambdaG
        self.beta = beta
        self.gamma = gamma
        self.img_size = img_size

        h,w = img_size
        L_len = w*h
        self.L_len = L_len
        #self.D_val = np.matlib.repmat([-1,1], 1, L_len).squeeze()         #Dy
        self.D_val = np.tile([-1,1], (1, L_len)).squeeze()
        self.Dx, self.Dy = self.get_dx_dy()
        self.Dx_t = np.transpose(self.Dx)
        self.Dy_t = np.transpose(self.Dy)     

        vec3 = np.ones(L_len)     
        ind_i = range(0,h*w)
        self.sp_eye = csr_matrix((vec3, (ind_i,ind_i)), shape=(L_len, L_len))

        self.A_R = self.sp_eye + self.beta*(self.Dx_t*self.Dx + self.Dy_t*self.Dy)

        weight_x = torch.tensor([[[[0,0,0],[-1.0,0,1],[0,0,0]]]])
        weight_y = torch.tensor([[[[0,-1.0,0],[0,0,0],[0,1,0]]]])
        self.grad_weight = torch.cat((weight_x, weight_y), dim=0)

    def get_dx_dy(self, mask=None):
        h,w = self.img_size
        L_len = w*h
        if mask is not None:
            mask = mask.transpose(1,0).ravel()
            A_len = int(np.sum(mask==0))
        else:
            A_len = L_len
        #D_val = np.matlib.repmat([-1,1], 1, A_len).squeeze()         #Dy
        D_val = np.tile([-1,1], (1, A_len)).squeeze()
        col1 = np.arange(-1, w*h-1)
        col2=col1+2
        col1[0::h]+=1
        col2[h-1::h]-=1
        if mask is not None:
            col1 = col1[mask==0]
            col2 = col2[mask==0]
        Dy_indx = np.asarray([col1, col2]).transpose(1,0).ravel()
        col1 = np.arange(0, w*h)
        if mask is not None:
            col1 = col1[0:A_len]
        col2 = col1
        D_indy = np.asarray([col1, col2]).transpose(1,0).ravel()
        Dy = csr_matrix((D_val, (D_indy,Dy_indx)), shape=(A_len, L_len))    
        col1 = np.arange(h)
        col2 = col1+h
        Dx_indx1 = np.asarray([col1,col2]).transpose(1,0).ravel()
        col1 = np.arange((w-2)*h)
        col2 = col1+2*h
        Dx_indx2 = np.asarray([col1,col2]).transpose(1,0).ravel()
        col1 = np.arange((w-2)*h,(w-1)*h)
        col2 = col1+h
        Dx_indx3 = np.asarray([col1,col2]).transpose(1,0).ravel()
        Dx_indx = np.concatenate((Dx_indx1, Dx_indx2, Dx_indx3))
        if mask is not None:
            Dx_indx = Dx_indx.reshape(-1,2)
            Dx_indx = Dx_indx[mask==0]
            Dx_indx = Dx_indx.ravel()
        Dx = csr_matrix((D_val, (D_indy,Dx_indx)), shape=(A_len, L_len))
        return Dx, Dy 

    def find_L(self, Img, mask=None, find_reg=None):
        HSV = rgb2hsv(Img)                               #convert to hsv 
        #print('converted to hsv')
        S = HSV[:,:,2]                                    #v component
        Hue = HSV[:,:,0]
        Sat = HSV[:,:,1]
        Lhat = 0.2989*Img[:,:,0]+0.5870*Img[:,:,1]+0.1140*Img[:,:,2]       #L estimation

        t0 = time.time()
        grad_Lhat_x, grad_Lhat_y = self.find_grad(Lhat)        #L hat gradients
        t1 = time.time()
        #print('--in find_L > time for find_grad: ', t1-t0)

        A_x = (self.alpha)/(abs(grad_Lhat_x)+self.epsA)    #Ad(x) 
        A_y = (self.alpha)/(abs(grad_Lhat_y)+self.epsA)
        A_xvec = A_x.transpose().ravel()
        A_yvec = A_y.transpose().ravel()

        if mask is not None:
            A_xvec = A_xvec[mask.transpose().ravel()==0]
            A_yvec = A_yvec[mask.transpose().ravel()==0]
        A_len = len(A_xvec)

        h, w = Lhat.shape
        Lhat_vec = Lhat.transpose().reshape(-1,1)        #vectorizing
        L_len = len(Lhat_vec)
        i_ind = range(A_len)
        t2= time.time()
        diag_ax = csr_matrix((A_xvec, (i_ind, i_ind)), shape=(A_len, A_len))
        diag_ay = csr_matrix((A_yvec, (i_ind, i_ind)), shape=(A_len, A_len))
        #print('---time for diag_ax and diag_ay: ', time.time()-t2)

        t4=time.time()

        if mask is None:  
            A = self.sp_eye + self.Dx_t*diag_ax*self.Dx+ self.Dy_t*diag_ay*self.Dy
        else:
            Dx, Dy = self.get_dx_dy(mask)
            Dx_t = np.transpose(Dx)
            Dy_t = np.transpose(Dy)
            A = self.sp_eye + Dx_t*diag_ax*Dx+ Dy_t*diag_ay*Dy
        # A = self.sp_eye + np.transpose(self.Dx)*diag_ax*self.Dx+ np.transpose(self.Dy)*diag_ay*self.Dy
        A = A.transpose()

        

        b = Lhat_vec;
        time_before = time.time()
        L_vec = bicgstab(A,b, maxiter=100)[0]
        #print('time for A and b calculation: ', time_before-t1)
        #print('time for bicgstab: ', time.time()-time_before)

        L = L_vec.reshape(w,h).transpose()
        L[L==0] = self.epsR
        Rhat = np.divide(S, L)                                  # R^

        

        if find_reg:
            L_reg = Ridge(alpha=1.0)
            #L_reg = XGBRegressor(n_estimators=10, booster='gblinear')
            unfold = nn.Unfold(kernel_size=(5,5), stride=1)
            Lhat_tensor = torch.from_numpy(Lhat).unsqueeze(0).unsqueeze(0)
            Lhat_tensor = F.pad(Lhat_tensor, (2,2,2,2), mode='reflect')
            Lhat_patches = unfold(Lhat_tensor).squeeze().numpy().transpose()
            L_ = L.reshape(L.shape[0]*L.shape[1], -1)
            L_reg.fit(Lhat_patches,L_)

        if mask is None:
            if find_reg is None:
                return  L, S, Hue, Sat, Rhat
            else:
                return  L, S, Hue, Sat, Rhat, L_reg
        else:
            if find_reg is None:
                return  L, S, Hue, Sat, Rhat, Dx_t, Dy_t
            else:
                return  L, S, Hue, Sat, Rhat, Dx_t, Dy_t, L_reg

    def denoise_rgb(self, rgb):
        max_val = np.max(rgb, axis=(0,1), keepdims=True)
        uint_rgb = np.uint8(rgb/max_val*255)
        # uint_denoised_rgb = fastNlMeansDenoisingColored(uint_rgb)
        uint_denoised_rgb = bilateralFilter(uint_rgb, 15, 75, 75)
        denoised_rgb = uint_denoised_rgb.astype(np.float64)/255.0*max_val
        return denoised_rgb

    def find_grad(self, X):
        X_tensor = torch.tensor(X, device=self.grad_weight.device, dtype = self.grad_weight.dtype).unsqueeze(0).unsqueeze(0)
        X_tensor = F.pad(X_tensor, (1,1,1,1), mode='reflect')

        out = F.conv2d(input=X_tensor, weight=self.grad_weight, bias=None).squeeze().cpu().numpy()
        grad_x = out[0]
        grad_y = out[1]
        return grad_x, grad_y

    def find_G(self, S):
        Gx,Gy = self.find_grad(S)

        Gx[abs(Gx)<self.epsG] = 0
        Gy[abs(Gy)<self.epsG] = 0

        Gx = self.lambdaG*Gx
        Gy = self.lambdaG*Gy
        return Gx, Gy

    def forward_reg(self, img, L_reg, R_reg):
        if np.max(img)>10:
            img=(img*1.0)/255.0
        else:
            img=img*1.0
        HSV = rgb2hsv(img)#convert to hsv 

        

        #print('converted to hsv')
        S = HSV[:,:,2]                                    #v component
        Hue = HSV[:,:,0]
        Sat = HSV[:,:,1]
        

        Lhat = 0.2989*img[:,:,0]+0.5870*img[:,:,1]+0.1140*img[:,:,2]       #L estimation
        unfold = nn.Unfold(kernel_size=(5,5), stride=1)
        Lhat_tensor = torch.from_numpy(Lhat).unsqueeze(0).unsqueeze(0)
        Lhat_tensor = F.pad(Lhat_tensor, (2,2,2,2), mode='reflect')
        Lhat_patches = unfold(Lhat_tensor).squeeze().numpy().transpose()
        L_ = L_reg.predict(Lhat_patches)
        L = L_.reshape(Lhat.shape[0], Lhat.shape[1])
        L[L==0] = self.epsR


        


        Rhat = np.divide(S, L)
        h = Rhat.shape[0]
        w = Rhat.shape[1]
        Rhat_hsv = np.concatenate((np.expand_dims(Hue, axis=-1),np.expand_dims(Sat, axis=-1),np.expand_dims(Rhat, axis=-1)), axis=-1)
        Rhat_rgb = hsv2rgb(Rhat_hsv)
        Rden = Rhat_rgb[:,:,0]
        Gden = Rhat_rgb[:,:,1]
        Bden = Rhat_rgb[:,:,2]
        RdenB4 = medfilt2d(Rden)                          # Median Filter Denoising 
        GdenB4 = medfilt2d(Gden)
        BdenB4 = medfilt2d(Bden)
        rgbB4 = np.concatenate((np.expand_dims(RdenB4, axis=-1), np.expand_dims(GdenB4, axis=-1), np.expand_dims(BdenB4, axis=-1)), axis=-1)
        

        


        RGBden = self.denoise_rgb(rgbB4)
        hsvnew = rgb2hsv(RGBden)
        


        hueDen = hsvnew[:,:,0]
        satDen = hsvnew[:,:,1]
        RhatDen = hsvnew[:,:,2]
        RhatDen_tensor = torch.from_numpy(RhatDen).unsqueeze(0).unsqueeze(0)
        RhatDen_tensor = F.pad(RhatDen_tensor, (2,2,2,2), mode='reflect')
        RhatDen_patches = unfold(RhatDen_tensor).squeeze().numpy().transpose()
        R_ = R_reg.predict(RhatDen_patches)
        R = R_.reshape(Rhat.shape[0],Rhat.shape[1])

        L[L<0]=0
        L_new = L**(1/self.gamma)
        S_cmplx = R*L_new
        S_real = S_cmplx.real
        HSV_n = np.concatenate((np.expand_dims(hueDen, axis=-1),np.expand_dims(satDen, axis=-1),np.expand_dims(S_real, axis=-1)), axis=-1)
        RGB = hsv2rgb(HSV_n)

        return (np.clip(RGB, a_min=0, a_max=1)*255).astype(np.uint8)

    def forward(self, img, mask=None, find_reg=None):
        if np.max(img)>10:
            img=(img*1.0)/255.0
        else:
            img=img*1.0

        t0 = time.time()
        if mask is None:
            if find_reg is None:
                L, S, Hue, Sat, Rhat = self.find_L(img, mask=mask, find_reg=find_reg) 
            else:
                L, S, Hue, Sat, Rhat, L_reg = self.find_L(img, mask=mask, find_reg=find_reg)
            Dx_t = None
            Dy_t = None
        else:
            if find_reg is None:
                L, S, Hue, Sat, Rhat, Dx_t, Dy_t = self.find_L(img, mask=mask, find_reg=find_reg) 
            else:
                L, S, Hue, Sat, Rhat, Dx_t, Dy_t, L_reg = self.find_L(img, mask=mask, find_reg=find_reg)
        t1= time.time()
        #print('time to find L:', t1-t0)
        
        if find_reg is None:
            R, hueDen, satDen = self.find_R_denoise(Rhat, Hue, Sat, S, mask=mask, Dx_t=Dx_t, Dy_t=Dy_t, find_reg=find_reg)
        else:
            R, hueDen, satDen, R_reg = self.find_R_denoise(Rhat, Hue, Sat, S, mask=mask, Dx_t=Dx_t, Dy_t=Dy_t, find_reg=find_reg)
        t2= time.time()
        #print('time to find R and denoise:', t2-t1)
        L[L<0]=0
        L_new = L**(1/self.gamma) 
        if np.isnan(L_new).any():
            #print('L_new has nan')
            exit()
        S_cmplx = R*L_new
        S_real = S_cmplx.real
        t3 = time.time()
        #print('time for the rest of computation:', t3-t2)
        #print('-----total timeL ', t3-t0)

        self.V=S_real
        HSV_n = np.concatenate((np.expand_dims(hueDen, axis=-1),np.expand_dims(satDen, axis=-1),np.expand_dims(S_real, axis=-1)), axis=-1)
        RGB = hsv2rgb(HSV_n)

        if find_reg is None:
            return (np.clip(RGB, a_min=0, a_max=1)*255).astype(np.uint8)
        else:
            return (np.clip(RGB, a_min=0, a_max=1)*255).astype(np.uint8), L_reg, R_reg

    def find_R_denoise(self, Rhat,Hue,Sat,S, mask=None, Dx_t=None, Dy_t=None, find_reg=None):
        h = Rhat.shape[0]
        w = Rhat.shape[1]

        

        Rhat_hsv = np.concatenate((np.expand_dims(Hue, axis=-1),np.expand_dims(Sat, axis=-1),np.expand_dims(Rhat, axis=-1)), axis=-1)
        Rhat_rgb = hsv2rgb(Rhat_hsv)

        Rden = Rhat_rgb[:,:,0]
        Gden = Rhat_rgb[:,:,1]
        Bden = Rhat_rgb[:,:,2]

        t0=time.time()
        RdenB4 = medfilt2d(Rden)                          # Median Filter Denoising 
        GdenB4 = medfilt2d(Gden)
        BdenB4 = medfilt2d(Bden)
        rgbB4 = np.concatenate((np.expand_dims(RdenB4, axis=-1), np.expand_dims(GdenB4, axis=-1), np.expand_dims(BdenB4, axis=-1)), axis=-1)
        RGBden = self.denoise_rgb(rgbB4)
        #print('----time to denoise R: ', time.time()-t0)
        t1 = time.time()

        rgbnew = RGBden
        hsvnew = rgb2hsv(rgbnew)
        hueDen = hsvnew[:,:,0]
        satDen = hsvnew[:,:,1]
        RhatDen = hsvnew[:,:,2]

        RhatDenVec = RhatDen.transpose().reshape(self.L_len,1)

        Gx, Gy = self.find_G(S) 
        #print('----time to compute G: ', time.time()-t1)
        t2= time.time()
        Gx_vec = Gx.transpose().reshape(self.L_len, 1)
        Gy_vec = Gy.transpose().reshape(self.L_len, 1)
        if mask is not None:
            Gx_vec = Gx_vec[mask.transpose().ravel()==0]
            Gy_vec = Gy_vec[mask.transpose().ravel()==0]
            b = RhatDenVec+self.beta*(Dx_t*Gx_vec + Dy_t*Gy_vec)

        else:
            b = RhatDenVec+self.beta*(self.Dx_t*Gx_vec + self.Dy_t*Gy_vec)
        #print('----time to compute b: ', time.time()-t2)
        t3= time.time()

        R_vec = bicgstab(self.A_R,b, maxiter=100)[0]
        R = R_vec.reshape(w, h).transpose()
        #print('----time to solve equation: ', time.time()-t2)

        if find_reg:
            R_reg = Ridge(alpha=1.0)
            #R_reg = XGBRegressor(n_estimators=10, booster='gblinear')
            unfold = nn.Unfold(kernel_size=(5,5), stride=1)
            RhatDen_tensor = torch.from_numpy(RhatDen).unsqueeze(0).unsqueeze(0)
            RhatDen_tensor = F.pad(RhatDen_tensor, (2,2,2,2), mode='reflect')
            RhatDen_patches = unfold(RhatDen_tensor).squeeze().numpy().transpose()
            R_ = R.reshape(R.shape[0]*R.shape[1], -1)
            R_reg.fit(RhatDen_patches,R_)
            return R, hueDen, satDen, R_reg
        else:
            return R, hueDen, satDen




