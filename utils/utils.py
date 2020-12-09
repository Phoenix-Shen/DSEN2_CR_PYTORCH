# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 22:50:11 2020

@author: ssk
"""

import os
import torch 
import cv2
import numpy as np
import  torch as t
#保存网络
def save_state_dict(net,epoch,iteration):
    net_path = os.path.join("./net_state_dict","net_epoch_{}_iteration_{}.pth".format(epoch,iteration))
    if not os.path.exists("./net_state_dict"):
        os.makedirs("./net_state_dict")     
    torch.save(net.state_dict(),net_path)
    print("第{}轮训练结果已经保存".format(epoch))
"""   
#生成Cloud and Shadow Mask
def Generate_Cluod_Mask(img,Tcl=0.2):
    toa = img.select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A', 'B9','B10', 'B11','B12']) \
              .divide(10000)
    toa = toa.addBands(img.select(['QA60']))
    # ['QA60', 'B1','B2',    'B3',    'B4',   'B5','B6','B7', 'B8','  B8A', 'B9',          'B10', 'B11','B12']
    # ['QA60','cb', 'blue', 'green', 'red', 're1','re2','re3','nir', 'nir2', 'waterVapor', 'cirrus','swir1', 'swir2'])
    #计算几个云指标 并取最小值
    score = ee.Image(1)
    #蓝色波段和卷云带的云十分明亮
    score = score.min(rescale(toa, 'img.B2', [0.1, 0.5]))
    score = score.min(rescale(toa, 'img.B1', [0.1, 0.3]))
    score = score.min(rescale(toa, 'img.B1 + img.B10', [0.15, 0.2]))
    #在可见波段（RGB），云都十分明亮
    score = score.min(rescale(toa, 'img.B4 + img.B3 + img.B2', [0.2, 0.8]))
    #NDSI =
    #[Green(band3)-SW IR(band11)]/
    #[Green(band3)+SW IR(band11)] ,
    #rescaled range [0.8; 0.6]
    #normalized difference snow index=>NDSI
    ndsi = img.normalizedDifference(['B3', 'B11'])
    score=score.min(rescale(ndsi, 'img', [0.8, 0.6]))
    
    #进行闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    score= cv2.morphologyEx(score,cv2.MORPH_CLOSE,kernel,iterations=2)
    #进行图像平均运算
    score=Averaging(score)
    #clip操作
    score=np.clip(score,0,1)
    #生成cloudmask
    score[score>Tcl]=1
    score[score<Tcl]=0
    return score
"""
def Generate_Cluod_Mask(img,Tcl=0.2):
    #首先将图片除以10000
    toa=img/10000
    _,length,width = img.shape
    score = np.ones((1,length,width),dtype=np.float32)
    #蓝色波段和卷云带的云十分明亮
    score = np.minimum(Rescale(toa[2,:,:],[0.1,0.5]),score)
    score = np.minimum(Rescale(toa[1,:,:],[0.1,0.3]),score)
    score = np.minimum(Rescale(toa[1,:,:]+toa[10,:,:],[0.15,0.2]),score)
    #在可见波段（RGB），云都十分明亮
    score = np.minimum(Rescale(toa[2,:,:]+toa[3,:,:]+toa[4,:,:],[0.2,0.8]),score)
    #NDSI =
    #[Green(band3)-SW IR(band11)]/
    #[Green(band3)+SW IR(band11)] ,
    #rescaled range [0.8; 0.6]
    #normalized difference snow index=>NDSI
    ndsi  = (toa[3,:,:]-toa[11,:,:])/(toa[3,:,:]+toa[11,:,:])
    score = np.minimum(Rescale(ndsi,[0.8,0.6]), score)
    #进行闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    score= cv2.morphologyEx(score,cv2.MORPH_CLOSE,kernel,iterations=2)
    #滤波
    score=cv2.blur(score,(7,7))
    #clip操作
    score=np.clip(score,0,1)
    #生成cloudmask
    score[score>Tcl]=1
    score[score<Tcl]=0
    #返回Cloud_Mask
    return score
    
    
    
#cv2.blur(img,(7,7))
#图像平均操作，在Generate_Cluod_Mask会用到 现已找到更好的方法 弃用！！！！！
def Averaging(img):
    img_H,img_W,img_ch = img.shape
    retimg=np.zeros((img_H,img_W,img_ch),dtype=np.uint8)
    for dstX in range(3,img_H-3):
        for dstY in range(3,img_W-3):
            average = np.zeros((img_ch))
            for i in [-3,-2,-1,0,1,2,3]:
                for j in [-3,-2,-1,0,1,2,3]:
                    average+=img[dstX+i,dstY+j]
            retimg[dstX,dstY]=average/49
    return retimg

#重新缩放功能，在Generate_Cluod_Mask会用到
def Rescale(img,thresholds):
    return (img-thresholds[0])/(thresholds[1]-thresholds[0])
"""
#重新缩放功能，在Generate_Cluod_Mask会用到
def rescale(img, exp, thresholds):
    return img.expression(exp, {"img": img}) \
              .subtract(thresholds[0]) \
              .divide(thresholds[1] - thresholds[0])
"""
              
#生成阴影掩膜
def Generate_Shadow_Mask(img,T_csi=3/4,T_wbi=5/6):
    img=img/1000
    csi = (img[7,:,:]+img[10,:,:])/2    
    wbi =img[0,:,:] 
    # ['QA60', 'B1','B2',    'B3',    'B4',   'B5','B6','B7', 'B8','  B8A', 'B9',          'B10', 'B11','B12']
    # ['QA60','cb', 'blue', 'green', 'red', 're1','re2','re3','nir', 'nir2', 'waterVapor', 'cirrus','swir1', 'swir2']
    shadow_mask = np.zeros((img.shape[1],img.shape[2]))
    shadow_mask[csi.any()<T_csi and wbi.any()<T_wbi]=1
    return np.expand_dims(shadow_mask,0)

#生成云掩膜和云阴影掩膜
def Generate_Cloud_and_Shadow_Mask(img):
    return np.logical_or(Generate_Cluod_Mask(img),Generate_Shadow_Mask(img)).astype(float)

def uint16to8(bands, lower_percent=0.001, higher_percent=99.999): 
    out = np.zeros_like(bands,dtype = np.uint8) 
    n = bands.shape[0] 
    for i in range(n): 
        a = 0 # np.min(band) 
        b = 255 # np.max(band) 
        c = np.percentile(bands[i, :, :], lower_percent) 
        d = np.percentile(bands[i, :, :], higher_percent) 
        
        t = a + (bands[i, :, :] - c) * (b - a) / (d - c) 
        t[t<a] = a 
        t[t>b] = b 
        out[i, :, :] = t 
    return out    

def getRGBImg(r,g,b,img_size=256):
    img=np.zeros((img_size,img_size,3),dtype=np.uint8)
    img[:,:,0]=r
    img[:,:,1]=g
    img[:,:,2]=b
    return img

def SaveImg(img,path):
    cv2.imwrite(path, img)
""" 
命名规范
['QA60', 'B1','B2',    'B3',    'B4',   'B5','B6','B7', 'B8','  B8A', 'B9',          'B10', 'B11','B12']
['QA60','cb', 'blue', 'green', 'red', 're1','re2','re3','nir', 'nir2', 'waterVapor', 'cirrus','swir1', 'swir2'])
[        1,    2,      3,       4,     5,    6,    7,    8,     9,      10,            11,      12,     13]) #gdal
[        0,    1,      2,       3,     4,    5,    6,    7,     8,      9,            10,      11,     12]) #numpy
[              BB      BG       BR                       BNIR                                  BSWIR1    BSWIR2
 ge. Bands 1, 2, 3, 8, 11, and 12 were utilized as BB , BG , BR , BNIR , BSWIR1 , and BSWIR2, respectively.
 
oprations:
img_cld   torch.Size([1, 15, 256, 256]) 取RGB
img_fake  torch.Size([1, 13, 256, 256]) 取RGB
img_truth torch.Size([1, 13, 256, 256]) 取RGB
img_csm   torch.Size([1, 1, 256, 256])  转RGB
"""

def GetQuadrupletsImg(img_cld,img_fake,img_truth,img_csm,img_size=256,scale=2000):
    #print(img_cld.shape,img_fake.shape,img_truth.shape)
    output_img=np.zeros((img_size,4*img_size,3),dtype=np.uint8)
    
    #压缩维度 转NUMPY 转维度 乘以缩放比 再转int8
    #转换之后维度分别为 256*256*15   256*256*15  256*256*15  256*256*1
    img_cld=  uint16to8((t.squeeze(img_cld  ).cpu().numpy()*scale).astype("uint16")).transpose(1,2,0)
    img_fake= uint16to8((t.squeeze(img_fake ).cpu().numpy()*scale).astype("uint16")).transpose(1,2,0)
    img_truth=uint16to8((t.squeeze(img_truth).cpu().numpy()*scale).astype("uint16")).transpose(1,2,0)
    
    img_csm=t.squeeze(img_csm,dim=0).cpu().numpy().transpose(1,2,0)
    #print(img_cld.shape,img_fake.shape,img_truth.shape)
    #取RGB
    img_cld_RGB=getRGBImg(img_cld[:,:,5], img_cld[:,:,4], img_cld[:,:,3], img_size)
    img_fake_RGB=getRGBImg(img_fake[:,:,3], img_fake[:,:,2], img_fake[:,:,1], img_size)
    img_truth_RGB=getRGBImg(img_truth[:,:,3], img_truth[:,:,2], img_truth[:,:,1], img_size)
    #print(img_cld_RGB,img_fake_RGB,img_truth_RGB)
    #CSM转三通道
    img_csm_RGB=np.concatenate((img_csm, img_csm, img_csm),axis=-1)*255
    
    #合成！
    output_img[:,0*img_size:1*img_size,:]=img_cld_RGB
    output_img[:,1*img_size:2*img_size,:]=img_fake_RGB
    output_img[:,2*img_size:3*img_size,:]=img_truth_RGB
    output_img[:,3*img_size:4*img_size,:]=img_csm_RGB
    return output_img