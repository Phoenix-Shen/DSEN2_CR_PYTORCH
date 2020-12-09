# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:24:49 2020

@author: ssk
"""
from models.DSen2_CR import DSen2_CR
import torch as t
import sen12ms_cr_dataLoader
import cv2
import numpy as np
from utils.utils import uint16to8
from utils.utils import getRGBImg
from utils.utils import GetQuadrupletsImg
from utils.utils import SaveImg
import os
import config

"""
步骤
1 取得网络输出
2 乘以scale
3 uint16 to uint8
4 np.squeeze()压缩维度 最后得到 13*256*256图片
5 取RGB图片输出
6 将无云图片 预测图片 网络输入（有云） 合成一张图片 保存输出
7 计算图像SSIM PSNR等参数
"""


    
def predict(config):
    #网络定义
    net = DSen2_CR(config.in_ch,config.out_ch,config.alpha,config.num_layers,config.feature_sizes)
    net = net.eval()
    
    #如果有初始化的网络路径 则初始化
    if config.net_init is not None:
        param = t.load(config.net_init)
        net.load_state_dict(param)
        print("载入{}作为网络模型".format(config.net_init))
    else:
        print("您没有输入网络模型路径，请在Config.py中的net_init行后面加上网络路径")
        return
    
    #数据集
    dataset = sen12ms_cr_dataLoader.SEN12MSCRDataset(config.predict_dataset_dir)
    dataloader = t.utils.data.DataLoader(dataset,batch_size=config.batch_size,shuffle=False)
    print("数据集初始化完毕，数据集大小为：{}".format(len(dataset)))
   
    #将数据装入gpu（在有gpu且使用GPU进行训练的情况下）
    cloud_img=t.FloatTensor(config.batch_size,config.in_ch,config.width,config.height)
    ground_truth=t.FloatTensor(config.batch_size,config.out_ch,config.width,config.height)
    csm_img=t.FloatTensor(config.batch_size,1,config.width,config.height)

    
    #如果使用GPU 则把这些玩意全部放进显存里面去
    #虽然不需要CSM 但是还是把它输出一下吧
    if config.use_gpu:
        net = net.cuda()
        cloud_img=cloud_img.cuda()
        ground_truth=ground_truth.cuda()
        csm_img=csm_img.cuda()
        
    
    with t.no_grad():
        for iteration,batch in enumerate(dataloader,1):
            img_cld,img_csm,img_truth=batch
            img_cld = cloud_img.resize_(img_cld.shape).copy_(img_cld)
            img_csm = csm_img.resize_(img_csm.shape).copy_(img_csm)
            img_truth = ground_truth.resize_(img_truth.shape).copy_(img_truth)
            img_fake = net(img_cld)
            
            output = GetQuadrupletsImg(img_cld, img_fake, img_truth, img_csm)
            
            SaveImg(output, os.path.join(config.output_dir,"iteration_{}.jpg".format(iteration)))
            
            
        
if __name__=="__main__":
    myconfig=config.config()
    predict(myconfig)

        
        
        