# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:52:25 2020

@author: ssk
"""

import torch as t
from models.DSen2_CR import DSen2_CR
import utils.visualize as visualize
import utils.utils as utility
import sen12ms_cr_dataLoader
import time
import utils.img_operations as imgop
import config

def train(config):
    
    
    #提示使用GPU
    if config.use_gpu==False and t.cuda.is_available()==True:
        print("您有一块NVIDIA GPU:{},请在config里面修改“use_gpu”为True来使用它".format(t.cuda.get_device_name()))
    
    else:
        print("您正在使用NVIDIA {} 进行训练".format(t.cuda.get_device_name()))
        
        
        
        
    #可视化操作
    vis = visualize.Visualizer("DSen2_CR")
    
    
    
    
    #数据集 dataset 与 dataloader
    print("正在创建数据集...")
    
    dataset = sen12ms_cr_dataLoader.SEN12MSCRDataset(config.train_datset_dir)
    train_size = int(config.train_size*len(dataset))
    val_size =len(dataset)-train_size
    
    train_dataset,val_dataset = t.utils.data.random_split(dataset,[train_size,val_size])
    
    train_dataloader = t.utils.data.DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True)
    val_dataloader = t.utils.data.DataLoader(val_dataset,batch_size=config.batch_size,shuffle=True)
    val_dataiter=iter(val_dataloader)
    print("训练集大小：{}\r\n测试集大小：{}\r\n数据集初始化完毕".format(len(train_dataset),len(val_dataset)))
    
    
    
    #网络定义
    net = DSen2_CR(config.in_ch,config.out_ch,config.alpha,config.num_layers,config.feature_sizes)
    
    #如果有初始化的网络路径 则初始化
    if config.net_init is not None:
        param = t.load(config.net_init)
        net.load_state_dict(param)
        print("载入{}作为网络模型".format(config.net_init))
    
    #优化器
    opt = t.optim.Adam(net.parameters(),lr=config.lr,betas=(config.beta1,0.999),weight_decay=0.00001)
    
    #损失函数 
    CARL_Loss = imgop.carl_error
    
    #将数据装入gpu（在有gpu且使用GPU进行训练的情况下）
    cloud_img=t.FloatTensor(config.batch_size,config.in_ch,config.width,config.height)
    ground_truth=t.FloatTensor(config.batch_size,config.out_ch,config.width,config.height)
    csm_img=t.FloatTensor(config.batch_size,1,config.width,config.height)
    
    #如果使用GPU 则把这些玩意全部放进显存里面去
    if config.use_gpu:
        net = net.cuda()
        #CARL_Loss= CARL_Loss.cuda() 这是定义的外部函数 不能放入CUDA
        cloud_img=cloud_img.cuda()
        ground_truth=ground_truth.cuda()
        csm_img=csm_img.cuda()
    
    print("开始训练...")
    
    
    #开始循环
    for epoch in range(1,config.epoch):
        epoch_start_time = time.time()
        #数据集的小循环
        for iteration,batch in enumerate(train_dataloader,start=1):
            #数据操作    numpy 数据转成tensor
            img_cld,img_csm,img_truth=batch
            
            img_cld = cloud_img.resize_(img_cld.shape).copy_(img_cld)
            img_csm = csm_img.resize_(img_csm.shape).copy_(img_csm)
            img_truth = ground_truth.resize_(img_truth.shape).copy_(img_truth)
            
            #print(img_cld.shape,img_csm.shape,img_truth.shape)
            
            #网络训练
            img_fake = net(img_cld)
            opt.zero_grad()
            loss= CARL_Loss(img_truth,img_csm,img_fake)
            loss.backward() 
            opt.step()
            
            
            #可视化操作 显示loss等操作 还缺少一些记录损失的操作等等
            if iteration % config.show_freq == 0:                
                 with t.no_grad():
                     print("epoch[{}]({}/{}):loss_fake:{:.8f}".format(
                          epoch,iteration,len(train_dataloader),loss.item()))
                     #验证步骤
                     inputdata,s2CSMimg,s2img=next(val_dataiter)
                     inputdata_cuda=inputdata.cuda()
                     net.eval()
                     fake_img = net(inputdata_cuda)
                     net.train()
                     img_out=utility.GetQuadrupletsImg(inputdata, fake_img, s2img, s2CSMimg)
                     #print(img_out.shape)
                     vis.img("云图：预测：原图：掩膜",img_out)
                     vis.plot("loss_d_fake", loss.item())
            #保存网络
            if iteration%config.save_frequency==0:
                utility.save_state_dict(net, epoch,iteration)
            
        print("第{}轮训练完毕，用时{}S".format(epoch,time.time()-epoch_start_time))
    
    
        
if __name__=="__main__":
    myconfig=config.config()
    train(myconfig)
