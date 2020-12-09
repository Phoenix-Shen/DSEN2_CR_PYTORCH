# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:46:11 2020

@author: ssk
"""


import numpy as np
from torch.utils import data
import os
from enum import Enum
from glob import glob
import rasterio
import utils.feature_detectors as  feature_detectors 
#S1波段定义
class S1Bands(Enum):
    VV = 1
    VH = 2
    ALL = [VV, VH]
    NONE = []

#S2波段定义
class S2Bands(Enum):
    B01 = aerosol = 1
    B02 = blue = 2
    B03 = green = 3
    B04 = red = 4
    B05 = re1 = 5
    B06 = re2 = 6
    B07 = re3 = 7
    B08 = nir1 = 8
    B08A = nir2 = 9
    B09 = vapor = 10
    B10 = cirrus = 11
    B11 = swir1 = 12
    B12 = swir2 = 13
    ALL = [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12]
    RGB = [B04, B03, B02]
    NONE = []

    
#四个季节 的枚举
class Seasons(Enum):
    SPRING = "ROIs1158_spring"
    SUMMER = "ROIs1868_summer"
    FALL = "ROIs1970_fall"
    WINTER = "ROIs2017_winter"
    ALL = [SPRING, SUMMER, FALL, WINTER]

#传感器 枚举
class Sensor(Enum):
    s1 = "s1"
    s2 = "s2"
    s2cloudy = "s2cloudy"
    
#训练集的定义
class SEN12MSCRDataset(data.Dataset):
    #初始化 获得所有图片列表
    def __init__(self,base_dir,scale=2000):
        
        self.scale=scale#光学图像缩小多少倍 img=img/scale
        self.base_dir = base_dir
        
        if not os.path.exists(self.base_dir):
            raise Exception("不存在该目录!请检查您的文件")
        
        self.img=[]
        for season in [Seasons.SPRING,Seasons.SUMMER,Seasons.FALL,Seasons.WINTER]:
            scene_ids = self.get_scene_ids(season)
            for scene_id in scene_ids:
                patch_ids=self.get_patch_ids(season,scene_id)
                for patch_id in patch_ids:
                    self.img.append([season,scene_id,patch_id])
        
                        
                       
        
    """
        Returns a list of scene ids for a specific season.
        获得指定季节的图片列表 如D:\迅雷下载\数据集\ROIs1970_fall目录下有哪些文件夹把这些文件夹去掉
        前缀只保留数字再取一个集合（无重复元素）
    """
    def get_scene_ids(self, season):
        season = Seasons(season).value
        path = os.path.join(self.base_dir, season)

        if not os.path.exists(path):
            raise NameError("Could not find season {} in base directory {}".format(
                season, self.base_dir))

        scene_list = [os.path.basename(s)
                      for s in glob(os.path.join(path, "*")) if "s2_cloudy" not in s]#找到该目录下面的所有文件
        scene_list = [int(s.split('_')[1]) for s in scene_list]#取数字部分
        return set(scene_list)#加set是为了去除重复的元素
    
    
    """
        Returns a list of patch ids for a specific scene within a specific season
        返回某个特定季节特定场景的文件，可以用len（patch_ids）查看它的长度
        D:\迅雷下载\数据集\ROIs1970_fall\lc_149 这个文件夹下面有多少个数据呢
    """
    def get_patch_ids(self, season, scene_id):
        season = Seasons(season).value
        path = os.path.join(self.base_dir, season, f"s1_{scene_id}")

        if not os.path.exists(path):
            raise NameError(
                "Could not find scene {} within season {}".format(scene_id, season))

        patch_ids = [os.path.splitext(os.path.basename(p))[0]
                     for p in glob(os.path.join(path, "*"))]
        patch_ids = [int(p.rsplit("_", 1)[1].split("p")[1]) for p in patch_ids]

        return patch_ids
    
    #len(dataset)来返回数据集长度
    def __len__(self):
        return len(self.img)
    
    #返回三个相应的数据 s1 s2 s2cloudy
    def get_triplets(self, season, scene_ids=None, patch_ids=None, s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, s2cloudy_bands=S2Bands.ALL):
        season = Seasons(season)
        scene_list = []
        patch_list = []
        bounds = []
        s1_data = []
        s2_data = []
        s2cloudy_data = []

        # This is due to the fact that not all patch ids are available in all scenes
        # And not all scenes exist in all seasons
        if isinstance(scene_ids, list) and isinstance(patch_ids, list):
            raise Exception("Only scene_ids or patch_ids can be a list, not both.")

        if scene_ids is None:
            scene_list = self.get_scene_ids(season)
        else:
            try:
                scene_list.extend(scene_ids)
            except TypeError:
                scene_list.append(scene_ids)

        if patch_ids is not None:
            try:
                patch_list.extend(patch_ids)
            except TypeError:
                patch_list.append(patch_ids)

        for sid in scene_list:
            if patch_ids is None:
                patch_list = self.get_patch_ids(season, sid)

            for pid in patch_list:
                s1, s2, s2cloudy, bound = self.get_s1s2s2cloudy_triplet(
                    season, sid, pid, s1_bands, s2_bands, s2cloudy_bands)
                s1_data.append(s1)
                s2_data.append(s2)
                s2cloudy_data.append(s2cloudy)
                bounds.append(bound)
        
        return np.stack(s1_data, axis=0), np.stack(s2_data, axis=0), np.stack(s2cloudy_data, axis=0), bounds
        
    
    def get_patch(self, season, scene_id, patch_id, bands, iscloudimg):
        season = Seasons(season).value
        sensor = None

        if isinstance(bands, (list, tuple)):
            b = bands[0]
        else:
            b = bands
        
        if isinstance(b, S1Bands):
            sensor = Sensor.s1.value
            bandEnum = S1Bands
        elif isinstance(b, S2Bands):
            sensor = Sensor.s2.value
            bandEnum = S2Bands
        else:
            raise Exception("Invalid bands specified")

        if isinstance(bands, (list, tuple)):
            bands = [b.value for b in bands]
        else:
            bands = bands.value
        if not iscloudimg:
            scene = "{}_{}".format(sensor, scene_id)
            filename = "{}_{}_p{}.tif".format(season, scene, patch_id)
            patch_path = os.path.join(self.base_dir, season, scene, filename)
            #print(patch_path)
        else:
            scene = "{}_cloudy_{}".format(sensor, scene_id)
            filename = "{}_{}_p{}.tif".format(season, scene, patch_id)
            patch_path = os.path.join(self.base_dir, season, scene, filename)
            #print(patch_path)

        with rasterio.open(patch_path) as patch:
            data = patch.read(bands)
            bounds = patch.bounds

        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)

        return data, bounds
    
   
    def get_s1s2s2cloudy_triplet(self, season, scene_id, patch_id, s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, s2cloudy_bands=S2Bands.ALL):
        s1, bounds = self.get_patch(season, scene_id, patch_id, s1_bands,False)
        s2, _ = self.get_patch(season, scene_id, patch_id, s2_bands,False)
        s2cloudy, _ = self.get_patch(season, scene_id, patch_id, s2cloudy_bands,True)

        return s1, s2, s2cloudy, bounds

    def __getitem__(self,index):
        s1img,s2img,s2cldimg,_=self.get_triplets(self.img[index][0],self.img[index][1],self.img[index][2])
        s1img=s1img.squeeze(0)
        s2img=s2img.squeeze(0)
        s2cldimg=s2cldimg.squeeze(0)
        
        #fill holes and artifacts
        s1img[np.isnan(s1img)] = np.nanmean(s1img)
        s2img[np.isnan(s2img)] = np.nanmean(s2img)
        s2cldimg[np.isnan(s2cldimg)] = np.nanmean(s2cldimg)
        #获得cloud and shadow mask (CSM) 获得的是256*256 要扩展一下维度
        s2CSMimg=feature_detectors.get_cloud_cloudshadow_mask(s2cldimg,0.2)
        s2CSMimg=np.expand_dims(s2CSMimg,axis=0)
        #进行最大值最小值裁剪 并进行适当的缩放
        s2img=(np.clip(s2img,0,10000)/self.scale).astype('float32')
        s2cldimg=(np.clip(s2cldimg,0,10000)/self.scale).astype('float32')
        
        s1img[0,:,:]=np.clip(s1img[0,:,:],-25.0,0.0)/(25)*2
        s1img[1,:,:]=np.clip(s1img[1,:,:],-35.0,0.0)/(35)*2
        
        inputdata=np.concatenate((s1img,s2cldimg),axis=0)
        #返回三个数据 输入神经网络的数据（15,256,256） Cloud and shadow mask（csm 1*256*256） 无云图片（target 13*256*256）
        
        return inputdata,s2CSMimg,s2img
        #return s1img,s2img,s2cldimg
        #注意 要使用或者输出的时候 要乘以一个scale

#测试
#dataset=SEN12MSCRDataset("E:\\dataset")
        