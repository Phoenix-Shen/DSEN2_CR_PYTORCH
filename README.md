# DSEN2_CR_PYTORCH
DSen2_CR神经网络的pytorch版本

数据集下载地址：https://dataserv.ub.tum.de/index.php/s/m1554803<br>
请您以一下方式组织数据：<br>
  dataset_name<br>
    - ROIs1158_spring<br>
      -- s1_* <br>
      -- s2_* <br>
      -- s2_cloudy_*<br>
    - ROIs1868_summer<br>
      -- 以此类推 <br>
    - ROIs1970_fall <br>
      -- 以此类推 <br>
    - ROIs2017_winter <br>
 
 # 可以使用visdom ，打开CMD， 输入 pip install visdom 以下载,在CMD中输入 PYTHON -m visdom.server 就可以在浏览器中看到训练的图片和损失，如果不想要这个功能，请在train.py中删除包含"vis"的代码。
 
 # 在config.py中可以调整网络设置。
 
 # 使用python train.py 来训练网络
 # 使用python predict.py 来预测。
      
