# DSEN2_CR_PYTORCH

2023.12.26: 似乎原作者已经上传了自己的代码，可以[点击此链接](https://github.com/ameraner/dsen2-cr)查看

2023.12.26: It seems that the original authors have uploaded their own code, [click on this link](https://github.com/ameraner/dsen2-cr) to view.

DSen2_CR神经网络的pytorch版本

数据集下载地址：https://dataserv.ub.tum.de/index.php/s/m1554803<br>
或者百度网盘
链接：https://pan.baidu.com/s/11lypfXe24byyk5FM8yZ1ZA 
提取码：fxps
<br>
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
 
 ## 可以使用visdom ，打开CMD， 输入 pip install visdom 以下载,在CMD中输入 PYTHON -m visdom.server 就可以在浏览器中看到训练的图片和损失，如果不想要这个功能，请在train.py中删除包含"vis"的代码。
 
 ## 在config.py中可以调整网络设置。
 
 ## 使用python train.py 来训练网络
 ## 使用python predict.py 来预测。
      
