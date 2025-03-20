# DSEN2_CR_PYTORCH

2023.12.26: 似乎原作者已经上传了自己的代码，可以[点击此链接](https://github.com/ameraner/dsen2-cr)查看  
2023.12.26: It seems that the original authors have uploaded their own code, [click on this link](https://github.com/ameraner/dsen2-cr) to view.

DSen2_CR神经网络的pytorch版本  
Pytorch version of the DSen2_CR neural network.

---

## 数据集下载地址 / Dataset Download Links  
- 官方下载地址 / Official Download Link: [https://dataserv.ub.tum.de/index.php/s/m1554803](https://dataserv.ub.tum.de/index.php/s/m1554803)  
- 百度网盘 / Baidu Netdisk:  
  - 链接 / Link: [https://pan.baidu.com/s/11lypfXe24byyk5FM8yZ1ZA](https://pan.baidu.com/s/11lypfXe24byyk5FM8yZ1ZA)  
  - 提取码 / Extraction Code: `fxps`  
  - 请注意，数据集很大，需要将所有压缩分卷下下来才能解压，建议预留500G磁盘空间，下载单个压缩包下来无法解压，如有任何问题，请在我的仓库下面提ISSUE，或者邮件联系我
---

## 数据组织方式 / Data Organization  
请按照以下方式组织数据：  
Please organize the data as follows:  
```bash
dataset_name
├── ROIs1158_spring
│ ├── s1_*
│ ├── s2_*
│ └── s2_cloudy_*
├── ROIs1868_summer
│ ├── s1_*
│ ├── s2_*
│ └── s2_cloudy_*
├── ROIs1970_fall
│ ├── s1_*
│ ├── s2_*
│ └── s2_cloudy_*
└── ROIs2017_winter
  ├── s1_*
  ├── s2_*
  └── s2_cloudy_*
```
---

## 使用说明 / Usage Instructions  

### 1. Visdom可视化 / Visdom Visualization  
- 可以使用Visdom进行可视化。  
  You can use Visdom for visualization.  
- 打开CMD，输入以下命令以下载Visdom：  
  Open CMD and enter the following command to install Visdom:  
  ```bash
  pip install visdom
  ```
- 启动Visdom服务器：
  Start the Visdom server:
  ```bash
  python -m visdom.server
  ```
- 在浏览器中查看训练图片和损失。
  View training images and losses in your browser.
- 如果不想要这个功能，请在train.py中删除包含"vis"的代码。
  If you don’t want this feature, delete the code containing "vis" in train.py.
  
### 2. 网络设置 / Network Configuration

- 在config.py中可以调整网络设置。
  You can adjust the network settings in `config.py`.
### 3. 训练与预测 / Training and Prediction
- 使用以下命令训练网络：
  Use the following command to train the network
  ```bash
  python train.py
  ```
- 使用以下命令进行预测：
  Use the following command for prediction:
  ```bash
  python predict.py
  ```
### 3. 注意事项 / Notes
- 确保数据集已正确组织并放置在dataset_name目录下。
  Ensure the dataset is organized correctly and placed in the dataset_name directory.
- 训练和预测过程中，请根据实际情况调整参数。
  Adjust parameters as needed during training and prediction.
