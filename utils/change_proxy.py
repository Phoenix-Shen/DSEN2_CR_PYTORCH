# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:08:22 2020

@author: ssk
"""

#之前准备用GEE来完成操作 现在不用了 该类弃用
import ee

import os
# update the proxy settings
# os.environ['HTTP_PROXY'] = 'my_proxy_id:proxy_port'
# os.environ['HTTPS_PROXY'] = 'my_proxy_id:proxy_port'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:8889'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1089'

# Initialize the Earth Engine object, using the authentication credentials.
ee.Initialize()
image = ee.Image('CGIAR/SRTM90_V4')
