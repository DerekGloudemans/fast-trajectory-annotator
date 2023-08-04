import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
import cv2
import csv
import copy
import sys
import string
import cv2 as cv
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import _pickle as pickle 
import time
import scipy

import time
import random


from torchvision.transforms import functional as F
from torchvision.ops import roi_align, nms



# personal modules and packages
from i24_rcs import I24_RCS
from i24_database_api import DBClient
#from nvc_buffer import NVC_Buffer



# import pyproj
# from pyproj import Proj, transform
import pandas as pd
import numpy as np
import torch



# Get that bread - and by bread I mean data
gps_a = "./data/GPS_10hz_smooth_CORRECTED.cpkl"
gps_b = "./data/GPS_10hz_smooth_attempt.cpkl"
manual = "labeled_data_sandbox.cpkl"
detections = ""
with open(gps_a,"rb") as f:
    gps_a = pickle.load(f)
with open(gps_b,"rb") as f:
    gps_b = pickle.load(f)
with open(manual,"rb") as f:
    [man_data,man_objects] = pickle.load(f)
# with open(detections,"rb") as f:
#     detections = pickle.load(f)
    
    
    
active_direction = -1
highlight_id = "8_1"


colors = np.random.rand(1000,3)
# There will be a total of 5 subfigures


# get min and max ts
min_ts = +np.inf
max_ts = -np.inf

for fd in man_data:
    for key in fd:
        if fd[key][2] < min_ts:
            min_ts = fd[key][2]
        if fd[key][2] > max_ts:
            max_ts = fd[key][2]
            

fig,axs = plt.subplots(2,3,figsize = (30,20),gridspec_kw={'width_ratios': [2,1, 1]})


###### Subplot 1  - WB time-space
for gid in gps_a:
    if gps_a[gid]["y"][0] * active_direction > 0:
        axs[0,0].plot(gps_a[gid]["ts"],gps_a[gid]["x"],color = [0.5,0.5,0.5])#,marker = "x")
    
if highlight_id is not None:
    gid = highlight_id
    if gps_a[gid]["y"][0] * active_direction > 0:
        axs[0,0].plot(gps_a[gid]["ts"],gps_a[gid]["x"],color = [0,0.2,0.8 ], linewidth = 5)
        
# for frame_data in man_data:
#     for did in frame_data:
#         datum = frame_data[did]
#         gid = int(man_objects[did]["gps_id"].split("_")[0])
        
#         if datum[1] * active_direction > 0:
#             axs[0,0].scatter(datum[2],datum[0],color = colors[gid])
    
axs[0,0].set_xlim([min_ts,max_ts])
axs[0,0].set_ylim([0,23500])
plt.show()

# plt.figure(figsize = (30,10))
# colors = np.random.rand(200,3)

# for gid in self.gps:
#     plt.plot(self.gps[gid]["x"],self.gps[gid]["y"],color = colors[int(gid.split("_")[0])])
    
# for frame_data in self.data:
#     for did in frame_data:
#         datum = frame_data[did]
#         gid = int(self.objects[did]["gps_id"].split("_")[0])
        

#         plt.scatter(datum[0],datum[1],color = colors[gid])
    
# plt.show()
