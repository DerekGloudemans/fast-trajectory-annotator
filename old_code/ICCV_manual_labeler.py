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
import re
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import _pickle as pickle 
import time
import scipy

import time
import random
import argparse

from homography import Homography,Homography_Wrapper
from datareader import Data_Reader , Camera_Wrapper
#from datareader import Camera_Wrapper_nvc as Camera_Wrapper

from curvilinear_homography import Curvilinear_Homography 

from scipy.signal import savgol_filter

from torchvision.transforms import functional as F
from torchvision.ops import roi_align, nms

detector_path = os.path.join("retinanet")
sys.path.insert(0,detector_path)

# filter and CNNs
from retinanet.model import resnet50 

class Annotator():
    """ 
    Annotator provides tools for labeling and correcting predicted labels
    for 3D objects tracked through space across multiple cameras. Camera timestamps
    are assumed to be out of phase and carry some error, which is adjustable 
    within this labeling framework. 
    
    Each camera and set of labels is in essence a discrete rasterization of continuous,vector-like data.
    Thus, several challenges arise in terms of how to manage out-of-phase and 
    out-of-period discretizations. The following guidelines are adhered to:
        
    i.  We base labels drawn in each camera view on the timestamp of that camera
    ii. We advance the first camera at each "frame", and adjust all other cameras 
        to be within 1/60th second of this time
    iii. We do not project labels from one camera into another
    iv. For most changes, we carry the change forward to all future frames in 
        the same camera view. These include:
            - shift in object x and y position
            - change in timestamp bias for a camera
    v.  We treat class and dimensions as constant for each object. 
        Adjusting these values adjusts them at all times across all cameras
    vi. When interpolating boxes, we assume constant velocity in space (ft)
    vii. We account for time bias once. Since we do not draw boxes across cameras,
         time bias is never used for plotting in this tool, but will be useful
         for labels later down the lien
    """
    
    
    def __init__(self,sequence_directory,scene_id = -1, homography_id = 1,exclude_p3c6 = False):
        
        
        # # get data
        # dr = Data_Reader(data,None,metric = False)
        # self.data = dr.data.copy()
        # del dr
        
        # # add camera tag to data keys
        # new_data = []
        # for frame_data in self.data:
        #     new_frame_data = {}
        #     for obj in frame_data.values():
        #         key = "{}_{}".format(obj["camera"],obj["id"])
        #         new_frame_data[key] = obj
        #     new_data.append(new_frame_data)
        # self.data = new_data
        
       
        self.scene_id = scene_id
        
        # get sequences
        self.sequences = {}
        for idx,sequence in enumerate(os.listdir(sequence_directory)):    
            if True and "p3c6" not in sequence or not exclude_p3c6:
                cap = Camera_Wrapper(os.path.join(sequence_directory,sequence))
                self.sequences[cap.name] = cap
        
        # get homography
        hid = "" if homography_id == 1 else "2"
        with open("EB_homography{}.cpkl".format(hid),"rb") as f:
            hg1 = pickle.load(f) 
        with open("WB_homography{}.cpkl".format(hid),"rb") as f:
            hg2 = pickle.load(f) 
        self.hg  = Homography_Wrapper(hg1=hg1,hg2=hg2)

        
        # sorted sequence list
        self.seq_keys = list(self.sequences.keys())
        self.seq_keys.sort()
        
        # # get ts biases
        # try:
        #     self.ts_bias = np.array([list(self.data[0].values())[0]["ts_bias"][key] for key in self.seq_keys])
        # except:
        #     for k_idx,key in enumerate(self.seq_keys):
        #         if key in  list(self.data[0].values())[0]["ts_bias"].keys():
        #             self.ts_bias[k_idx] = list(self.data[0].values())[0]["ts_bias"][key]
        
        self.cameras = [self.sequences[key] for key in self.seq_keys]
        [next(camera) for camera in self.cameras]
        self.active_cam = 0
        
        
        try:
            self.reload()  
            #self.reload_chg_data()
        except:
            self.data = []
            self.ts_bias = np.zeros(len(self.seq_keys))
            self.all_ts = []
            self.poly_params = dict([(camera.name+"_EB",[0,0,0]) for camera in self.cameras]+[(camera.name+"_WB",[0,0,0]) for camera in self.cameras])
            self.curve_points = dict([(camera.name+"_EB",[]) for camera in self.cameras]+[(camera.name+"_WB",[]) for camera in self.cameras])
        
        # get length of cameras, and ensure data is long enough to hold all entries
        self.max_frames = max([len(camera) for camera in self.cameras])
        while len(self.data) < self.max_frames:
            self.data.append({})

                

        # remove all data older than 1/60th second before last camera timestamp
        # max_cam_time = max([cam.ts for cam in self.cameras])
        # if not overwrite:
        #     while list(self.data[0].values())[0]["timestamp"] + 1/60.0 < max_cam_time:
        #         self.data = self.data[1:]

        # get first frames from each camera according to first frame of data
        self.buffer_frame_idx = -1
        self.buffer_lim = 1200
        self.last_frame = 2700
        self.buffer = []
        
        self.frame_idx = 0
        self.advance_all()

        self.cont = True
        self.new = None
        self.clicked = False
        self.clicked_camera = None
        self.TEXT = True
        self.LANES = True
        self.MASK = False
        
        self.active_command = "DIMENSION"
        self.right_click = False
        self.copied_box = None
        
        self.label_buffer = copy.deepcopy(self.data)
    
        self.colors =  np.random.rand(2000,3)
        
        loc_cp = "./localizer_april_112.pt"
        self.detector = resnet50(num_classes=8)
        cp = torch.load(loc_cp)
        self.detector.load_state_dict(cp) 
        self.detector.cuda()
        
        self.toggle_auto = True
        self.AUTO = True
        
        self.stride = 20
        self.plot_idx = 0
        
        
        ranges = {}
        extended_boxes = dict([(camera.name,0) for camera in self.cameras])
        
        for cam in self.cameras:
            cam = cam.name
            
            space_pts1 = self.hg.hg1.correspondence[cam]["space_pts"]
            space_pts2 = self.hg.hg2.correspondence[cam]["space_pts"]
            space_pts = np.concatenate((space_pts1,space_pts2),axis = 0)
    
            minx = np.min(space_pts[:,0])
            maxx = np.max(space_pts[:,0])
            
            ranges[cam] = [minx,maxx]
        self.ranges = ranges

        # TODO - fix path
        self.chg_save_file = "/home/derek/Documents/i24/ICCV_2023_Dataset/hg_scene2.cpkl"
        self.chg = Curvilinear_Homography(save_file = self.chg_save_file, downsample = 2)
        
        # load masks
        self.mask_ims = {}
        mask_dir = "/home/derek/Data/ICCV_2023/masks/scene2"
        mask_paths = os.listdir(mask_dir)
        for path in mask_paths:
            if "1080" in path:
                key = path.split("_")[0]
                path = os.path.join(mask_dir,path)
                im = cv2.imread(path)
                
                self.mask_ims[key] = im
        
    def port_homography(self):
        """
        Moves annotations from original roadway coordinate system to new curvilinear
        roadway coordinate system
        
        i.) select each individual annotation
        ii.) convert to image coordinates
        iii.) convert to curvilinear coordinates
        iv.) create tweak grid (+- F feet in either direction)
        v.) convert all boxes back to space
        vi.) find box with minimum error
        vii.) Save box and error
        """
        
        new_data = []
        data_porting_errors = []
        
        # i.) select each individual annotation
        idx = 0
        for frame_data in self.data:
            print("On frame {}".format(idx))
            new_frame_data = {}
            for key in frame_data.keys():
                obj_data = frame_data[key]
                obj_data = self.offset_box_y(copy.deepcopy(obj_data),reverse = True)
                
                # ii.) convert to image coordinates homography expects input of n_objects, 6 for state-> im conversion
                obj = torch.tensor([obj_data["x"],obj_data["y"],obj_data["l"],obj_data["w"],obj_data["h"],obj_data["direction"]]).unsqueeze(0).float()
                hg_name = key.split("_")[0]
                im_obj = self.hg.state_to_im(obj,hg_name)
                
                # iii.) convert to curvilinear coordinates
                d = "WB" if obj_data["direction"] == -1 else "EB"
                curve_hg_name = hg_name + "_" + d
                try:
                    curve_obj = self.chg.im_to_state(im_obj,name = [curve_hg_name], heights = torch.tensor(obj_data["h"]).unsqueeze(0))
                except KeyError as e:
                    print(e)
                    curve_obj = self.chg.im_to_state(im_obj,name = ["p2c3"], heights = torch.tensor(obj_data["h"]).unsqueeze(0))
                    continue
                #curve_obj = self.chg.im_to_state (im_obj*1,name = [curve_hg_name], heights = torch.tensor(obj_data["h"]).unsqueeze(0) )

                # overwrite dimensions with original dimensions
                # curve_obj[0,2] = obj_data["l"]
                # curve_obj[0,3] = obj_data["w"]
                # curve_obj[0,4] = obj_data["h"]
    
                # iv.) Generate object grid
                
                x_width = 3
                y_width = 3
                step = 0.1             
                x_tweak = torch.arange(-x_width,x_width,step).unsqueeze(1).repeat(int(2*y_width/step),1).view(-1)
                y_tweak = torch.arange(-y_width,y_width,step).unsqueeze(1).repeat(1,int(2*x_width/step)).view(-1)

                curve_obj_exp = curve_obj.repeat(int(2*y_width/step)*int(2*x_width/step),1)
                
                curve_obj_exp[:,0] += x_tweak
                curve_obj_exp[:,1] += y_tweak
                
                #.v) convert back to space
                curve_obj_im = self.chg.state_to_im(curve_obj_exp,name = [curve_hg_name for _ in range(len(curve_obj_exp))]) 
                
                #vi.) find box with minimum error
                im_obj_exp = im_obj.repeat(curve_obj_exp.shape[0],1,1)
                
                port_error = ((im_obj_exp[:,:4,:] - curve_obj_im[:,:4,:])**2).sum(dim = 2).sqrt().mean(dim = 1)
                
                min_error,min_idx = torch.min(port_error,dim = 0)
                min_x_tweak, min_y_tweak = x_tweak[min_idx],y_tweak[min_idx]
                
                new_datum = {}
                for key2 in obj_data.keys():
                    new_datum[key2] = obj_data[key2]
                new_datum["x"] = self.safe(curve_obj[0,0] + min_x_tweak)
                new_datum["y"] = self.safe(curve_obj[0,1] + min_y_tweak)
                
                new_frame_data[key] = new_datum
                data_porting_errors.append(min_error.item())
                
                #print(key,min_error.item())
            
            new_data.append(new_frame_data)
            idx += 1
            
            if idx %10 == 0:
        
        
                #print("Done Porting")
                dpe = torch.tensor(data_porting_errors)
                lt1 = torch.where(dpe < 1, 1,0).sum() / len(dpe)
                lt2 = torch.where(dpe < 2, 1,0).sum() / len(dpe)
                lt3 = torch.where(dpe < 3, 1,0).sum() / len(dpe)
                lt5 = torch.where(dpe < 5, 1,0).sum() / len(dpe)
                print("Less than 1px: {:.3f}   2px: {:.3f}    3px: {:.3f}  5px: {:.3f}".format(lt1,lt2,lt3,lt5))
                
            if idx > self.last_frame + 1 :
                break
            
        self.port_data = new_data
        self.data = new_data
        
        return data_porting_errors
    
    def safe(self,x):
        """
        Casts single-element tensor as an variable, otherwise does nothing
        """
        try:
            x = x.item()
        except:
            pass
        return x
    
    def clear_data(self):
        """
        For each timestep, a dummy object is added to store the time, and 
        all other objects are removed.
        """
        
        for f_idx in range(len(self.data)):
            self.data[f_idx] = {}
        
    def count(self):
        count = 0
        for frame_data in self.data:
            for key in frame_data.keys():
                count += 1
        print("{} total boxes".format(count))
    
    def toggle_cams(self,dir):
        """dir should be -1 or 1"""
        
        if self.active_cam + dir < len(self.seq_keys) -1 and self.active_cam + dir >= 0:
            self.active_cam += dir
            self.plot()
       
        if self.toggle_auto:
            self.AUTO = True
        
        if self.cameras[self.active_cam].name in ["p1c3","p1c4","p2c3","p2c4","p3c3","p3c4"]:
            self.stride = 10
        else:
            self.stride = 20
    
    def advance_cameras_to_current_ts(self):
        for c_idx,camera in enumerate(self.cameras):
            while camera.ts + self.ts_bias[c_idx] < self.current_ts - 1/60.0:
                next(camera)
        
        frames = [[cam.frame,cam.ts] for cam in self.cameras]
        
        self.buffer.append(frames)
        if len(self.buffer) > self.buffer_lim:
            self.buffer = self.buffer[1:]
      
    def advance_all(self):
        for c_idx,camera in enumerate(self.cameras):
                next(camera)
        
        frames = [[cam.frame,cam.ts] for cam in self.cameras]
        
        timestamps = {}
        for camera in self.cameras:
            timestamps[camera.name] = camera.ts
            
        if len(self.all_ts) <= self.frame_idx:
            self.all_ts.append(timestamps)
        
        self.buffer.append(frames)
        if len(self.buffer) > self.buffer_lim:
            #self.buffer = self.buffer[1:]
            del self.buffer[0]
            
    def advance_2(self,camera_idx = 0):
        for c_idx in [camera_idx,camera_idx+1]:
                camera = self.cameras[c_idx]
                next(camera)
        
        frames = [[cam.frame,cam.ts] for cam in self.cameras]
        
        timestamps = {}
        for camera in self.cameras:
            timestamps[camera.name] = camera.ts
            
        if len(self.all_ts) <= self.frame_idx:
            self.all_ts.append(timestamps)
        
        self.buffer.append(frames)
        if len(self.buffer) > self.buffer_lim:
            #self.buffer = self.buffer[1:]
            del self.buffer[0]
     
    def fill_buffer(self,n):
        for i in range(n):
            self.next()
            if i% 100 == 0: print("On frame {}".format(self.frame_idx))
        self.plot()
        print("Done")
               
    def next(self):
        """
        Advance a "frame"
        """        
        self.label_buffer = None
        
        if self.toggle_auto:
            self.AUTO = True

        if self.frame_idx < len(self.data) and self.frame_idx < self.last_frame:
            self.frame_idx += 1
            
            
            # if we are in the buffer, move forward one frame in the buffer
            if self.buffer_frame_idx < -1:
                self.buffer_frame_idx += 1
                
            # if we are at the end of the buffer, advance frames and store
            else:
                # advance cameras
                self.advance_all()
        else:
            print("On last frame")
    
    
    
    def prev(self):
        self.label_buffer = None
        
        if self.toggle_auto:
            self.AUTO = True

        
        if self.frame_idx > 0 and self.buffer_frame_idx > -self.buffer_lim:
            self.frame_idx -= 1            
            self.buffer_frame_idx -= 1
        else:
            print("Cannot return to previous frame. First frame or buffer limit")
                        
    def plot(self,extension_distance = 200):        
        plot_frames = []
        ranges = self.ranges
        
        
        for i in range(self.active_cam, self.active_cam+2):
           camera = self.cameras[i]
           cam_ts_bias =  self.ts_bias[i] # TODO!!!

           frame,frame_ts = self.buffer[self.buffer_frame_idx][i]
           frame = frame.copy()
           
           # get frame objects
           # stack objects as tensor and aggregate other data for label
           ts_data = list(self.data[self.frame_idx].values())
           ts_data = list(filter(lambda x: x["camera"] == camera.name,ts_data))
           
           
           #chg_data = list(self.port_data[self.frame_idx].values())
           #chg_data = list(filter(lambda x: x["camera"] == camera.name,chg_data))
           
           #ts_data = list(filter(lambda x: x["id"] == self.get_unused_id() - 1,ts_data))

           if False:
                ts_data = [self.offset_box_y(copy.deepcopy(obj),reverse = True) for obj in ts_data]
           ids = [item["id"] for item in ts_data]
           if len(ts_data) > 0:
               boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"]]).float() for obj in ts_data])
               
               # convert into image space
               cname = [camera.name for b in boxes]

               im_boxes = self.chg.state_to_im(boxes,name = cname)
                
               # plot on frame
               frame = self.hg.plot_state_boxes(frame,boxes,name = camera.name,color = (0,255,0),secondary_color = (0,255,0),thickness = 2,jitter_px = 0)
               frame = self.chg.plot_state_boxes(frame,boxes,name = cname,color = (0,100,255),thickness = 2)
           
               
               # plot labels
               if self.TEXT:
                   times = [item["timestamp"] for item in ts_data]
                   classes = [item["class"] for item in ts_data]
                   ids = [item["id"] for item in ts_data]
                   directions = [item["direction"] for item in ts_data]
                   directions = ["WB" if item == -1 else "EB" for item in directions]
                   camera.frame = Data_Reader.plot_labels(None,frame,im_boxes,boxes,classes,ids,None,directions,times)
                   
           # TEMP, to be removed after port
           if False and len(chg_data) > 0:
               boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"]]).float() for obj in chg_data])
               
               # convert into image space
               cname = [camera.name for b in boxes]
               chgim_boxes = self.chg.state_to_im(boxes,name = cname)
                
               # plot on frame
               frame = self.chg.plot_state_boxes(frame,boxes,name = cname,color = (0,0,255),thickness = 1)
           
            
           
           # if False and "c3" not in camera.name and "c4" not in camera.name: # plot splines
           #     times = [item["timestamp"] for item in ts_data]
           #     plot_spline_boxes = []
           #     # get objects visible in adjacent cameras (2 in each direction)
           #     ts_data2 = list(self.data[self.frame_idx].values())
           #     if True:
           #         ts_data2 = [self.offset_box_y(copy.deepcopy(obj),reverse = True) for obj in ts_data2]
           #     else:
           #         ts_data2 = [copy.deepcopy(obj) for obj in ts_data2]
           #     for obj in ts_data2:
           #         id = obj["id"]
           #         if id in ids:
           #             continue
           #         x_spline,y_spline = self.splines[id]
           #         if x_spline is None or y_spline is None or len(times) == 0:
           #             continue
           #         else:
           #             obj["x"] = x_spline([times[0]])[0]
           #             obj["y"] = y_spline([times[0]])[0]
                       
           #         if obj["x"] > ranges[camera.name][0] - extension_distance and obj["x"] < ranges[camera.name][1] + extension_distance: 
           #                 plot_spline_boxes.append(obj)
           #     if len(plot_spline_boxes) > 0:
           #         boxes2 = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"]]).float() for obj in plot_spline_boxes])
           #         im_boxes2 = self.hg.state_to_im(boxes2,name = camera.name)
                   
           #         keep = (torch.max(im_boxes2[:,:,0],dim = 1)[0] < 1920).int() * (torch.min(im_boxes2[:,:,0],dim = 1)[0] > 0).int() * (torch.max(im_boxes2[:,:,1],dim = 1)[0] < 1080).int() * (torch.max(im_boxes2[:,:,0],dim = 1)[1] > 0).int()
           #         boxes2 = boxes2[keep.nonzero().squeeze(1)]
           #         if len(boxes2) > 0:
           #             frame = self.hg.plot_state_boxes(frame,boxes2,name = camera.name,color = (255,190,0),secondary_color = (255,190,0),thickness = 1)



               
               # try:
               #     ts_data = list(self.spline_data[self.frame_idx].values())
               #     ts_data = list(filter(lambda x: x["camera"] == camera.name,ts_data))
               #     if True:
               #          ts_data = [self.offset_box_y(copy.deepcopy(obj),reverse = True) for obj in ts_data]
               #     if len(ts_data) > 0:
               #         boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"]]).float() for obj in ts_data])
                       
               #         # convert into image space
               #         im_boxes = self.hg.state_to_im(boxes,name = camera.name)
                        
               #         # plot on frame
               #         frame = self.hg.plot_state_boxes(frame,boxes,name = camera.name,color = (255,255,255),secondary_color = (255,255,255),thickness = 1)
               # except:
               #       pass
            
           if self.LANES:
                
                    for lane in range(-60,60,12):
                        # get polyline coordinates in space
                        x_curve = np.linspace(20000,30000,4000)
                        y_curve = np.ones(x_curve.shape) * lane
                        zeros = np.zeros(x_curve.shape)
                        curve = np.stack([x_curve,y_curve,zeros,zeros,zeros,zeros],axis = 1)
                        curve = torch.from_numpy(curve)
                        cname = [camera.name for i in range(x_curve.shape[0])]
                        curve_im = self.chg.state_to_im(curve,name = cname)
                        #curve_im = curve_im[:,0,:]
                       
                        mask = ((curve_im[:,:,0] > 0).int() + (curve_im[:,:,0] < 1920).int() + (curve_im[:,:,1] > 0).int() + (curve_im[:,:,1] < 1080).int()) == 4
                        curve_im = curve_im[mask,:]
                       
                        curve_im = curve_im.data.numpy().astype(int)
                        cv2.polylines(frame,[curve_im],False,(255,100,0),1)
                  
                    for tick in range(20000,30000,10):
                                y_curve = np.linspace(-60,60,8)
                                x_curve = y_curve *0 + tick
                                z_curve = y_curve *0
                                curve = np.stack([x_curve,y_curve,z_curve,z_curve,z_curve,z_curve],axis = 1)
                                curve = torch.from_numpy(curve)
                                cname = [camera.name for i in range(x_curve.shape[0])]
                                curve_im = self.chg.state_to_im(curve,name = cname)
                               
                                mask = ((curve_im[:,:,0] > 0).int() + (curve_im[:,:,0] < 1920).int() + (curve_im[:,:,1] > 0).int() + (curve_im[:,:,1] < 1080).int()) == 4
                                curve_im = curve_im[mask,:]
                               
                                curve_im = curve_im.data.numpy().astype(int)
                               
                                th = 1
                                color = (150,150,150)
                                if tick % 200 == 0:
                                    th = 2
                                    color = (255,100,0)
                                elif tick % 40 == 0:
                                     th = 2
                                    
                                cv2.polylines(frame,[curve_im],False,color,th)
                   
                
                   
           # print the estimated time_error for camera relative to first sequence
           # error_label = "Estimated Frame Time: {}".format(frame_ts)
           # text_size = 1.6
           # frame = cv2.putText(frame, error_label, (20,30), cv2.FONT_HERSHEY_PLAIN,text_size, [1,1,1], 2)
           # frame = cv2.putText(frame, error_label, (20,30), cv2.FONT_HERSHEY_PLAIN,text_size, [0,0,0], 1)
           # error_label = "Estimated Frame Bias: {}".format(cam_ts_bias)
           # text_size = 1.6
           # frame = cv2.putText(frame, error_label, (20,60), cv2.FONT_HERSHEY_PLAIN,text_size, [1,1,1], 2)
           # frame = cv2.putText(frame, error_label, (20,60), cv2.FONT_HERSHEY_PLAIN,text_size, [0,0,0], 1)
           
           
           
           if self.MASK:
               mask_im = self.mask_ims[camera.name]/255
               blur_im = cv2.blur(frame,(17,17))
               frame = frame*mask_im + blur_im * (1-mask_im)*0.7
           
           if True:
               font =  cv2.FONT_HERSHEY_SIMPLEX
               header_text = "{} frame {}".format(camera.name,self.frame_idx)
               frame = cv2.putText(frame,header_text,(30,30),font,1,(255,255,255),1)
               
           plot_frames.append(frame)
       
        # concatenate frames
        n_ims = len(plot_frames)
        n_row = int(np.round(np.sqrt(n_ims)))
        n_col = int(np.ceil(n_ims/n_row))
        
        cat_im = np.zeros([1080*n_row,1920*n_col,3]).astype(float)
        for i in range(len(plot_frames)):
            im = plot_frames[i]
            row = i // n_row
            col = i % n_row
            
            cat_im[col*1080:(col+1)*1080,row*1920:(row+1)*1920,:] = im
            
        # view frame and if necessary write to file
        cat_im /= 255.0
        
        
        self.plot_frame = cat_im
        
    def output_vid(self):
        self.LANES = False
        self.MASK = True
        #self.active_cam = 6
        directory = "video/scene2/{}".format(str(self.active_cam))
    
        while self.frame_idx < self.last_frame:
            
            # while self.active_cam < len(self.seq_keys):
            if not os.path.exists(directory):
                os.mkdir(directory)
        
            self.plot()
            #resize_im = cv2.resize(self.plot_frame*255,(3840,2160))
            cv2.imwrite("{}/{}.png".format(directory,str(self.frame_idx).zfill(4)),self.plot_frame*255)
            
            
            # warning, this de-syncs the cameras and is pretty dangerous for that reason
            self.advance_2(camera_idx = self.active_cam)
            self.frame_idx += 1
            
            if self.frame_idx % 10 == 0:
                print("On frame {}".format(self.frame_idx))
                
                
        # convert frames to video
        all_files = os.listdir(directory)
        all_files.sort()
        for filename in all_files:
            filename = os.path.join(directory, filename)
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            break
        
        n = 0
        f_name = os.path.join("/home/derek/Desktop",'{}_{}.mp4'.format(self.cameras[self.active_cam].name,self.cameras[self.active_cam+1].name))
        temp_name =  os.path.join("/home/derek/Desktop",'temp.mp4')
        
        out = cv2.VideoWriter(temp_name,cv2.VideoWriter_fourcc(*'mp4v'), 8, size)
         
        for filename in all_files:
            filename = os.path.join(directory, filename)
            img = cv2.imread(filename)
            out.write(img)
            print("Wrote frame {}".format(n))
            n += 1
            
            
        out.release()
        
        os.system("/usr/bin/ffmpeg -i {} -vcodec libx264 {}".format(temp_name,f_name))
        
        

    def add(self,obj_idx,location):
        
        xy = self.box_to_state(location)[0,:].data.numpy()
        
        
        
        # create new object
        obj = {
            "x": float(xy[0]),
            "y": float(xy[1]),
            "l": self.chg.class_dims["midsize"][0],
            "w": self.chg.class_dims["midsize"][1],
            "h": self.chg.class_dims["midsize"][2],
            "direction": 1 if xy[1] < 60 else -1,
            "class":"midsize",
            "timestamp": self.all_ts[self.frame_idx][self.clicked_camera],
            "id": obj_idx,
            "camera":self.clicked_camera,
            "gen":"Manual"
            }
        
        # try:
        #     key = "{}_{}".format(self.clicked_camera,obj_idx-1)
        #     obj["l"] = self.data[self.frame_idx][key]["l"]
        #     obj["w"] = self.data[self.frame_idx][key]["w"]
        #     obj["h"] = self.data[self.frame_idx][key]["h"]
        # except:
        #     pass
        
        key = "{}_{}".format(self.clicked_camera,obj_idx)
        self.data[self.frame_idx][key] = obj
        #self.save2()

    
    def box_to_state(self,point,direction = False):
        """
        Input box is a 2D rectangle in image space. Returns the corresponding 
        start and end locations in space
        point - indexable data type with 4 values (start x/y, end x/y)
        state_point - 2x2 tensor of start and end point in space
        """
        point = point.copy()
        #transform point into state space
        if point[0] > 1920:
            cam = self.seq_keys[self.active_cam+1]
            point[0] -= 1920
            point[2] -= 1920
        else:
            cam = self.seq_keys[self.active_cam]

        point1 = torch.tensor([point[0],point[1]]).unsqueeze(0).unsqueeze(0).repeat(1,8,1)
        point2 = torch.tensor([point[2],point[3]]).unsqueeze(0).unsqueeze(0).repeat(1,8,1)
        point = torch.cat((point1,point2),dim = 0)
        
        state_point = self.chg.im_to_state(point,name = [cam,cam], heights = torch.tensor([0]))
        
        return state_point[:,:2]
    
        
    def shift(self,obj_idx,box, dx = 0, dy = 0):
        
        key = "{}_{}".format(self.clicked_camera,obj_idx)
        item =  self.data[self.frame_idx].get(key)
        if item is not None:
            item["gen"] = "Manual"
        
        
        
        if dx == 0 and dy == 0:
            state_box = self.box_to_state(box)
            dx = state_box[1,0] - state_box[0,0]
            dy = state_box[1,1] - state_box[0,1]
        
        
        if np.abs(dy) > np.abs(dx): # shift y if greater magnitude of change
            # shift y for obj_idx in this and all subsequent frames
            for frame in range(self.frame_idx,len(self.data)):
                key = "{}_{}".format(self.clicked_camera,obj_idx)
                item =  self.data[frame].get(key)
                if item is not None:
                    item["y"] += dy
                break
        else:
            # shift x for obj_idx in this and all subsequent frames
           for frame in range(self.frame_idx,len(self.data)):
                key = "{}_{}".format(self.clicked_camera,obj_idx)
                item =  self.data[frame].get(key)
                if item is not None:
                    item["x"] += dx
                break
                                
    
    def change_class(self,obj_idx,cls):
        for camera in self.cameras:
             cam_name = camera.name
             for frame in range(0,len(self.data)):
                 key = "{}_{}".format(cam_name,obj_idx)
                 item =  self.data[frame].get(key)
                 if item is not None:
                     item["class"] = cls
    
    def paste_in_2D_bbox(self,box):
        """
        Finds best position for copied box such that the image projection of that box 
        matches the 2D bbox with minimal MSE error
        """
        
        if self.copied_box is None:
            return
        
        base = self.copied_box[1].copy()
        center = self.box_to_state(box).mean(dim = 0)
        
        if box[0] > 1920:
            box[[0,2]] -= 1920
        
        search_rad = 50
        grid_size = 11
        while search_rad > 1:
            x = np.linspace(center[0]-search_rad,center[0]+search_rad,grid_size)
            y = np.linspace(center[1]-search_rad,center[1]+search_rad,grid_size)
            shifts = []
            for i in x:
                for j in y:
                    shift_box = torch.tensor([i,j, base["l"],base["w"],base["h"],base["direction"]])
                    shifts.append(shift_box)
        
            # convert shifted grid of boxes into 2D space
            shifts = torch.stack(shifts)
            cname = [self.clicked_camera for _ in range(shifts.shape[0])]
            boxes_space = self.chg.state_to_im(shifts,name = cname)
            
            # find 2D bbox extents of each
            boxes_new =   torch.zeros([boxes_space.shape[0],4])
            boxes_new[:,0] = torch.min(boxes_space[:,:,0],dim = 1)[0]
            boxes_new[:,2] = torch.max(boxes_space[:,:,0],dim = 1)[0]
            boxes_new[:,1] = torch.min(boxes_space[:,:,1],dim = 1)[0]
            boxes_new[:,3] = torch.max(boxes_space[:,:,1],dim = 1)[0]
            
            # compute error between 2D box and each shifted box
            box_expanded = torch.from_numpy(box).unsqueeze(0).repeat(boxes_new.shape[0],1)  
            error = ((boxes_new - box_expanded)**2).mean(dim = 1)
            
            # find min_error and assign to center
            min_idx = torch.argmin(error)
            center = x[min_idx//grid_size],y[min_idx%grid_size]
            search_rad /= 5
            #print("With search_granularity {}, best error {} at {}".format(search_rad/grid_size,torch.sqrt(error[min_idx]),center))
        
        # save box
        base["x"] = self.safe(center[0])
        base["y"] = self.safe(center[1])
        base["camera"] = self.clicked_camera
        base["gen"] = "Manual"
        base["timestamp"] = self.all_ts[self.frame_idx][self.clicked_camera]
        key = "{}_{}".format(self.clicked_camera,base["id"])
        self.data[self.frame_idx][key] = base
        
    def automate(self,obj_idx):
        """
        Crop locally around expected box coordinates based on constant velocity
        assumption. Localize on this location. Use the resulting 2D bbox to align 3D template
        Repeat at regularly spaced intervals until expected object location is out of frame
        """
        # store base box for future copy ops
        cam = self.clicked_camera
        key = "{}_{}".format(cam,obj_idx)
        prev_box = self.data[self.frame_idx].get(key)
        
        if prev_box is None:
            return
        
        for c_idx in range(len(self.cameras)):
            if self.cameras[c_idx].name == cam:
                break
        
        crop_state = torch.tensor([prev_box["x"],prev_box["y"],prev_box["l"],prev_box["w"],prev_box["h"],prev_box["direction"]]).unsqueeze(0)
        boxes_space = self.chg.state_to_im(crop_state,name = [cam])
        boxes_new =   torch.zeros([boxes_space.shape[0],4])
        boxes_new[:,0] = torch.min(boxes_space[:,:,0],dim = 1)[0]
        boxes_new[:,2] = torch.max(boxes_space[:,:,0],dim = 1)[0]
        boxes_new[:,1] = torch.min(boxes_space[:,:,1],dim = 1)[0]
        boxes_new[:,3] = torch.max(boxes_space[:,:,1],dim = 1)[0]
        crop_box = boxes_new[0]
        
        # if crop box is near edge, break
        if crop_box[0] < 0 or crop_box[1] < 0 or crop_box[2] > 1920 or crop_box[3] > 1080:
            return
        
        # copy current frame
        frame = self.buffer[self.buffer_frame_idx][c_idx][0].copy()
        
        # get 2D bbox from detector
        box_2D = self.crop_detect(frame,crop_box)
        box_2D = box_2D.data.numpy()
        
        #shift to right view if necessary
        if self.active_cam != c_idx:
            crop_box[[0,2]] += 1920
            box_2D[[0,2]] += 1920
        
        # find corresponding 3D bbox
        self.paste_in_2D_bbox(box_2D.copy())
        
        # show 
        self.plot()
        
        #plot Crop box and 2D box
        self.plot_frame = cv2.rectangle(self.plot_frame,(int(crop_box[0]),int(crop_box[1])),(int(crop_box[2]),int(crop_box[3])),(0,0,255),2)
        self.plot_frame = cv2.rectangle(self.plot_frame,(int(box_2D[0]),int(box_2D[1])),(int(box_2D[2]),int(box_2D[3])),(0,0,255),1)
        cv2.imshow("window", self.plot_frame)
        cv2.waitKey(100)
    
    def crop_detect(self,frame,crop,ber = 1.2,cs = 112):
        """
        Detects a single object within the cropped portion of the frame
        """
        
        # expand crop to square size
        
        
        w = crop[2] - crop[0]
        h = crop[3] - crop[1]
        scale = max(w,h) * ber
        
        
        # find a tight box around each object in xysr formulation
        minx = (crop[2] + crop[0])/2.0 - (scale)/2.0
        maxx = (crop[2] + crop[0])/2.0 + (scale)/2.0
        miny = (crop[3] + crop[1])/2.0 - (scale)/2.0
        maxy = (crop[3] + crop[1])/2.0 + (scale)/2.0
        crop = torch.tensor([0,minx,miny,maxx,maxy])
        
        # crop and normalize image
        im = F.to_tensor(frame)
        im = roi_align(im.unsqueeze(0),crop.unsqueeze(0).float(),(cs,cs))[0]
        im = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]).unsqueeze(0)
        im = im.cuda()
        
        # detect
        self.detector.eval()
        self.detector.training = False
        with torch.no_grad():                       
            reg_boxes, classes = self.detector(im,LOCALIZE = True)
            confs,classes = torch.max(classes, dim = 2)
            
        # select best box
        max_idx = torch.argmax(confs.squeeze(0))
        max_box = reg_boxes[0,max_idx].data.cpu()
        
        # convert to global frame coordinates
        max_box = max_box * scale / cs
        max_box[[0,2]] += minx
        max_box[[1,3]] += miny
        return max_box
        
    
    def dimension(self,obj_idx,box, dx = 0, dy = 0):
        """
        Adjust relevant dimension in all frames based on input box. Relevant dimension
        is selected based on:
            1. if self.right_click, height is adjusted - in this case, a set ratio
               of pixels to height is used because there is inherent uncertainty 
               in pixels to height conversion
            2. otherwise, object is adjusted in the principle direction of displacement vector
        """
        
        key = "{}_{}".format(self.clicked_camera,obj_idx)
        item =  self.data[self.frame_idx].get(key)
        if item is not None:
            item["gen"] = "Manual"
        
        if dx == 0 and dy == 0:
            state_box = self.box_to_state(box)
            dx = state_box[1,0] - state_box[0,0]
            dy = state_box[1,1] - state_box[0,1]
            dh = -(box[3] - box[1]) * 0.02 # we say that 50 pixels in y direction = 1 foot of change
        else:
            dh = dy
        
        key = "{}_{}".format(self.clicked_camera,obj_idx)
        
        try:
            l = self.data[self.frame_idx][key]["l"]
            w = self.data[self.frame_idx][key]["w"]
            h = self.data[self.frame_idx][key]["h"]
        except:
            return
        
        if self.right_click:
            relevant_change = dh + h
            relevant_key = "h"
        elif np.abs(dx) > np.abs(dy): 
            relevant_change = dx + l
            relevant_key = "l"
        else:
            relevant_change = dy + w
            relevant_key = "w"
        
        for camera in self.cameras:
            cam = camera.name
            for frame in range(0,len(self.data)):
                 key = "{}_{}".format(cam,obj_idx)
                 item =  self.data[frame].get(key)
                 if item is not None:
                     item[relevant_key] = relevant_change
   
        # also adjust the copied box if necessary
        if self.copied_box is not None and self.copied_box[0] == obj_idx:
             self.copied_box[1][relevant_key] = relevant_change
    
    def copy_paste(self,point):     
        if self.copied_box is None:
            obj_idx = self.find_box(point)
            
            if obj_idx is None:
                return
            
            state_point = self.box_to_state(point)[0]
            
            key = "{}_{}".format(self.clicked_camera,obj_idx)
            obj =  self.data[self.frame_idx].get(key)
            
            if obj is None:
                return
            
            base_box = obj.copy()
            
            # save the copied box
            self.copied_box = (obj_idx,base_box,[state_point[0],state_point[1]].copy())
            
        
        else: # paste the copied box
            start = time.time()
            state_point = self.box_to_state(point)[0]
            
            obj_idx = self.copied_box[0]
            new_obj = copy.deepcopy(self.copied_box[1])


            dx = state_point[0] - self.copied_box[2][0] 
            dy = state_point[1] - self.copied_box[2][1] 
            new_obj["x"] += dx
            new_obj["y"] += dy
            new_obj["x"]  = new_obj["x"].item()
            new_obj["y"]  = new_obj["y"].item()
            new_obj["timestamp"] = self.all_ts[self.frame_idx][self.clicked_camera]
            new_obj["camera"] = self.clicked_camera
            new_obj["gen"] = "Manual"
            
            # remove existing box if there is one
            key = "{}_{}".format(self.clicked_camera,obj_idx)
            # obj =  self.data[self.frame_idx].get(key)
            # if obj is not None:
            #     del self.data[self.frame_idx][key]
                
            self.data[self.frame_idx][key] = new_obj
            
            if self.AUTO:
                self.automate(obj_idx)
                self.AUTO = False
            

            
    def interpolate(self,obj_idx,verbose = True):
        
        #self.print_all(obj_idx)
        
        for cur_cam in self.cameras:
            cam_name = cur_cam.name
        
            prev_idx = -1
            prev_box = None
            for f_idx in range(0,len(self.data)):
                frame_data = self.data[f_idx]
                    
                # get  obj_idx box for this frame if there is one
                cur_box = None
                for obj in frame_data.values():
                    if obj["id"] == obj_idx and obj["camera"] == cam_name:
                        del cur_box
                        cur_box = copy.deepcopy(obj)
                        break
                    
                if prev_box is not None and cur_box is not None:
                    
                    
                    for inter_idx in range(prev_idx+1, f_idx):   
                        
                        # doesn't assume all frames are evenly spaced in time
                        t1 = self.all_ts[prev_idx][cam_name]
                        t2 = self.all_ts[f_idx][cam_name]
                        ti = self.all_ts[inter_idx][cam_name]
                        p1 = float(t2 - ti) / float(t2 - t1)
                        p2 = 1.0 - p1                    
                        
                        
                        new_obj = {
                            "x": p1 * prev_box["x"] + p2 * cur_box["x"],
                            "y": p1 * prev_box["y"] + p2 * cur_box["y"],
                            "l": prev_box["l"],
                            "w": prev_box["w"],
                            "h": prev_box["h"],
                            "direction": prev_box["direction"],
                            "id": obj_idx,
                            "class": prev_box["class"],
                            "timestamp": self.all_ts[inter_idx][cam_name],
                            "camera":cam_name,
                            "gen":"Interpolation"
                            }
                        
                        key = "{}_{}".format(cam_name,obj_idx)
                        self.data[inter_idx][key] = new_obj
                
                # lastly, update prev_frame
                if cur_box is not None:
                    prev_idx = f_idx 
                    del prev_box
                    prev_box = copy.deepcopy(cur_box)
        
        if verbose: print("Interpolated boxes for object {}".format(obj_idx))
        
    def correct_homography_Z(self,box):
        dx = self.safe(box[2]-box[0]) 
        if dx > 500:
            sign = -1
        else:
            sign = 1
        # get dy in image space
        dy = self.safe(box[3] - box[1])
        delta = 10**(dy/1000.0)
        
        direction = torch.sign(self.box_to_state(box)[0,1])
        
        # if direction == 1:
        #     self.hg.hg1.correspondence[self.clicked_camera]["P"][:,2] *= sign*delta
        # else:   
        #     self.hg.hg2.correspondence[self.clicked_camera]["P"][:,2] *= sign*delta
        if direction == 1: self.chg.correspondence[self.clicked_camera + "_EB"]["P"][:,2] *= sign*delta
        else:              self.chg.correspondence[self.clicked_camera + "_WB"]["P"][:,2] *= sign*delta
            
        self.chg.save(self.chg_save_file)
        
    def correct_time_bias(self,box):
        
        # get relevant camera idx
        
        if box[0] > 1920:
            camera_idx = self.active_cam + 1
        else:
            camera_idx = self.active_cam
            
        # get dy in image space
        dy = box[3] - box[1]
        
        # 10 pixels = 0.001
        self.ts_bias[camera_idx] += dy* 0.0001
        
        self.plot_all_trajectories()
    
    def delete(self,obj_idx, n_frames = -1):
        """
        Delete object obj_idx in this and n_frames -1 subsequent frames. If n_frames 
        = -1, deletes obj_idx in all subsequent frames
        """
        frame_idx = self.frame_idx
        
        stop_idx = frame_idx + n_frames 
        if n_frames == -1:
            stop_idx = len(self.data)
        
        while frame_idx < stop_idx:
            try:
                key = "{}_{}".format(self.clicked_camera,obj_idx)
                obj =  self.data[frame_idx].get(key)
                if obj is not None:
                    del self.data[frame_idx][key]
            except KeyError:
                pass
            frame_idx += 1
        
   
    def get_unused_id(self):
        all_ids = []
        for frame_data in self.data:
            for item in frame_data.values():
                all_ids.append(item["id"])
                
        all_ids = list(set(all_ids))
        
        new_id = 0
        while True:
            if new_id in all_ids:
                new_id += 1
            else:
                return new_id
        
    def on_mouse(self,event, x, y, flags, params):
       if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
         self.start_point = (x,y)
         self.clicked = True 
       elif event == cv.EVENT_LBUTTONUP:
            box = np.array([self.start_point[0],self.start_point[1],x,y])
            self.new = box
            self.clicked = False
            
            
            if x > 1920:
                self.clicked_camera = self.seq_keys[self.active_cam+1]
                self.clicked_idx = self.active_cam + 1
            else:
                self.clicked_camera = self.seq_keys[self.active_cam]
                self.clicked_idx = self.active_cam
        
       # some commands have right-click-specific toggling
       elif event == cv.EVENT_RBUTTONDOWN:
            self.right_click = not self.right_click
            self.copied_box = None
            
       # elif event == cv.EVENT_MOUSEWHEEL:
       #      print(x,y,flags)
    
    def find_box(self,point):
        point = point.copy()
        
        #transform point into state space
        if point[0] > 1920:
            cam = self.seq_keys[self.active_cam+1]
            point[0] -= 1920
        else:
            cam = self.seq_keys[self.active_cam]

        point = torch.tensor([point[0],point[1]]).unsqueeze(0).unsqueeze(0).repeat(1,8,1)
        state_point = self.chg.im_to_state(point,name = [cam], heights = torch.tensor([0])).squeeze(0)
        
        min_dist = np.inf
        min_id = None
        
        for box in self.data[self.frame_idx].values():
            
            dist = (box["x"] - state_point[0] )**2 + (box["y"] - state_point[1])**2
            if dist < min_dist:
                min_dist = dist
                min_id = box["id"]  
        
        return min_id

    def keyboard_input(self):
        keys = ""
        letters = string.ascii_lowercase + string.digits
        while True:
            key = cv2.waitKey(1)
            for letter in letters:
                if key == ord(letter):
                    keys = keys + letter
            if key == ord("\n") or key == ord("\r"):
                break
        return keys    
      
    def quit(self):      
        self.cont = False
        cv2.destroyAllWindows()
        for cam in self.cameras:
            cam.release()
            
        self.save2()
    
    def undo(self):
        if self.label_buffer is not None:
            self.data[self.frame_idx] = self.label_buffer
            self.label_buffer = None
            self.plot()
        else:
            print("Can't undo")
    
    
    def plot_trajectory(self,obj_idx = 0):
        all_x = []
        all_y = []
        all_v = []
        all_time = []
        all_ids = []
        all_lengths = []
        all_cameras = []
        
        t0 = min(list(self.all_ts[0].values()))
        
        for cam_idx, camera in enumerate(self.cameras):
            cam_name = camera.name
        
            
            x = []
            y = []
            v = []
            camera = []
            time = []
            
            for frame in range(0,len(self.data),10):
                key = "{}_{}".format(cam_name,obj_idx)
                item = self.data[frame].get(key)
                if item is not None:
                    x.append(self.safe(item["x"]))
                    y.append(self.safe(item["y"]))
                    time.append(self.safe(item["timestamp"]) + self.ts_bias[cam_idx])
                    length = item["l"]
                    camera.append(item["camera"])
            
            
            
            if len(time) > 1:
                time = [item - t0 for item in time]

                # finite difference velocity estimation
                v = [(x[i] - x[i-1]) / (time[i] - time[i-1] + 1e-08) for i in range(1,len(x))] 
                v += [v[-1]]
                
                
                all_time.append(time)
                all_v.append(v)
                all_x.append(x)
                all_y.append(y)
                all_ids.append(obj_idx)
                all_lengths.append(length)
                all_cameras.append(camera)
                
        fig, axs = plt.subplots(3,sharex = True,figsize = (24,18))
        colors = self.colors
        
        for i in range(len(all_v)):
            
            cidx = all_ids[i]
            mk = ["s","D","o"][i%3]
            
            axs[0].scatter(all_time[i],all_x[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            axs[1].scatter(all_time[i],all_v[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            axs[2].scatter(all_time[i],all_y[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            for j in range(len(all_time[i])):
                axs[2].annotate(all_cameras[i][j],(all_time[i][j],all_y[i][j]))
            
            axs[0].plot(all_time[i],all_x[i],color = colors[cidx])#/(i%1+1))
            axs[1].plot(all_time[i],all_v[i],color = colors[cidx])#/(i%3+1))
            axs[2].plot(all_time[i],all_y[i],color = colors[cidx])#/(i%3+1))
            
            all_x2 = [all_lengths[i] + item for item in all_x[i]]
            axs[0].fill_between(all_time[i],all_x[i],all_x2,color = colors[cidx])
            
            axs[2].set(xlabel='time(s)', ylabel='Y-pos (ft)')
            axs[1].set(ylabel='Velocity (ft/s)')
            axs[0].set(ylabel='X-pos (ft)')

            axs[1].set_ylim(-150,150)
            
            fig.suptitle("Object {}".format(obj_idx))
        
        plt.show()  
    
    def plot_all_trajectories(self):
        all_x = []
        all_y = []
        all_v = []
        all_time = []
        all_ids = []
        all_lengths = []
        
        t0 = min(list(self.all_ts[0].values()))
        
        for cam_idx, camera in enumerate(self.cameras):
            cam_name = camera.name
        
            for obj_idx in range(self.get_unused_id()):
                x = []
                y = []
                v = []
                time = []
                
                for frame in range(0,len(self.data),10):
                    key = "{}_{}".format(cam_name,obj_idx)
                    item = self.data[frame].get(key)
                    if item is not None:
                        x.append(self.safe(item["x"]))
                        y.append(self.safe(item["y"]))
                        time.append(self.safe(item["timestamp"]) + self.ts_bias[cam_idx])
                        length = item["l"]
               
                
                
                
                if len(time) > 1:
                    time = [item - t0 for item in time]

                    # finite difference velocity estimation
                    v = [(x[i] - x[i-1]) / (time[i] - time[i-1] + 1e-08) for i in range(1,len(x))] 
                    v += [v[-1]]
                    
                    
                    all_time.append(time)
                    all_v.append(v)
                    all_x.append(x)
                    all_y.append(y)
                    all_ids.append(obj_idx)
                    all_lengths.append(length)
        
        fig, axs = plt.subplots(3,sharex = True,figsize = (24,18))
        colors = self.colors
        
        for i in range(len(all_v)):
            
            cidx = all_ids[i]
            mk = ["s","D","o"][i%3]
            
            # axs[0].scatter(all_time[i],all_x[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            # axs[1].scatter(all_time[i],all_v[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            # axs[2].scatter(all_time[i],all_y[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            
            axs[0].plot(all_time[i],all_x[i],color = colors[cidx])#/(i%1+1))
            axs[1].plot(all_time[i],all_v[i],color = colors[cidx])#/(i%3+1))
            axs[2].plot(all_time[i],all_y[i],color = colors[cidx])#/(i%3+1))
            
            all_x2 = [all_lengths[i] + item for item in all_x[i]]
            axs[0].fill_between(all_time[i],all_x[i],all_x2,color = colors[cidx])
            
            axs[2].set(xlabel='time(s)', ylabel='Y-pos (ft)')
            axs[1].set(ylabel='Velocity (ft/s)')
            axs[0].set(ylabel='X-pos (ft)')

            axs[1].set_ylim(-150,150)
        
        plt.show()  
        
    
    
    def create_trajectory(self,idx,y_order = 2, x_order = 3, plot = False,weight = True,metric = "ape",verbose = True,complexity = 2,space_knots = True):
        """
        Create a unified trajectory that minimizes the MSE betweeen trajectory
        and each consitituent bounding box (error computed in image space, trajectory
        computed in 3D space). Trajectory is generated by means of a sliding window
        weighted polynomial fit. The best fit polynomial is sampled at every point for
        which there exists a unique frame timestamp.
        
        idx - (int) index of unique object id for which to generate trajectory
        include_interp - (bool) if False, interpolated points are weighted lower than 
        y_order - (int > 0) order of polynomial with which to fit y coords (2 = constant acceleration)
        x_order - (int > 0) order polynomial to fit x coords (3 = constant jerk)
        plot - (bool) if True, show resulting trajectory and original trajectory
        """
        
        # for holding final trajectory points
        traj_x  = []
        traj_y  = []
        traj_ts = []
        x_err   = []
        y_err   = []
        
        # 1. Compile all boxes from all camera views
        cameras = []
        boxes = []
        for f_idx,frame in enumerate(self.data):
            
            if f_idx > self.last_frame:# and len(self.data[f_idx]) == 0:
                break
            
            for cam in self.seq_keys:
                
                key = "{}_{}".format(cam,idx)
                box = frame.get(key)
                
                if box is not None:
                    boxes.append(box)
                    cameras.append(cam)
        # stack 
        if len(boxes) == 0:
            return [None,None,None,None,None]
        interp = torch.tensor([(1 if ("gen" in item.keys() and item["gen"] == "Interpolation") else 1) for item in boxes])
        boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"],obj["timestamp"]],dtype = torch.double) for i,obj in enumerate(boxes)])
        
        
        # 2. for each point, find the x/y sensitivity to change 
        # a one foot change in space results in a _ pixel change in image space
         
        # convert boxes to im space - n_boxes x 8 x 2 in order: fbr,fbl,bbr,bbl,ftr,ftl,fbr,fbl
        boxes_im = self.chg.state_to_im(boxes.float(),name = cameras)
        
        # y_weight = width in pixels / width in feet
        #y_diff = torch.mean(torch.sum(torch.pow(boxes_im[:,[0,2],:] - boxes_im[:,[1,3],:],2), dim = 2).sqrt(),dim = 1)
        y_diff = torch.sum(torch.pow(torch.mean(boxes_im[:,[0,2],:] - boxes_im[:,[1,3],:],dim = 1),2),dim = 1).sqrt()        
        y_weight = y_diff / boxes[:,3]
        
        # x_weight = length in pixels / legnth in feet
        #x_diff = torch.mean(torch.sum(torch.pow(boxes_im[:,[0,1],:] - boxes_im[:,[2,3],:],2), dim = 2).sqrt(),dim = 1)
        x_diff = torch.sum(torch.pow(torch.mean(boxes_im[:,[0,1],:] - boxes_im[:,[2,3],:],dim = 1),2),dim = 1).sqrt()
        x_weight = x_diff / boxes[:,2]

        # 3. sort points and sensitivities by increasing timestamp
        min_ts = torch.min(boxes[:,6]).item()
        #boxes[:,6] -= min_ts 
        
        min_ts = torch.min(boxes[:,6]).item()
        max_ts = torch.max(boxes[:,6]).item()
        duration = max_ts - min_ts
        
        order    = boxes[:,6].sort()[1]
        boxes    = boxes[order]
       
        interp   = interp[order]
        y_weight = y_weight[order]
        x_weight = x_weight[order]  
        
        if weight:
            y_weight2 = y_weight * interp
            x_weight2 = x_weight * interp
        else:
            y_weight2 =  interp
            x_weight2 =  interp
        
        if duration < 3:
            return [None,None,None,None,None]
        # weight last point and derivative highly
        x_weight2[[0,-1]] *= 500
        y_weight2[[0,-1]] *= 500
        x_weight2[[1,-2]] *= 100
        y_weight2[[1,-2]] *= 100
        
        
        # sum of weighted squared errors (weight applied before exponent) <= s
        # we constrain s so s.t. on average, each box is misaligned below some threshold in pixels
        # if target pixel error is tpe,  we say sum((w * error)**2) <= len(interp)*tpe**2
        
        # we want to continually lower tpe until max error is below a threshold, number of knots is too high, or there is no satisfying spline
        t = boxes[:,6].data.numpy()

        if not space_knots:
            best_max_error = np.inf
            best_spline = None
            for tpex in list(np.logspace(1,0,200)):
                x_spline = scipy.interpolate.fitpack2.UnivariateSpline(boxes[:,6],boxes[:,0],k = x_order,w=x_weight,s = tpex**2*sum(interp))
    
                if np.isnan(x_spline(0)):
                     continue
                if len(x_spline.get_knots()) > np.floor(duration)*complexity+2:
                    continue
                
                else:
                    xpe = (x_weight2.data.numpy()*(x_spline(t) - boxes[:,0].data.numpy()))**2
                    
                    if metric == "ape":
                        ape =  np.sqrt(np.mean(xpe))
                        if ape < best_max_error:
                            best_spline = x_spline
                            best_max_error = ape
                    else:
                        mpe =  np.sqrt(np.max(xpe))
                        if mpe < best_max_error:
                            best_spline = x_spline
                            best_max_error = mpe   
            x_spline = best_spline
            
            best_max_error = np.inf
            best_spline = None
            for tpey in list(np.logspace(2,0,100)):
                y_spline = scipy.interpolate.fitpack2.UnivariateSpline(boxes[:,6],boxes[:,1],k = y_order,w=y_weight,s = tpey**2*sum(interp))
                if np.isnan(y_spline(0)):
                     continue
                if len(y_spline.get_knots()) > np.floor(duration)+2:
                    continue
                
                else:
                    ype = (y_weight.data.numpy()*(y_spline(t) - boxes[:,1].data.numpy()))**2
                    if metric == "ape":
                        ape =  np.sqrt(np.mean(ype))
                        if ape < best_max_error:
                            best_spline = y_spline
                            best_max_error = ape
                    else:
                        mpe =  np.mean(np.max(ype))
                        if mpe < best_max_error:
                            best_spline = y_spline
                            best_max_error = mpe
            y_spline = best_spline
            
        else:
            try:
                spc = 0.5
                knot_t = np.arange(min_ts,max_ts,spc,dtype = np.double)[x_order:-x_order]
                x_spline = scipy.interpolate.LSQUnivariateSpline(boxes[:,6],boxes[:,0],knot_t,k = x_order,w = x_weight)
    
    #            print(knot_t,min_ts,max_ts)
    
                spc = 3
                knot_t = np.arange(min_ts,max_ts,spc,dtype = np.double)[y_order:-y_order]
                y_spline = scipy.interpolate.LSQUnivariateSpline(boxes[:,6],boxes[:,1],knot_t,k = y_order,w = y_weight)
            except:
                x_spline = None
                y_spline = None
                print("Exception for object {}- no spline".format(idx))

          
        try:
            xpe = (x_weight.data.numpy()*(x_spline(t) - boxes[:,0].data.numpy()))**2
            ype = (y_weight.data.numpy()*(y_spline(t) - boxes[:,1].data.numpy()))**2
            xse = (x_spline(t) - boxes[:,0].data.numpy())**2
            yse = (y_spline(t) - boxes[:,1].data.numpy())**2
        except:
            return [None,None,None,None,None]        
        
        avg_x = np.sqrt(np.mean(xse))
        avg_y = np.sqrt(np.mean(yse))
        avg_xp = np.sqrt(np.mean(xpe))
        avg_yp = np.sqrt(np.mean(ype))
        max_xp = np.sqrt(np.max(xpe))
        max_yp = np.sqrt(np.max(ype))
        
        if verbose:
            print("Object {}: Space error: {:.2f}ft x, {:.2f}ft y ------ Pixel Error: {:.2f}/{:.2f}px x, {:.2f}/{:.2f}px y".format(
            idx,avg_x,avg_y,avg_xp,max_xp,avg_yp,max_yp))

        
        if plot:
            fig, axs = plt.subplots(2,sharex = True,figsize = (24,18))
            t = boxes[:,6].data.numpy()
            t2 = t - min_ts
            axs[0].scatter(t2,boxes[:,0],c = [(0.8,0.3,0)])
            axs[1].scatter(t2,boxes[:,1],c = [(0.8,0.3,0)])
            t = np.linspace(min_ts,max_ts,1000)
            t2 = t - min_ts
            axs[0].plot(t2,x_spline(t),color = (0,0,0.8),linewidth = 2)#/(i%1+1))
            axs[1].plot(t2,y_spline(t),color = (0,0.6,0),linewidth = 2)#/(i%3+1))
            
            axs[0].set_ylabel("X-position (ft)", fontsize = 24)
            axs[1].set_ylabel("Y-position (ft)", fontsize = 24)
            axs[1].set_xlabel("time (s)", fontsize = 24)


            #axs[0].set(ylabel='X-pos (ft)',fontsize = 24)
            axs[0].tick_params(axis='x', labelsize=18 )
            axs[0].tick_params(axis='y', labelsize=18 )
            axs[1].tick_params(axis='x', labelsize=18 )
            axs[1].tick_params(axis='y', labelsize=18 )
            
            axs[0].set_xlim([0,60])
            axs[1].set_xlim([0,60])
            plt.subplots_adjust(hspace=0.02)
            plt.savefig("splines{}.pdf".format(idx))
            plt.show()        
        
        
        return [t,x_spline,y_spline,avg_x,avg_xp]
    
    
    def gen_trajectories(self):
        spline_data = copy.deepcopy(self.data)
        for idx in range(self.get_unused_id()):
            if self.splines is None:
                t,x_spline,y_spline,ape,mpe = self.create_trajectory(idx)
            else:
                x_spline,y_spline = self.splines[idx]
            if x_spline is None: 
                continue
                print("skip")
                
            for f_idx in range(len(spline_data)):
                for cam in self.seq_keys:
                    key = "{}_{}".format(cam,idx)
                    
                    box = spline_data[f_idx].get(key)
                    if box is not None:
                        box["x"] = x_spline(box["timestamp"]).item()
                        box["y"] = y_spline(box["timestamp"]).item()
            
        self.spline_data = spline_data
      
    def get_splines(self,plot = True,metric = "ape"):
        splines = []
        apes = []
        ases = []
        for idx in range(self.get_unused_id()):
            t,x_spline,y_spline,ase,ape = self.create_trajectory(idx,plot = plot,metric = metric)
            splines.append([x_spline,y_spline])
            if ape is not None:
                ases.append(ase)
                apes.append(ape)
                
        print("Spline errors: {}ft ase, {}px ape".format(sum(ases)/len(ases),sum(apes)/len(apes)))    
        self.splines = splines 
     
    def adjust_boxes_with_trajectories(self,max_shift_x = 2,max_shift_y = 2,verbose = False):
        """
        Adjust each box by up to max_shift pixels in x and y direction towards the best-fit spline
        """               
        
        print("NOT IMPLEMENTED")
        
        pixel_shifts = []
        try:
            self.splines
        except:
            print("Splines not yet fit - cannot adjust boxes using splines")
            return
        
        for f_idx,frame_data in enumerate(self.data):
            
            if f_idx > self.last_frame:
                break
            if len(self.data[f_idx]) == 0:
                continue
            
            ids = [obj["id"] for obj in frame_data.values()]
            cameras = [obj["camera"] for obj in frame_data.values()]
            boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"],obj["timestamp"]],dtype = torch.double) for obj in frame_data.values()])

                
            boxes_im = self.hg.state_to_im(boxes.float(),name = cameras)
            boxes = boxes.data.numpy()
            
            # get feet per pixel ratio for y
            y_diff = torch.sum(torch.pow(torch.mean(boxes_im[:,[0,2],:] - boxes_im[:,[1,3],:],dim = 1),2),dim = 1).sqrt()        
            y_lim = (boxes[:,3] / y_diff * max_shift_y).data.numpy()
            
            # get feet per pixel ration for x        
            x_diff = torch.sum(torch.pow(torch.mean(boxes_im[:,[0,1],:] - boxes_im[:,[2,3],:],dim = 1),2),dim = 1).sqrt()
            x_lim = (boxes[:,2] / x_diff * max_shift_x).data.numpy()
            
            # for each box
            for i in range(len(ids)):
                
                if self.splines[ids[i]][0] is not None:
                    #get xpos on spline
                    ts = boxes[i,6]
                    x_spl = self.splines[ids[i]][0](ts)
                    
                    # get difference
                    x_diff = x_spl - boxes[i,0]
                    
                    if x_diff < -x_lim[i]:
                        x_diff = -x_lim[i]
                    elif x_diff > x_lim[i]:
                        x_diff = x_lim[i]
                    
                    # move box either to spline or to x_lim
                    key = "{}_{}".format(cameras[i],ids[i])
                    frame_data[key]["x"] += x_diff
                
                if self.splines[ids[i]][1] is not None:
                    #get ypos on spline
                    ts = boxes[i,6]
                    y_spl = self.splines[ids[i]][1](ts)
                    
                    # get difference
                    y_diff = y_spl - boxes[i,1]
                
                    if y_diff < -y_lim[i]:
                        y_diff = -y_lim[i]
                    elif y_diff > y_lim[i]:
                        y_diff = y_lim[i]
                    
                    # move box either to spline or to x_lim
                    key = "{}_{}".format(cameras[i],ids[i])
                    frame_data[key]["y"] += y_diff
                    
                    pixel_shifts.append(np.sqrt(x_diff**2 + y_diff**2))
            
            if verbose and (f_idx %100 == 0): print("Adusted boxes for frame {}".format(f_idx))
            
        return pixel_shifts
           
                        
    def adjust_ts_with_trajectories(self,max_shift = 1,trials = 101,overwrite_ts_data = False,metric = "ape", use_running_error = False,verbose = True):
        
        print("NOT IMPLEMENTED")
        
        splines = self.splines
        
        running_error = [self.ts_bias[i] for i in range(len(self.seq_keys))]
        if not use_running_error:
            running_error = [0 for item in running_error]

        # for each camera, for each frame, get labels
        for f_idx,frame_data in enumerate(self.data):
            
            if f_idx % 100 == 0:
                print("Adjusting ts for frame {}".format(f_idx))
                
            if f_idx > self.last_frame and len(self.data[f_idx]) == 0:
                break 
            for c_idx, cam in enumerate(self.seq_keys):
                
                # get all frame/camera labels
                objs = []
                ids = []
                for idx in range(self.get_unused_id()):
                    key = "{}_{}".format(cam,idx)
                    obj = frame_data.get(key)
                    
                    if obj is not None:
                        objs.append(obj)
                        ids.append(idx)
                
                id_splines = [splines[id][0] for id in ids]
                yid_splines = [splines[id][1] for id in ids]
                
                if len(id_splines) == 0:
                    continue
                
                # get x_weights
                boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"],obj["timestamp"]],dtype = torch.double) for obj in objs])
                
                # a one foot change in space results in a _ pixel change in image spac         
                # convert boxes to im space - n_boxes x 8 x 2 in order: fbr,fbl,bbr,bbl,ftr,ftl,fbr,fbl
                cname = [cam for i in range(len(boxes))]
                #boxes_im = self.chg.state_to_im(boxes.float(),name = cname)
                boxes_im = self.hg.state_to_im(boxes.float(),name = cam)

                
                
                # x_weight = length in pixels / legnth in feet
               
                
                # remove all objects without valid splines from consideration
                keep = [True if item is not None else False for item in id_splines]
                keep_splines = []
                for i in range(len(keep)):
                    if keep[i]:
                        keep_splines.append(id_splines[i])
                id_splines = keep_splines
                
                
                x_weight = 1
                if True:
                    #x_diff = torch.mean(torch.sum(torch.pow(boxes_im[:,[0,1],:] - boxes_im[:,[2,3],:],2), dim = 2).sqrt(),dim = 1)
                    x_diff = torch.sum(torch.pow(torch.mean(boxes_im[:,[0,1],:] - boxes_im[:,[2,3],:],dim = 1),2),dim = 1).sqrt()
                    x_weight = x_diff / boxes[:,2]  
                    x_weight = x_weight[keep]
                
                boxes = boxes[keep]

                
                if len(id_splines) == 0:
                    continue
                
                best_time = copy.deepcopy(self.all_ts[f_idx][cam])
                best_error = np.inf
                
                #get initial error
                xs = torch.tensor([id_splines[i](best_time).item() for i in range(len(id_splines))])
                xlabel = boxes[:,0]
                if metric == "ape":
                        init_error = torch.mean(((xs-xlabel)*x_weight).pow(2)).sqrt()
                else:
                        init_error =  torch.max(((xs-xlabel)*x_weight).pow(2)).sqrt()            
                
                for shift in np.linspace(-max_shift,max_shift,trials):
                    if use_running_error:
                        shift += running_error[c_idx]
                    new_time = self.all_ts[f_idx][cam] + shift 
                
                    # for each timestamp shift, compute the error between spline position and label position
                    xs = torch.tensor([id_splines[i](new_time).item() for i in range(len(id_splines))])
                    xlabel = boxes[:,0]
                    

                    
                    if metric == "ape":
                        mse = torch.mean(((xs-xlabel)*x_weight).pow(2)).sqrt()
                    else:
                        mse =  torch.max(((xs-xlabel)*x_weight).pow(2)).sqrt()
                        
                    if mse < best_error:
                        best_error = mse
                        best_time = new_time
                
                if verbose:
                    print("{} frame {}: shifted time by {:.3f}s --- {:.2f}x initial error".format(cam,f_idx,best_time - self.all_ts[f_idx][cam],best_error/init_error))
                
                for idx in range(self.get_unused_id()):
                    key = "{}_{}".format(cam,idx)
                    obj = frame_data.get(key)
                    
                    if obj is not None:
                        self.data[f_idx][key]["timestamp"] = best_time
                
                if overwrite_ts_data:
                    self.all_ts[f_idx][cam] = best_time

                if use_running_error:
                    running_error[c_idx] += shift
        
        
        
    def plot_one_lane(self,lane = (70,85)):
        
        print("NOT IMPLEMENTED")
        
        all_x = []
        all_y = []
        all_v = []
        all_time = []
        all_ids = []
        all_lengths = []
        
        t0 = min(list(self.all_ts[0].values()))
        
        for cam_idx, camera in enumerate(self.cameras):
            cam_name = camera.name
        
            for obj_idx in range(self.get_unused_id()):
                x = []
                y = []
                v = []
                time = []
                
                for frame in range(0,len(self.data)):
                    key = "{}_{}".format(cam_name,obj_idx)
                    item = self.data[frame].get(key)
                    if item is not None:
                        
                        y_test = self.safe(item["y"])
                        if y_test > lane[0] and y_test < lane[1]:
                            x.append(self.safe(item["x"]))
                            y.append(self.safe(item["y"]))
                            time.append(self.safe(item["timestamp"]) + self.ts_bias[cam_idx])
                            length = item["l"]
               
                
                
                
                if len(time) > 1:
                    time = [item - t0 for item in time]

                    # finite difference velocity estimation
                    v = [(x[i] - x[i-1]) / (time[i] - time[i-1] + 1e-08) for i in range(1,len(x))] 
                    v += [v[-1]]
                    
                   
                    
                    all_time.append(time)
                    all_v.append(v)
                    all_x.append(x)
                    all_y.append(y)
                    all_ids.append(obj_idx)
                    all_lengths.append(length)
        
        fig, axs = plt.subplots(2,sharex = True,figsize = (24,18))
        colors = self.colors
        
        for i in range(len(all_v)):
            
            
            cidx = all_ids[i]
            mk = ["s","D","o"][i%3]
            
            # axs[0].scatter(all_time[i],all_x[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            # axs[1].scatter(all_time[i],all_v[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            # axs[2].scatter(all_time[i],all_y[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            
            axs[0].plot(all_time[i],all_x[i],color = colors[cidx])#/(i%1+1))
            try:
                v = np.convolve(v,np.hamming(15),mode = "same")
                axs[1].plot(all_time[i],all_v[i],color = colors[cidx])#/(i%3+1))

            except:
                try:
                    v = np.convolve(v,np.hamming(5),mode = "same")
                    axs[1].plot(all_time[i],all_v[i],color = colors[cidx])#/(i%3+1))
                except:
                    axs[1].plot(all_time[i],all_v[i],color = colors[cidx])#/(i%3+1))
            
            all_x2 = [all_lengths[i] + item for item in all_x[i]]
            axs[0].fill_between(all_time[i],all_x[i],all_x2,color = colors[cidx])
            
            axs[1].set(xlabel='time(s)',ylabel='Velocity (ft/s)')
            axs[0].set(ylabel='X-pos (ft)')

            axs[1].set_ylim(-60,0)
        
        plt.show()  

    def replace_timestamps(self):
        """
        Replace timestamps with timestamps at nominal framerate. 
        Then reinterpolate boxes based on these timestamps
        """
        
        # get nominal framerate for each camera
        start_ts = [self.all_ts[0][key] for key in self.seq_keys]
        spans = [self.all_ts[-1][key] - self.all_ts[0][key] for key in self.seq_keys]
        frame_count = len(self.all_ts)
        
        fps_nom = [frame_count/spans[i] for i in range(len(spans))]
        print (fps_nom)
        
        # modify self.ts for each camera
        for i in range(len(self.all_ts)):
            for j in range(len(self.seq_keys)):
                key = self.seq_keys[j]
                self.all_ts[i][key] = start_ts[j] + 1.0/fps_nom[j]*i
        
        # ts bias dictionary
        #ts_bias_dict =  dict([ (self.seq_keys[i],self.ts_bias[i]) for i in range(len(self.ts_bias))])
        
        new_data = []
        # overwrite timestamps in all labels
        for f_idx in range(len(self.data)):
            new_data.append({})
            for key in self.data[f_idx]:
                cam = self.data[f_idx][key]["camera"]
                self.data[f_idx][key]["timestamp"] = self.all_ts[f_idx][cam] #+ ts_bias_dict[cam]
                
                if "gen" in self.data[f_idx][key].keys() and self.data[f_idx][key]["gen"] != "Manual":
                    continue
                new_data[f_idx][key] =  copy.deepcopy(self.data[f_idx][key])
        
        
        # delete and reinterpolate boxes as necessary
        self.data = new_data
        for i in range(self.get_unused_id()):
            self.interpolate(i,verbose = False)
        
        self.plot_all_trajectories()
        print("Replaced timestamps")
        
    
    def unbias_timestamps(self):
        """
        Replace timestamps with timestamps at nominal framerate. 
        Then reinterpolate boxes based on these timestamps
        """
        
        # get nominal framerate for each camera
        self.estimate_ts_bias()
        
        print(self.ts_bias)
        print(self.seq_keys)
        
        # ts bias dictionary
        ts_bias_dict =  dict([ (self.seq_keys[i],self.ts_bias[i]) for i in range(len(self.seq_keys))])
        
        new_data = []
        # overwrite timestamps in all labels
        for f_idx in range(len(self.data)):
            new_data.append({})
            for key in self.data[f_idx]:
                cam = self.data[f_idx][key]["camera"]
                try:
                    self.data[f_idx][key]["timestamp"] += ts_bias_dict[cam]
                except:
                    print("KeyError")
                
        # overwrite timestamps in self.all_ts
        for f_idx in range(len(self.all_ts)):
            for i in range(len(self.seq_keys)):
                key = self.seq_keys[i]
                try:
                    self.all_ts[f_idx][key] += self.ts_bias[i]
                except KeyError:
                    print("KeyError")
                    pass
                
        print("Un-biased timestamps")
            
    def replace_homgraphy(self):
        raise Exception ("Dont Call this!")
        
        # get replacement homography
        hid = 5
        with open("EB_homography{}.cpkl".format(hid),"rb") as f:
            hg1 = pickle.load(f) 
        with open("WB_homography{}.cpkl".format(hid),"rb") as f:
            hg2 = pickle.load(f) 
        hg_new  = Homography_Wrapper(hg1=hg1,hg2=hg2)    
    
        # # copy height scale values from old homography to new homography
        # for corr in hg_new.hg1.correspondence.keys():
        #     if corr in self.hg.hg1.correspondence.keys():
        #         hg_new.hg1.correspondence[corr]["P"][]
        # if direction == 1:
        #         self.hg.hg1.correspondence[self.clicked_camera]["P"][:,2] *= sign*delta
        #     else:   
        #         self.hg.hg2.correspondence[self.clicked_camera]["P"][:,2] *= sign*delta
        
        # create new copy of data
        new_data = []
        all_errors = [0]
        # for each frame in data
        for f_idx,frame_data in enumerate(self.data):
            
            if f_idx > self.last_frame:
                break
            
            new_data.append({})
            if f_idx % 100 == 0:
                print("On frame {}. Average error so far: {}".format(f_idx,sum(all_errors)/len(all_errors)))
            
            # for each camera in frame data
            for camera in self.cameras:
                cam = camera.name
                
                # for each box in camera 
                for obj_idx in range(self.get_unused_id()):
                   key = "{}_{}".format(cam,obj_idx)
                   if frame_data.get(key): 
                       obj = frame_data.get(key)
                       
                       # if box was manually drawn
                       if obj["gen"] == "Manual":
                           
                           base = copy.deepcopy(obj)
                           
                           # get old box image coordinates
                           old_box = torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"]]).unsqueeze(0)
                           old_box_im = self.hg.state_to_im(old_box,name = cam)
                           
                           # find new box that minimizes the reprojection error of corner coordinates
                           center = obj["x"],obj["y"]
                           search_rad = 50
                           grid_size = 11
                           while search_rad > 1:
                               x = np.linspace(center[0]-search_rad,center[0]+search_rad,grid_size)
                               y = np.linspace(center[1]-search_rad,center[1]+search_rad,grid_size)
                               shifts = []
                               for i in x:
                                   for j in y:
                                       shift_box = torch.tensor([i,j, base["l"],base["w"],base["h"],base["direction"]])
                                       shifts.append(shift_box)
                                    
                               # convert shifted grid of boxes into 2D space
                               shifts = torch.stack(shifts)
                               boxes_space = hg_new.state_to_im(shifts,name = cam)
                                 
                                
                                
                               # compute error between old_box_im and each shifted box footprints
                               box_expanded = old_box_im.repeat(boxes_space.shape[0],1,1)  
                               error = ((boxes_space[:,:4,:] - box_expanded[:,:4,:])**2).mean(dim = 2).mean(dim = 1)
                                
                               # find min_error and assign to center
                               min_idx = torch.argmin(error)
                               center = x[min_idx//grid_size],y[min_idx%grid_size]
                               search_rad /= 5
                                
                           # save box
                           min_err = error[min_idx].item()
                           all_errors.append(min_err)
                           base["x"] = self.safe(center[0])
                           base["y"] = self.safe(center[1])
                           
                           new_data[f_idx][key] = base
                           
                           # di = "EB" if obj["direction"] == 1 else "WB"
                           # print("Camera {}, {} obj {}: Error {}".format(cam,di,obj_idx,min_err))
                           
        
        
        # overwrite self.data with new_data
        self.data = new_data
        # overwrite self.hg with hg_new
        self.hg = hg_new
                
        # reinterpolate rest of data
        for i in range(self.get_unused_id()):
            self.interpolate(i,verbose = False)
        
        self.plot_all_trajectories()

            
    def replace_y(self,reverse  = False):
            
        # create new copy of data
        new_data = []
        
        # for each frame in data
        for f_idx,frame_data in enumerate(self.data):
            
            if f_idx > self.last_frame and len(self.data[f_idx]) == 0:
                break
            
            new_data.append({})
            if f_idx % 100 == 0:
                print("On frame {}.".format(f_idx))
            
            # for each camera in frame data
            for camera in self.cameras:
                cam = camera.name
                
                # for each box in camera 
                for obj_idx in range(self.get_unused_id()):
                   key = "{}_{}".format(cam,obj_idx)
                   if frame_data.get(key): 
                       obj = frame_data.get(key)
                       
                       # if box was manually drawn
                       
                       if "gen" not in obj.keys() or obj["gen"] == "Manual":
                           
                           base = copy.deepcopy(obj)
                           new_box = self.offset_box_y(base,reverse = reverse)
                           
                           new_data[f_idx][key] = new_box
                           
        # overwrite self.data with new_data
        self.data = new_data
        
        # reinterpolate rest of data
        for i in range(self.get_unused_id()):
            self.interpolate(i,verbose = False) 
            

        
        self.plot_all_trajectories()

            
    def offset_box_y(self,box,reverse = False):
        
        camera = box["camera"]
        direction = box["direction"]
        
        x = box["x"]
        
        
        direct =  "_EB" if direction == 1 else"_WB"
        key = camera + direct
        
        p2,p1,p0 = self.poly_params[key]
        
        y_offset = x**2*p2 + x*p1 + p0
        
        # if on the WB side, we need to account for the non-zero location of the leftmost line so we don't shift all the way back to near 0
        if direction == -1:
            y_straight_offset = self.hg.hg2.correspondence[camera]["space_pts"][0][1]
            y_offset -= y_straight_offset
            
        if not reverse:
            box["y"] -= y_offset
        else:
            box["y"] += y_offset

        
        return box
            
        
    def fit_curvature(self,box,min_pts = 4):   
        """
        Stores clicked points in array for each camera. If >= min_pts points have been clicked, after each subsequent clicked point the curvature lines are refit
        """
        
        point = self.box_to_state(box)[0]
        direction = "_EB" if point[1] < 60 else "_WB"

        # store point in curvature_points[cam_idx]
        self.curve_points[self.clicked_camera+direction].append(point)
        
        # if there are sufficient fitting points, recompute poly_params for active camera
        if len(self.curve_points[self.clicked_camera+direction]) >= min_pts:
            x_curve = np.array([self.safe(p[0]) for p in self.curve_points[self.clicked_camera+direction]])
            y_curve = np.array([self.safe(p[1]) for p in self.curve_points[self.clicked_camera+direction]])
            pparams = np.polyfit(x_curve,y_curve,2)
            print("Fit {} poly params for camera {}".format(self.clicked_camera,direction))
            self.poly_params[self.clicked_camera+direction] = pparams
            
            self.plot()

    
    def erase_curvature(self,box):
        """
        Removes all clicked curvature points for selected camera
        """
        point = self.box_to_state(box)[0]
        
        direction = "_EB" if point[1] < 60 else "_WB"
        # erase all points
        self.curve_points[self.clicked_camera+direction] = []

        # reset curvature polynomial coefficients to 0
        self.poly_params[self.clicked_camera+direction] = [0,0,0]
        
        self.plot()
    
    def estimate_ts_bias(self):
        """
        Moving sequentially through the cameras, estimate ts_bias of camera n
        relative to camera 0 (tsb_n = tsb relative to n-1 + tsb_n-1)
        - Find all objects that are seen in both camera n and n-1, and that 
        overlap in x-space
        - Sample p evenly-spaced x points from the overlap
        - For each point, compute the time for each camera tracklet for that object
        - Store the difference as ts_bias estimate
        - Average all ts_bias estimates to get ts_bias
        - For analysis, print statistics on the error estiamtes
        """
        
        self.ts_bias[0] = 0
        
        for cam_idx  in range(1,len(self.cameras)):
            cam = self.cameras[cam_idx].name
            decrement = 1
            while True:
                prev_cam = self.cameras[cam_idx-decrement].name
                
                diffs = []
                
                for obj_idx in range(self.get_unused_id()):
                    
                    # check whether object exists in both cameras and overlaps
                    c1x = []
                    c1t = []
                    c0x = []
                    c0t = []
                    
                    for frame_data in self.data:
                        key = "{}_{}".format(cam,obj_idx)
                        if frame_data.get(key):
                            obj = frame_data.get(key)
                            c1x.append(self.safe(obj["x"]))
                            c1t.append(self.safe(obj["timestamp"]))
                        
                        key = "{}_{}".format(prev_cam,obj_idx)
                        if frame_data.get(key):
                            obj = frame_data.get(key)
                            c0x.append(self.safe(obj["x"]))
                            c0t.append(self.safe(obj["timestamp"]))
                
                    if len(c0x) > 1 and len(c1x) > 1 and max(c0x) > min (c1x):
                        
                        # camera objects overlap from minx to maxx
                        minx = max(min(c1x),min(c0x))
                        maxx = min(max(c1x),max(c0x))
                        
                        # get p evenly spaced x points
                        p = 5
                        ran = maxx - minx
                        sample_points = []
                        for i in range(p):
                            point = minx + ran/(p-1)*i
                            sample_points.append(point)
                            
                        for point in sample_points:
                            time = None
                            prev_time = None
                            # estimate time at which cam object was at point
                            for i in range(1,len(c1x)):
                                if (c1x[i] - point) *  (c1x[i-1]- point) <= 0:
                                    ratio = (point-c1x[i-1])/ (c1x[i]-c1x[i-1]+ 1e-08)
                                    time = c1t[i-1] + (c1t[i] - c1t[i-1])*ratio
                            
                            # estimate time at which prev_cam object was at point
                            for j in range(1,len(c0x)):
                                if (c0x[j] - point) *  (c0x[j-1]- point) <= 0:
                                    ratio = (point-c0x[j-1])/ (c0x[j]-c0x[j-1] + 1e-08)
                                    prev_time = c0t[j-1] + (c0t[j] - c0t[j-1])*ratio
                            
                            # relative to previous camera, cam time is diff later when object is at same location
                            if time and prev_time:
                                diff = self.safe(time - prev_time)
                                #diff = np.sign(diff) * np.power(diff,2)
                                diffs.append(diff)
                
                # after all objects have been considered
                if len(diffs) > 0:
                    diffs = np.array(diffs)
                    #avg_diff = np.sqrt(np.abs(np.mean(diffs))) * np.sign(np.mean(diffs))
                    avg_diff = np.mean(diffs)
                    stdev = np.std(diffs)
                    
                    # since diff is positive if camera clock is ahead, we subtract it such that adding ts_bias to camera timestamps corrects the error
                    abs_bias = self.ts_bias[cam_idx-decrement] -avg_diff
                    
                    print("Camera {} ofset relative to camera {}: {}s ({}s absolute)".format(cam,prev_cam,avg_diff,abs_bias))
                    self.ts_bias[cam_idx] = abs_bias
                    
                    break
            
                else:
                    
                    print("No matching points for cameras {} and {}".format(cam,prev_cam))
                    decrement += 1
                    if cam_idx - decrement >= 0:
                        prev_cam = self.cameras[cam_idx-2].name
                    else:
                        break
        print("Done)")
    
                
    def est_y_error(self):
        """
        Moving sequentially through the cameras, estimate y-error of camera n
        - Find all objects that are seen in both camera n and n-1, and that 
        overlap in x-space
        - Sample p evenly-spaced x points from the overlap
        - Store the difference in y
        - Average all ts_bias estimates to get ts_bias
        - For analysis, print statistics on the error estiamtes
        """
        
        all_diffs = [] 
        
        for cam_idx  in range(1,len(self.cameras)):
            cam = self.cameras[cam_idx].name
            prev_cam = self.cameras[cam_idx-1].name
            
            diffs = []
            
            for obj_idx in range(self.get_unused_id()):
                
                # check whether object exists in both cameras and overlaps
                c1x = []
                c1y = []
                c0x = []
                c0y = []
                
                for frame_data in self.data:
                    key = "{}_{}".format(cam,obj_idx)
                    if frame_data.get(key):
                        obj = frame_data.get(key)
                        c1x.append(self.safe(obj["x"]))
                        c1y.append(self.safe(obj["y"]))
                    
                    key = "{}_{}".format(prev_cam,obj_idx)
                    if frame_data.get(key):
                        obj = frame_data.get(key)
                        c0x.append(self.safe(obj["x"]))
                        c0y.append(self.safe(obj["y"]))
            
                if len(c0x) > 1 and len(c1x) > 1 and max(c0x) > min (c1x):
                    
                    # camera objects overlap from minx to maxx
                    minx = max(min(c1x),min(c0x))
                    maxx = min(max(c1x),max(c0x))
                    
                    # get p evenly spaced x points
                    p = 5
                    ran = maxx - minx
                    sample_points = []
                    for i in range(p):
                        point = minx + ran/(p-1)*i
                        sample_points.append(point)
                        
                    for point in sample_points:
                        
                        # estimate time at which cam object was at point
                        for i in range(1,len(c1x)):
                            if (c1x[i] - point) *  (c1x[i-1]- point) <= 0:
                                ratio = (point-c1x[i-1])/ (c1x[i]-c1x[i-1]+ 1e-08)
                                y1 = c1y[i-1] + (c1y[i] - c1y[i-1])*ratio
                        
                        # estimate time at which prev_cam object was at point
                        for j in range(1,len(c0x)):
                            if (c0x[j] - point) *  (c0x[j-1]- point) <= 0:
                                ratio = (point-c0x[j-1])/ (c0x[j]-c0x[j-1] + 1e-08)
                                y2 = c0y[j-1] + (c0y[j] - c0y[j-1])*ratio
                        

                        diff = np.abs(self.safe(y2-y1))
                        diffs.append(diff)
        
            # after all objects have been considered
            if len(diffs) > 0:
                diffs = np.array(diffs)
                avg_diff = np.mean(diffs)
                stdev = np.std(diffs)
                
                # since diff is positive if camera clock is ahead, we subtract it such that adding ts_bias to camera timestamps corrects the error
                
                print("Camera {} and {} average y-error: {}ft ({})ft stdev".format(cam,prev_cam,avg_diff,stdev))
                all_diffs.append(avg_diff)
            
            else:
                print("No matching points for cameras {} and {}".format(cam,prev_cam))
        if len(all_diffs) > 0:
            print("Average y-error over all cameras: {}".format(sum(all_diffs)/len(all_diffs)))
        
        
    
    def save2(self):
        with open("chg_labels_sequence_{}.cpkl".format(self.scene_id),"wb") as f:
            data = [self.data,self.all_ts,self.ts_bias,self.chg]
            pickle.dump(data,f)
            print("Saved labels")
            self.count()
            
    def reload_chg_data(self):
        with open("chg_labels_sequence_{}.cpkl".format(self.scene_id),"rb") as f:
            self.data,self.all_ts,self.ts_bias,self.chg = pickle.load(f)
            print("Reloaded data")
        
    def reload(self):
        try:
            with open("labeler_cache_sequence_{}.cpkl".format(self.scene_id),"rb") as f:
                self.data,self.all_ts,self.ts_bias,self.hg,self.poly_params = pickle.load(f)
                self.curve_points = dict([(camera.name+"_EB",[]) for camera in self.cameras]+[(camera.name+"_WB",[]) for camera in self.cameras])
            
        except:
            with open("labeler_cache_sequence_{}.cpkl".format(self.scene_id),"rb") as f:
                self.data,self.all_ts,self.ts_bias,self.hg = pickle.load(f)
                self.poly_params = dict([(camera.name+"_EB",[0,0,0]) for camera in self.cameras]+[(camera.name+"_WB",[0,0,0]) for camera in self.cameras])
                self.curve_points = dict([(camera.name+"_EB",[]) for camera in self.cameras]+[(camera.name+"_WB",[]) for camera in self.cameras])
    
    def save(self):
        outfile = "working_3D_tracking_data.csv"
        
        data_header = [
            "Frame #",
            "Timestamp",
            "Object ID",
            "Object class",
            "BBox xmin",
            "BBox ymin",
            "BBox xmax",
            "BBox ymax",
            "vel_x",
            "vel_y",
            "Generation method",
            "fbrx",
            "fbry",
            "fblx",
            "fbly",
            "bbrx",
            "bbry",
            "bblx",
            "bbly",
            "ftrx",
            "ftry",
            "ftlx",
            "ftly",
            "btrx",
            "btry",
            "btlx",
            "btly",
            "fbr_x",
            "fbr_y",
            "fbl_x",
            "fbl_y",
            "bbr_x",
            "bbr_y",
            "bbl_x",
            "bbl_y",
            "direction",
            "camera",
            "acceleration",
            "speed",
            "veh rear x",
            "veh center y",
            "theta",
            "width",
            "length",
            "height",
            "ts_bias for cameras {}".format(self.seq_keys)
            ]

        
        
        
        with open(outfile, mode='w') as f:
            out = csv.writer(f, delimiter=',')
            
            # write main chunk
            out.writerow(data_header)
            #print("\n")
            gen = "3D Detector"
            
            for i,ts_data in enumerate(self.data):
                print("\rWriting outputs for time {} of {}".format(i,len(self.data)), end = '\r', flush = True)

                for item in ts_data.values():
                    id = item["id"]
                    timestamp = item["timestamp"]
                    cls = item["class"]
                    
                    try:
                        camera = item["camera"]
                    except:
                        camera = "p1c1"
                    
                    ts_bias = [t for t in self.ts_bias]
                    state = torch.tensor([item["x"],item["y"],item["l"],item["w"],item["h"],item["direction"],0]) # vel = 0 assumption
                            
                        
                    state = state.float()
                    
                    if state[0] != 0:
                        
                        # generate space coords
                        space = self.hg.state_to_space(state.unsqueeze(0))
                        space = space.squeeze(0)[:4,:2]
                        flat_space = list(space.reshape(-1).data.numpy())
                        
                        # generate im coords
                        bbox_3D = self.hg.state_to_im(state.unsqueeze(0),name = camera)
                        flat_3D = list(bbox_3D.squeeze(0).reshape(-1).data.numpy())
                        
                        # generate im 2D bbox
                        minx = torch.min(bbox_3D[:,:,0],dim = 1)[0].item()
                        maxx = torch.max(bbox_3D[:,:,0],dim = 1)[0].item()
                        miny = torch.min(bbox_3D[:,:,1],dim = 1)[0].item()
                        maxy = torch.max(bbox_3D[:,:,1],dim = 1)[0].item()
                        
                        
                        obj_line = []
                        
                        obj_line.append(i) # frame number is not useful in this data
                        obj_line.append(timestamp)
                        obj_line.append(id)
                        obj_line.append(cls)
                        obj_line.append(minx)
                        obj_line.append(miny)
                        obj_line.append(maxx)
                        obj_line.append(maxy)
                        obj_line.append(0)
                        obj_line.append(0)
    
                        obj_line.append(gen)
                        obj_line = obj_line + flat_3D + flat_space 
                        state = state.data.numpy()
                        obj_line.append(state[5])
                        
                        obj_line.append(camera)
                        
                        obj_line.append(0) # acceleration = 0 assumption
                        obj_line.append(state[6])
                        obj_line.append(state[0])
                        obj_line.append(state[1])
                        obj_line.append(np.pi/2.0 if state[5] == -1 else 0)
                        obj_line.append(state[3])
                        obj_line.append(state[2])
                        obj_line.append(state[4])
    
    
                        obj_line.append(ts_bias)
                        out.writerow(obj_line)
    
    def remove_outliers(self):
        outliers= 0
        # for each frame in data
        for f_idx,frame_data in enumerate(self.data):
            print(f_idx)
            if f_idx > self.last_frame and len(self.data[f_idx]) == 0:
                break
            
            for camera in self.cameras:
                cam = camera.name
                
                # for each box in camera 
                for obj_idx in range(self.get_unused_id()):
                    key = "{}_{}".format(cam,obj_idx)
                    if frame_data.get(key): 
                        obj = frame_data.get(key)
                      
                   
                        try:
                           prev_obj = self.data[f_idx-1][key]
                        except:
                            prev_obj = None
                         
                        try:
                            next_obj = self.data[f_idx + 1][key]
                        except:
                             next_obj = None
                             
                        if prev_obj is not None:
                             if (np.abs(obj["x"] - prev_obj["x"] )> 15 or np.abs(obj["y"] - prev_obj["y"] > 5)):
                                 del self.data[f_idx][key]
                                 outliers += 1
                                 print("Removed outlier")
                                 continue
                                 
                        if next_obj is not None:
                             if (np.abs(obj["x"] - next_obj["x"] )> 15 or np.abs(obj["y"] - next_obj["y"] > 5)):
                                 del self.data[f_idx][key]
                                 print("Removed outlier")
                                 outliers += 1
                                 
        print("Removed {} outlier data points".format(outliers))
      
    def count_classes(self):
        classes = {}
        for frame_data in self.data:
            for item in frame_data.values():                  
                cls = item["class"]
                try:
                    classes[cls] += 1
                except KeyError:
                    classes[cls] = 1
        
        print("Class counts:")
        for key in classes:
            print("{}:{}".format(key,classes[key]))

    def estimate_projection_error(self,reverse_curve_offset = False):
        """
        Moving sequentially through the cameras, estimate ts_bias of camera n
        relative to camera 0 (tsb_n = tsb relative to n-1 + tsb_n-1)
        - Find all objects that are seen in both camera n and n-1, and that 
        overlap in x-space
        - Sample p evenly-spaced x points from the overlap
        - For each point, compute the time for each camera tracklet for that object
        - Store the difference as ts_bias estimate
        - Average all ts_bias estimates to get ts_bias
        - For analysis, print statistics on the error estiamtes
        """
        
        print("Not Implemented")
        
        x_errors = []        
        y_errors = []
        p_errors = []
        
        for cam_idx  in range(1,len(self.cameras)):
            cam = self.cameras[cam_idx].name
            decrement = 1
            while True and decrement < cam_idx:
                prev_cam = self.cameras[cam_idx-decrement].name
                
                diffs = []
                
                for obj_idx in range(self.get_unused_id()):
                    
                    # check whether object exists in both cameras and overlaps
                    c1x = []
                    c1t = []
                    c0x = []
                    c0t = []
                    c0y = []
                    c1y = []
                    
                    for frame_data in self.data:
                        key = "{}_{}".format(cam,obj_idx)
                        if frame_data.get(key):
                            obj = frame_data.get(key)
                            c1x.append(self.safe(obj["x"]))
                            c1y.append(self.safe(obj["y"]))
                            c1t.append(self.safe(obj["timestamp"]))
                        
                        key = "{}_{}".format(prev_cam,obj_idx)
                        if frame_data.get(key):
                            obj = frame_data.get(key)
                            c0x.append(self.safe(obj["x"]))
                            c0y.append(self.safe(obj["y"]))
                            c0t.append(self.safe(obj["timestamp"]))
                
                    if len(c0x) > 1 and len(c1x) > 1 and max(c0t) > min (c1t):
                        
                        # camera objects overlap from minx to maxx
                        mint = max(min(c1t),min(c0t))
                        maxt = min(max(c1t),max(c0t))
                        
                        # get p evenly spaced t points
                        p = 5
                        ran = maxt - mint
                        sample_points = []
                        for i in range(p):
                            point = mint + ran/(p-1)*i
                            sample_points.append(point)
                            
                        for point in sample_points:
                            cur_x  = None
                            prev_x = None
                            cur_y  = None
                            prev_y = None
                            # estimate time at which cam object was at point
                            for i in range(1,len(c1x)):
                                if (c1t[i] - point) *  (c1t[i-1]- point) <= 0:
                                    ratio = (point-c1t[i-1])/ (c1t[i]-c1t[i-1]+ 1e-08)
                                    cur_x = c1x[i-1] + (c1x[i] - c1x[i-1])*ratio
                                    cur_y = c1y[i-1] + (c1y[i] - c1y[i-1])*ratio
                            
                            # estimate time at which prev_cam object was at point
                            for j in range(1,len(c0t)):
                                if (c0t[j] - point) *  (c0t[j-1]- point) <= 0:
                                    ratio = (point-c0t[j-1])/ (c0t[j]-c0t[j-1] + 1e-08)
                                    prev_x = c0x[j-1] + (c0x[j] - c0x[j-1])*ratio
                                    prev_y = c0y[j-1] + (c0y[j] - c0y[j-1])*ratio

                            # relative to previous camera, cam time is diff later when object is at same location
                            if cur_x and prev_x:
                                
                                if reverse_curve_offset:
                                    direction = "EB" if cur_y < 60 else "WB"
                                    try:
                                        key = "{}_{}".format(cam,direction)
                                        p2,p1,p0 = self.poly_params[key]
                                        cur_y -= p0 +p1*cur_x + p2*cur_x**2
                                    except:
                                        pass
                                        print("Error 1458 search for it")

                                    try:
                                        key = "{}_{}".format(prev_cam,direction)
                                        p2,p1,p0 = self.poly_params[key]
                                        prev_y -= p0 +p1*prev_x + p2*prev_x**2
                                    except:
                                        pass
                                        print("Error 1458 search for it")
                                    
                                x_errors.append(np.abs(cur_x - prev_x))
                                y_errors.append(np.abs(cur_y - prev_y))
                                
                                # convert each point into each plane to determine how far apart they are
                                points = torch.tensor([[cur_x,cur_y,0,1],[prev_x,prev_y,0,1]])
                                for cm in [cam,prev_cam]:
                                    if cur_y > 60:
                                        P = torch.from_numpy(self.hg.hg2.correspondence[cm]["P"]).float()
                                    else:
                                        P = torch.from_numpy(self.hg.hg1.correspondence[cm]["P"]).float()
                                        
                                new_pts = torch.matmul(P,points.transpose(0,1)).transpose(0,1)
                                # divide each point 0th and 1st column by the 2nd column
                                new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
                                new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
                                px_diff = torch.sqrt((new_pts[0,0] - new_pts[1,0]).pow(2) + (new_pts[0,1] - new_pts[1,1]).pow(2))
                                p_errors.append(px_diff.item())
            
                                #print("{} {} projection error: {} ft".format(cam,prev_cam,diff))
                    decrement += 1
                
        return x_errors,y_errors,p_errors

    def estimate_transform_reprojection_error(self,use_curvature = False):
        x_errors = []
        y_errors = []
        
        print("Not implemented")

        # for each point in each hg transform
        direction = "EB"
        for corr in self.hg.hg1.correspondence:
            im_pts = self.hg.hg1.correspondence[corr]["corr_pts"]
            space_pts = self.hg.hg1.correspondence[corr]["space_pts"]
            
            
            # convert the pixel coordinate into space
            im_pts = torch.tensor(im_pts).unsqueeze(1).repeat(1,8,1)
            heights = torch.zeros(im_pts.shape[0])
            im_proj = self.hg.hg1.im_to_space(im_pts,name = corr,heights = heights)[:,0,:]
            
            # apply curvature correction if necessary
            if use_curvature:
                try:
                    key = "{}_{}".format(corr,direction)
                    p2,p1,p0 = self.poly_params[key]
                    im_proj[:,1] += im_proj[:,0].pow(2)*p2 + im_proj[:,0]*p1 + p0
                
                except:
                    pass
                
            # get difference from supposed coordinate        
            space_pts = torch.tensor(space_pts)
            x_diff = torch.abs(space_pts[:,0] - im_proj[:,0])
            y_diff = torch.abs(space_pts[:,1] - im_proj[:,1])
            
            if y_diff.max() > 20: # uses the wrong side hg I think
                print("skipped points for {}".format(corr))
                continue
            
            x_diff = [val.item() for val in x_diff]
            y_diff = [val.item() for val in y_diff]
            
            x_errors += x_diff
            y_errors += y_diff
            
            
        direction = "WB"
        for corr in self.hg.hg2.correspondence:
            im_pts = self.hg.hg2.correspondence[corr]["corr_pts"]
            space_pts = self.hg.hg2.correspondence[corr]["space_pts"]
            
            
            # convert the pixel coordinate into space
            im_pts = torch.tensor(im_pts).unsqueeze(1).repeat(1,8,1)
            heights = torch.zeros(im_pts.shape[0])
            im_proj = self.hg.hg2.im_to_space(im_pts,name = corr,heights = heights).squeeze(1)[:,0,:]
            
            # apply curvature correction if necessary
            if use_curvature:
                try:
                    key = "{}_{}".format(corr,direction)
                    p2,p1,p0 = self.poly_params[key]
                    im_proj[:,1] += im_proj[:,0].pow(2)*p2 + im_proj[:,0]*p1 + p0  - space_pts[0][1]
                
                except:
                    pass
                
            # get difference from supposed coordinate        
            space_pts = torch.tensor(space_pts)
            x_diff = torch.abs(space_pts[:,0] - im_proj[:,0])
            y_diff = torch.abs(space_pts[:,1] - im_proj[:,1])
            
            if y_diff.max() > 20: # uses the wrong side hg I think
                print("skipped points for {}".format(corr))
                continue
            
            # convert to list
            x_diff = [val.item() for val in x_diff]
            y_diff = [val.item() for val in y_diff]
            
            x_errors += x_diff
            y_errors += y_diff
            
        #print(y_errors)
        return [x_errors,y_errors]
    
    def estimate_transform_reprojection_error2(self,use_curvature = False):
        x_errors = []
        y_errors = []
        p_errors = []

        # for each point in each hg transform
        direction = "EB"
        for i in range(len(self.hg.hg1.correspondence)):
            for j in range(i+1,len(self.hg.hg1.correspondence)):
                corr1 = list(self.hg.hg1.correspondence.keys())[i]
                corr2 = list(self.hg.hg1.correspondence.keys())[j]
                
                im_pts = self.hg.hg1.correspondence[corr1]["corr_pts"]
                space_pts = self.hg.hg1.correspondence[corr1]["space_pts"]
            
                # convert the pixel coordinate into space
                im_pts = torch.tensor(im_pts).unsqueeze(1).repeat(1,8,1)
                heights = torch.zeros(im_pts.shape[0])
                im_proj = self.hg.hg1.im_to_space(im_pts,name = corr1,heights = heights)[:,0,:]
            
                # apply curvature correction if necessary
                if use_curvature:
                    try:
                        key = "{}_{}".format(corr1,direction)
                        p2,p1,p0 = self.poly_params[key]
                        im_proj[:,1] += im_proj[:,0].pow(2)*p2 + im_proj[:,0]*p1 + p0
                    
                    except:
                        pass
                    
                # repeat for corr2 
                im_pts2 = self.hg.hg1.correspondence[corr2]["corr_pts"]
                space_pts2 = self.hg.hg1.correspondence[corr2]["space_pts"]
            
                # convert the pixel coordinate into space
                im_pts2 = torch.tensor(im_pts2).unsqueeze(1).repeat(1,8,1)
                heights = torch.zeros(im_pts2.shape[0])
                im_proj2 = self.hg.hg1.im_to_space(im_pts2,name = corr2,heights = heights)[:,0,:]
            
                # apply curvature correction if necessary
                if use_curvature:
                    try:
                        key = "{}_{}".format(corr2,direction)
                        p2,p1,p0 = self.poly_params[key]
                        im_proj2[:,1] += im_proj2[:,0].pow(2)*p2 + im_proj2[:,0]*p1 + p0
                    
                    except:
                        pass
              
                # get difference from supposed coordinate        
                for i2 in range(len(space_pts)):
                    for j2 in range(len(space_pts2)):
                        if np.abs(space_pts[i2,0] - space_pts2[j2,0]) < 1 and np.abs(space_pts[i2,1] - space_pts2[j2,1]) < 1:
                            x_diff = torch.abs(im_proj2[j2,0] - im_proj[i2,0])
                            y_diff = torch.abs(im_proj2[j2,1] - im_proj[i2,1])
                            
                            if y_diff > 20: # uses the wrong side hg I think
                                continue
                            x_errors.append(x_diff.item())
                            y_errors.append(y_diff.item())
                            
                            # convert each point into each plane to determine how far apart they are
                            points = torch.tensor([[im_proj2[j2,0],im_proj2[j2,1],0,1],[im_proj[i2,0],im_proj[i2,1],0,1]])
                            for cm in [corr1,corr2]:
                                    P = torch.from_numpy(self.hg.hg1.correspondence[cm]["P"])
                                    
                            new_pts = torch.matmul(P,points.transpose(0,1)).transpose(0,1)
                            # divide each point 0th and 1st column by the 2nd column
                            new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
                            new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
                            px_diff = torch.sqrt((new_pts[0,0] - new_pts[1,0]).pow(2) + (new_pts[0,1] - new_pts[1,1]).pow(2))
                            p_errors.append(px_diff.item())
            
            
            
        # for each point in each hg transform
        direction = "WB"
        for i in range(len(self.hg.hg2.correspondence)):
            for j in range(i+1,len(self.hg.hg2.correspondence)):
                corr1 = list(self.hg.hg1.correspondence.keys())[i]
                corr2 = list(self.hg.hg1.correspondence.keys())[j]
                
                im_pts = self.hg.hg2.correspondence[corr1]["corr_pts"]
                space_pts = self.hg.hg2.correspondence[corr1]["space_pts"]
            
                # convert the pixel coordinate into space
                im_pts = torch.tensor(im_pts).unsqueeze(1).repeat(1,8,1)
                heights = torch.zeros(im_pts.shape[0])
                im_proj = self.hg.hg2.im_to_space(im_pts,name = corr1,heights = heights)[:,0,:]
            
                # apply curvature correction if necessary
                if use_curvature:
                    try:
                        key = "{}_{}".format(corr1,direction)
                        p2,p1,p0 = self.poly_params[key]
                        im_proj[:,1] += im_proj[:,0].pow(2)*p2 + im_proj[:,0]*p1 + p0
                    
                    except:
                        pass
                    
                # repeat for corr2 
                im_pts2 = self.hg.hg2.correspondence[corr2]["corr_pts"]
                space_pts2 = self.hg.hg2.correspondence[corr2]["space_pts"]
            
                # convert the pixel coordinate into space
                im_pts2 = torch.tensor(im_pts2).unsqueeze(1).repeat(1,8,1)
                heights = torch.zeros(im_pts2.shape[0])
                im_proj2 = self.hg.hg2.im_to_space(im_pts2,name = corr2,heights = heights)[:,0,:]
            
                # apply curvature correction if necessary
                if use_curvature:
                    try:
                        key = "{}_{}".format(corr2,direction)
                        p2,p1,p0 = self.poly_params[key]
                        im_proj2[:,1] += im_proj2[:,0].pow(2)*p2 + im_proj2[:,0]*p1 + p0
                    
                    except:
                        pass
              
                # get difference from supposed coordinate        
                for i2 in range(len(space_pts)):
                    for j2 in range(len(space_pts2)):
                        if np.abs(space_pts[i2,0] - space_pts2[j2,0]) < 1 and np.abs(space_pts[i2,1] - space_pts2[j2,1]) < 1:
                            x_diff = torch.abs(im_proj2[j2,0] - im_proj[i2,0])
                            y_diff = torch.abs(im_proj2[j2,1] - im_proj[i2,1])
                            
                            if y_diff > 20: # uses the wrong side hg I think
                                continue
                            x_errors.append(x_diff.item())
                            y_errors.append(y_diff.item())
                            
                            # convert each point into each plane to determine how far apart they are
                            points = torch.tensor([[im_proj2[j2,0],im_proj2[j2,1],0,1],[im_proj[i2,0],im_proj[i2,1],0,1]])
                            for cm in [corr1,corr2]:
                                    P = torch.from_numpy(self.hg.hg2.correspondence[cm]["P"])
                                    
                            new_pts = torch.matmul(P,points.transpose(0,1)).transpose(0,1)
                            # divide each point 0th and 1st column by the 2nd column
                            new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
                            new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
                            px_diff = torch.sqrt((new_pts[0,0] - new_pts[1,0]).pow(2) + (new_pts[0,1] - new_pts[1,1]).pow(2))
                            p_errors.append(px_diff.item())
                            
        return x_errors,y_errors,p_errors
      
        
    def count_extended_data(self,extension_distance = 200):
        # get min and max range for camera objects
        ranges = {}
        extended_boxes = dict([(camera.name,0) for camera in self.cameras])
        
        for cam in self.cameras:
            cam = cam.name
            
            space_pts1 = self.hg.hg1.correspondence[cam]["space_pts"]
            space_pts2 = self.hg.hg2.correspondence[cam]["space_pts"]
            space_pts = np.concatenate((space_pts1,space_pts2),axis = 0)
    
            minx = np.min(space_pts[:,0])
            maxx = np.max(space_pts[:,0])
            
            ranges[cam] = [minx,maxx]
            
        
        for f_idx in range(len(self.data)):
            if f_idx % 100 == 0:
                print("On frame {}".format(f_idx))
            
            if len(self.data[f_idx]) == 0:
                break
            
            for camera in self.cameras:
                cam = camera.name
                if "c4" in cam or "c3" in cam: # cross-lane stuff is no good for these
                    continue
                
                ts_data = list(self.data[self.frame_idx].values())
                ts_data = list(filter(lambda x: x["camera"] == cam,ts_data))
                ids = [item["id"] for item in ts_data]
                
                if len(ts_data) == 0: # need to get timestamp for frame to extend
                    continue
                time = ts_data[0]["timestamp"]
                    
                
                ts_data2 = list(self.data[self.frame_idx].values())
                keep_boxes = []
                keep_ids = []
                
                for obj in ts_data2:
                   id = obj["id"]
                   if id in ids or id in keep_ids:
                       continue
                   x_spline,y_spline = self.splines[id]
                   if x_spline is None or y_spline is None:
                       continue
                   else:
                       obj["x"] = x_spline([time])[0]
                       obj["y"] = y_spline([time])[0]
                       
                       if obj["x"] > ranges[cam][0] - extension_distance and obj["x"] < ranges[cam][1] + extension_distance:
                           keep_boxes.append(obj)
                           keep_ids.append(id)
                if len(keep_boxes) > 0:
                   boxes2 = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"]]).float() for obj in keep_boxes])
                   im_boxes2 = self.hg.state_to_im(boxes2,name = camera.name)
                   
                   keep = (torch.max(im_boxes2[:,:,0],dim = 1)[0] < 1920).int() * (torch.min(im_boxes2[:,:,0],dim = 1)[0] > 0).int() * (torch.max(im_boxes2[:,:,1],dim = 1)[0] < 1080).int() * (torch.max(im_boxes2[:,:,0],dim = 1)[1] > 0).int()
                   boxes2 = boxes2[keep.nonzero().squeeze(1)]
                   if len(boxes2) > 0:
                       extended_boxes[cam] += boxes2.shape[0]
           
        return extended_boxes    

        
        # for each frame
        
        # for each camera
        
        # get the set of all objects
        
        # get the set of objects in that camera and remove
        
        
        # of the remaining objects, get spline position for each
        
        # remove objects outside of extension range
        
        # convert to image space
        
        # verify that at least 4 corners fall within camera
        
        # increment counter
    

    def run(self):
        """
        Main processing loop
        """
        
        cv2.namedWindow("window")
        cv.setMouseCallback("window", self.on_mouse, 0)
        self.plot()
        
        while(self.cont): # one frame
            
           ### handle click actions
           
           if self.new is not None:
                # buffer one change
                self.label_buffer = copy.deepcopy(self.data[self.frame_idx])
                
                # Add and delete objects
                if self.active_command == "DELETE":
                    obj_idx = self.find_box(self.new)
                    try:
                        n_frames = int(self.keyboard_input())
                    except:
                        n_frames = -1    
                    self.delete(obj_idx,n_frames = n_frames)
                    
                elif self.active_command == "ADD":
                    # get obj_idx
                    obj_idx = self.get_unused_id()
                    self.add(obj_idx,self.new)
                
                # Shift object
                elif self.active_command == "SHIFT":
                    obj_idx = self.find_box(self.new)
                    self.shift(obj_idx,self.new)
                
                # Adjust object dimensions
                elif self.active_command == "DIMENSION":
                    obj_idx = self.find_box(self.new)
                    self.dimension(obj_idx,self.new)
                   
                # copy and paste a box across frames
                elif self.active_command == "COPY PASTE":
                    self.copy_paste(self.new)
                    
                # interpolate between copy-pasted frames
                elif self.active_command == "INTERPOLATE":
                    obj_idx = self.find_box(self.new)
                    self.interpolate(obj_idx)  

                # correct vehicle class
                elif self.active_command == "VEHICLE CLASS":
                    obj_idx = self.find_box(self.new)
                    try:
                        cls = (self.keyboard_input())  
                    except:
                        cls = "midsize"
                    self.change_class(obj_idx,cls)

                # adjust time bias
                elif self.active_command == "TIME BIAS":
                    self.correct_time_bias(self.new)
                    
                # adjust homography
                elif self.active_command == "HOMOGRAPHY":
                    self.correct_homography_Z(self.new)
                
                elif self.active_command == "2D PASTE":
                    self.paste_in_2D_bbox(self.new)
                    
                elif self.active_command == "CURVE":
                    self.fit_curvature(self.new)
                    
                elif self.active_command == "ERASE CURVE":
                    self.erase_curvature(self.new)
                    
                self.plot()

                self.new = None   
                

           
           ### Show frame
                
           #self.cur_frame = cv2.resize(self.cur_frame,(1920,1080))
           cv2.imshow("window", self.plot_frame)
           title = "{} {}     Frame {}/{}, Cameras {} and {}".format("R" if self.right_click else "",self.active_command,self.frame_idx,self.max_frames,self.seq_keys[self.active_cam],self.seq_keys[self.active_cam + 1])
           cv2.setWindowTitle("window",str(title))
           
           
           ### Handle keystrokes 
           key = cv2.waitKey(1)

           
           if key == ord('9'):
                self.next()
                self.plot()
           elif key == ord('8'):
                self.prev()  
                self.plot()
           elif key == ord("q"):
               self.quit()
           elif key == ord("w"):
               self.save2()
               self.plot_all_trajectories()
           elif key == ord("@"):
               self.toggle_auto = not(self.toggle_auto)
               print("Automatic box pasting: {}".format(self.toggle_auto))
               
           elif key == ord("["):
               self.toggle_cams(-1)
           elif key == ord("]"):
               self.toggle_cams(1)
               
           elif key == ord("u"):
               self.undo()
           elif key == ord("-"):
                [self.prev() for i in range(self.stride)]
                self.plot()
           elif key == ord("="):
                [self.next() for i in range(self.stride)]
                self.plot()
           elif key == ord("+"):
               print("Filling buffer. Type number of frames to buffer...")
               n = int(self.keyboard_input())  
               self.fill_buffer(n)
               
           elif key == ord("?"):
               self.estimate_ts_bias()
               self.plot_all_trajectories()
           elif key == ord("t"):
               self.TEXT = not self.TEXT
               self.plot()               
           elif key == ord("l"):
               self.LANES = not self.LANES
               self.plot()
           elif key == ord("m"):
               self.MASK = not self.MASK
               self.plot()
               
           elif key == ord("p"):
               
               try:
                   n = int(self.keyboard_input())
               except:
                   n = self.plot_idx
               self.plot_trajectory(obj_idx = n)
               self.plot_idx = n + 1
               
           # toggle commands
           elif key == ord("a"):
               self.active_command = "ADD"
           elif key == ord("r"):
               self.active_command = "DELETE"
           elif key == ord("s"):
               self.active_command = "SHIFT"
           elif key == ord("d"):
               self.active_command = "DIMENSION"
           elif key == ord("c"):
               self.active_command = "COPY PASTE"
           elif key == ord("i"):
               self.active_command = "INTERPOLATE"
           elif key == ord("v"):
               self.active_command = "VEHICLE CLASS"
           elif key == ord("t"):
               self.active_command = "TIME BIAS"
           elif key == ord("h"):
               self.active_command = "HOMOGRAPHY"
           # elif key == ord("p"):
           #     self.active_command = "2D PASTE"
           elif key == ord("*"):
               self.active_command = "CURVE"
           elif key == ord("&"):
               self.active_command = "ERASE CURVE"
               
          
           elif self.active_command == "COPY PASTE" and self.copied_box:
               nudge = 0.25
               if key == ord("1"):
                   self.shift(self.copied_box[0],None,dx = -nudge)
                   self.plot()
               if key == ord("5"):
                   self.shift(self.copied_box[0],None,dy =  nudge)
                   self.plot()
               if key == ord("3"):
                   self.shift(self.copied_box[0],None,dx =  nudge)
                   self.plot()
               if key == ord("2"):
                   self.shift(self.copied_box[0],None,dy = -nudge)
                   self.plot()
            
           elif self.active_command == "DIMENSION" and self.copied_box:
               nudge = 0.1
               if key == ord("1"):
                   self.dimension(self.copied_box[0],None,dx = -nudge*2)
                   self.plot()
               if key == ord("5"):
                   self.dimension(self.copied_box[0],None,dy =  nudge)
                   self.plot()
               if key == ord("3"):
                   self.dimension(self.copied_box[0],None,dx =  nudge*2)
                   self.plot()
               if key == ord("2"):
                   self.dimension(self.copied_box[0],None,dy = -nudge)
                   self.plot() 
  
            
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
           
#%%
def plot_proj_error(all_errors,bins = 100,cutoff_error = 20,names = []):
    plt.figure(figsize = (7,5))

    colors = np.array([[0,0,1],[1,0,0],[0,0.5,0.7],[0.7,0.5,0],[0.5,0.5,0],[0.5,0.5,0]])
    xmeans = []
    ymeans = []
    count_max = []
    for i,errors in enumerate(all_errors): 
        minv = 0
        maxv = max(errors)
        cutoff = cutoff_error
        
        ran = np.linspace(minv,min(cutoff,maxv),100)
        count = np.zeros(ran.shape)
        
        for item in errors:
            binned = False
            for r in range(1,len(ran)):
                if item > ran[r-1] and item < ran[r]:
                    count[r-1] += 1
                    binned = True
                    break
            if not binned:
                count[-1] += 1
        count /= len(errors)

        # remove all 0 bins
        ran = ran[np.where(count > 0)]
        count = count[np.where(count > 0)]
        
        plt.plot(ran,count,c = colors[i])
        
        mean = sum(errors)/len(errors)
        # find closest bin
        idx = 0
        while mean > ran[idx]:
            idx += 1
            print(mean,idx,ran[idx],count[idx])
            
        ymeans.append(count[idx])
        xmeans.append(mean)
        m = np.max(count)
        count_max.append(m)
        
    for i in range(len(xmeans)-1):
        plt.annotate( "{:2f} ft".format(xmeans[i]),(xmeans[i],ymeans[i]),rotation= 45,fontsize = 16)
        plt.axvline(x = xmeans[i],ymax = ymeans[i]/max(count_max),ls = ":",c = colors[i],label='_nolegend_')
    
    plt.xlim([0,cutoff])
    
    plt.ylim([0,np.max(count)])
    plt.yticks([])
    
    plt.xticks(fontsize = 18)
    plt.legend(names,fontsize = 18)
    plt.ylabel("Relative frequency", fontsize = 24)
    plt.xlabel("Cross-camera projection error (ft)", fontsize = 24)
    plt.savefig("histogram.pdf",bbox_inches="tight")
    plt.show()
    
    
def plot_histograms(cutoff_error = [3,2],n_bins =30):
    
    # for pixel moving plot
    rescale = 7
    colors = np.array([[1,0,0],[0.5,0,0.5],[1,0,1],[0,0,1],[0.3,0.2,0.9],[0,0.5,0.5],[0,0.7,0.2],[.3,1,0],[0,1,0]])
    directory = "histogram_data3"
    includex = [1,0,1,1,1,1,1,1,0,0]
    includey = [1,1,0,0,1,1,1,1,0,0]   
    includep = [1,1,1,1,1,1,1,1,0,0]
    llx      = [20,20,35,45,110,105,0,0,0]
    lly      = [20,20,50,115,105,75,0,0,0]
    llp      = [20,20,50,115,105,75,0,0,0]
    
    rescale = 5
    rot = 45
    # colors = np.array([[0.2,0.5,0.2],[1,0,0],[0.5,0,0.5],[0,0,1],[0,0.7,0.7],[0.2,0.6,0.2],[0,0,0],[0,0,0],[0,0,0]])
    # directory = "histogram_data"
    # includex = [1,1,0,1,0,0]
    # includey = [1,1,1,0,0,0]   
    # llx      = [20,20,35,45,110,105,0,0,0]
    # lly      = [20,20,50,115,105,75,0,0,0]
    
    xmeans1    = []
    ymeans1    = []
    count_max1 = []
    xmeans2    = []
    ymeans2    = []
    count_max2 = []
    
    
    legend1 = []
    legend2 = []
    fig, axs = plt.subplots(1, 2,figsize = (13,5))
    paths = os.listdir(directory)
    paths.sort()
    
    for f_idx,file in enumerate(paths):
        # get data
        path = os.path.join(directory,file)
        
        name = path.split("/")[-1].split(".")[0][4:]
        name = name.replace("_"," ")
        
        try:
            with open(path,"rb") as f:
                [x_err,y_err,p_err,_,_,_] = pickle.load(f)
        except:
            continue
            
        # plot x data
        clipped = 0
        if includex[f_idx]:
            

            
            # plot histogram
            minv = 0
            maxv = max(x_err)
            cutoff = cutoff_error[0]
            
            ran = np.linspace(minv,cutoff+0.1,n_bins)
            count = np.zeros(ran.shape)
            
            for item in x_err:
                binned = False
                for r in range(1,len(ran)):
                    if item > ran[r-1] and item < ran[r]:
                        count[r-1] += 1
                        binned = True
                        break
                if not binned:
                    clipped += 1
            count /= len(x_err)
    
            # remove all 0 bins
            #ran = ran[np.where(count > 0)]
            #count = count[np.where(count > 0)]
            
            axs[0].plot(ran,count*rescale,c = colors[f_idx])
            
            mean = sum(x_err)/len(x_err)
            # find closest bin
            idx = 0
            while mean > ran[idx]:
                idx += 1
                #print(mean,idx,ran[idx],count[idx])
                
            ratio = (mean-ran[idx-1])/(ran[idx] -ran[idx-1])
            ymeans1.append((count[idx]*ratio+count[idx-1]*(1-ratio)))
            xmeans1.append(mean)
            m = np.max(count)
            count_max1.append(m)
            clip_percent = int((1-clipped/len(x_err)) * 1000)/10
            legend1.append("{} ({:.1f}%)".format(name,clip_percent))
            
        else:
            ymeans1.append(0)
            xmeans1.append(0)
            count_max1.append(0)
            
        # plot y data
        clipped = 0
        if includey[f_idx]:
            
            # plot histogram
            minv = 0
            maxv = max(y_err)
            cutoff = cutoff_error[1]
            
            ran = np.linspace(minv,cutoff+.1,n_bins)
            count = np.zeros(ran.shape)
            
            for item in y_err:
                binned = False
                for r in range(1,len(ran)):
                    if item > ran[r-1] and item < ran[r]:
                        count[r-1] += 1
                        binned = True
                        break
                if not binned:
                    clipped += 1
            count /= len(y_err)
            # remove all 0 bins
            # ran = ran[np.where(count > 0)]
            # count = count[np.where(count > 0)]
            
            axs[1].plot(ran,count*rescale,c = colors[f_idx])
            
            mean = sum(y_err)/len(y_err)
            # find closest bin
            idx = 0
            while mean > ran[idx]:
                idx += 1
                #print(mean,idx,ran[idx],count[idx])
            ratio = (mean-ran[idx-1])/(ran[idx] -ran[idx-1])
            ymeans2.append((count[idx]*ratio+count[idx-1]*(1-ratio)))
            xmeans2.append(mean)
            m = np.max(count)
            count_max2.append(m)
            clip_percent = int((1-clipped/len(x_err)) * 1000)/10
            legend2.append("{} ({:.1f}%)".format(name,clip_percent))
        else:
            ymeans2.append(0)
            xmeans2.append(0)
            count_max2.append(0)
            

    # plot x means
    for i in range(len(paths)):
        if includex[i]:
            axs[0].annotate("{:.3f} ft".format(xmeans1[i]),
                            xycoords='data',
                            xy=(xmeans1[i],min(ymeans1[i]*rescale,0.5)),
                            textcoords='offset points',
                            xytext=(xmeans1[i]+llx[i]*np.cos(np.pi/180*rot),ymeans1[i]+llx[i]*np.sin(np.pi/180*rot)),
                            arrowprops=dict(arrowstyle='-', color='black'),
                            rotation= rot,
                            fontsize = 14)
            axs[0].axvline(x = xmeans1[i],ymax = ymeans1[i]/max(count_max1)*rescale,ls = ":",c = colors[i],label='_nolegend_')
            
    # plot y means
    for i in range(len(paths)):
        if includey[i]:
            plus = np.random.rand()*60
            axs[1].annotate("{:.3f} ft".format(xmeans2[i]),
                            xycoords='data',
                            xy=(xmeans2[i],min(ymeans2[i]*rescale,0.5)),
                            textcoords='offset points',
                            xytext=(xmeans2[i]+lly[i]*np.cos(np.pi/180*rot),ymeans2[i]+lly[i]*np.sin(np.pi/180*rot)),
                            arrowprops=dict(arrowstyle='-', color='black'),
                            rotation= rot,
                            fontsize = 14)
            axs[1].axvline(x = xmeans2[i],ymax = ymeans2[i]/max(count_max2)*rescale,ls = ":",c = colors[i],label='_nolegend_')
        
    axs[0].set_xlim([0,cutoff_error[0]])
    axs[0].set_ylim([0,np.max(count_max1)])
    axs[1].set_xlim([0,cutoff_error[1]])
    axs[1].set_ylim([0,np.max(count_max2)])
    
    axs[0].set_yticks([])
    axs[1].set_yticks([])
    axs[0].xaxis.set_tick_params(labelsize=14)
    axs[1].xaxis.set_tick_params(labelsize=14)
    
    axs[0].set_ylabel("Relative frequency", fontsize = 18)
    axs[0].set_xlabel("Cross-camera x-error (ft)", fontsize = 18)
    axs[1].set_xlabel("Cross-camera y-error (ft)", fontsize = 18)

    plt.subplots_adjust(wspace=0.1, hspace=0)

    axs[0].legend(legend1,fontsize = 14)
    axs[1].legend(legend2,fontsize = 14)

    
    plt.savefig("histogram.pdf",bbox_inches="tight")
    plt.show()

def plot_histograms2(cutoff_error = [3,1.5,25],n_bins =30):
    
    # for pixel moving plot
    colors = np.array([[1,0,0],[0.5,0,0.5],[1,0,1],[0,0,1],[0.3,0.2,0.9],[0,0.5,0.5],[0.6,0.7,0.4],[.3,1,0],[0,1,0]])
    directory = "histogram_data3"
    includex = [0,0,0,1,1,1,1,1,0,0]
    includey = [0,0,0,1,1,1,1,1,0,0]   
    includep = [0,0,0,1,1,1,1,1,0,0]
    llx      = [20,10,20,40,50,80,50,140,20]
    lly      = [20,20,20,20,35,55,160,60,20]
    llp      = [20,40,10,30,40,130,80,160,50]
    
    rescale = 5
    rot = 70
    # colors = np.array([[0.2,0.5,0.2],[1,0,0],[0.5,0,0.5],[0,0,1],[0,0.7,0.7],[0.2,0.6,0.2],[0,0,0],[0,0,0],[0,0,0]])
    # directory = "histogram_data"
    # includex = [1,1,0,1,0,0]
    # includey = [1,1,1,0,0,0]   
    # llx      = [20,20,35,45,110,105,0,0,0]
    # lly      = [20,20,50,115,105,75,0,0,0]
    
    xmeans1    = []
    ymeans1    = []
    count_max1 = []
    xmeans2    = []
    ymeans2    = []
    count_max2 = []
    xmeans3    = []
    ymeans3    = []
    count_max3 = []
    
    
    legend1 = []
    legend2 = []
    legend3 = []
    fig, axs = plt.subplots(1, 3,figsize = (21,5))
    paths = os.listdir(directory)
    paths.sort()
    
    for f_idx,file in enumerate(paths):
        # get data
        path = os.path.join(directory,file)
        
        name = path.split("/")[-1].split(".")[0][4:]
        name = name.replace("_"," ")
        
        try:
            with open(path,"rb") as f:
                [x_err,y_err,p_err,_,_,_] = pickle.load(f)
        except:
            with open(path,"rb") as f:
                [x_err,y_err,p_err,_,_] = pickle.load(f)
            
        # plot x data
        clipped = 0
        if includex[f_idx]:
            

            
            # plot histogram
            minv = 0
            maxv = max(x_err)
            cutoff = cutoff_error[0]
            
            ran = np.linspace(minv,cutoff+0.1,n_bins)
            count = np.zeros(ran.shape)
            
            for item in x_err:
                binned = False
                for r in range(1,len(ran)):
                    if item >= ran[r-1] and item < ran[r]:
                        count[r-1] += 1
                        binned = True
                        break
                if not binned and item > ran[-1]:
                    clipped += 1
            count /= len(x_err)
    
            # remove all 0 bins
            #ran = ran[np.where(count > 0)]
            #count = count[np.where(count > 0)]
            
            axs[0].plot(ran,count*rescale,c = colors[f_idx])
            
            mean = sum(x_err)/len(x_err)
            # find closest bin
            idx = 0
            while mean > ran[idx]:
                idx += 1
                #print(mean,idx,ran[idx],count[idx])
                
            ratio = (mean-ran[idx-1])/(ran[idx] -ran[idx-1])
            ymeans1.append((count[idx]*ratio+count[idx-1]*(1-ratio)))
            xmeans1.append(mean)
            m = np.max(count)
            count_max1.append(m)
            clip_percent = int((1-clipped/len(x_err)) * 1000)/10
            legend1.append("{} ({:.1f}%)".format(name,clip_percent))
            
        else:
            ymeans1.append(0)
            xmeans1.append(0)
            count_max1.append(0)
            
        # plot y data
        clipped = 0
        if includey[f_idx]:
            
            # plot histogram
            minv = 0
            maxv = max(y_err)
            cutoff = cutoff_error[1]
            
            ran = np.linspace(minv,cutoff+.1,n_bins)
            count = np.zeros(ran.shape)
            
            for item in y_err:
                binned = False
                for r in range(1,len(ran)):
                    if item >= ran[r-1] and item < ran[r]:
                        count[r-1] += 1
                        binned = True
                        break
                if not binned:
                    clipped += 1
            count /= len(y_err)
            # remove all 0 bins
            # ran = ran[np.where(count > 0)]
            # count = count[np.where(count > 0)]
            
            axs[1].plot(ran,count*rescale,c = colors[f_idx])
            
            mean = sum(y_err)/len(y_err)
            # find closest bin
            idx = 0
            while mean > ran[idx]:
                idx += 1
                #print(mean,idx,ran[idx],count[idx])
            ratio = (mean-ran[idx-1])/(ran[idx] -ran[idx-1])
            ymeans2.append((count[idx]*ratio+count[idx-1]*(1-ratio)))
            xmeans2.append(mean)
            m = np.max(count)
            count_max2.append(m)
            clip_percent = int((1-clipped/len(y_err)) * 1000)/10
            legend2.append("{} ({:.1f}%)".format(name,clip_percent))
        else:
            ymeans2.append(0)
            xmeans2.append(0)
            count_max2.append(0)
            
        
        # plot pixel data
        clipped = 0
        if includep[f_idx]:
            
            # plot histogram
            minv = 0
            maxv = max(p_err)
            cutoff = cutoff_error[2]
            
            ran = np.linspace(minv,cutoff+1,n_bins)
            count = np.zeros(ran.shape)
            
            for item in p_err:
                binned = False
                for r in range(1,len(ran)):
                    if item >= ran[r-1] and item < ran[r]:
                        count[r-1] += 1
                        binned = True
                        break
                if not binned:
                    clipped += 1
            count /= len(p_err)
            # remove all 0 bins
            # ran = ran[np.where(count > 0)]
            # count = count[np.where(count > 0)]
            
            axs[2].plot(ran,count*rescale,c = colors[f_idx])
            
            mean = sum(p_err)/len(p_err)
            # find closest bin
            idx = 0
            while mean > ran[idx]:
                idx += 1
                #print(mean,idx,ran[idx],count[idx])
            ratio = (mean-ran[idx-1])/(ran[idx] -ran[idx-1])
            ymeans3.append((count[idx]*ratio+count[idx-1]*(1-ratio)))
            xmeans3.append(mean)
            m = np.max(count)
            count_max3.append(m)
            clip_percent = int((1-clipped/len(p_err)) * 1000)/10
            legend3.append("{} ({:.1f}%)".format(name,clip_percent))
        else:
            ymeans3.append(0)
            xmeans3.append(0)
            count_max3.append(0)
            

    # plot x means
    for i in range(len(paths)):
        if includex[i]:
            axs[0].annotate("{:.2f} ft".format(xmeans1[i]),
                            xycoords='data',
                            xy=(xmeans1[i],min(ymeans1[i]*rescale,0.5)),
                            textcoords='offset points',
                            xytext=(xmeans1[i]+llx[i]*np.cos(np.pi/180*rot),ymeans1[i]+llx[i]*np.sin(np.pi/180*rot)),
                            arrowprops=dict(arrowstyle='-', color='black'),
                            rotation= rot,
                            fontsize = 14)
            axs[0].axvline(x = xmeans1[i],ymax = ymeans1[i]/max(count_max1)*rescale,ls = ":",c = colors[i],label='_nolegend_')
            
    # plot y means
    for i in range(len(paths)):
        if includey[i]:
            axs[1].annotate("{:.2f} ft".format(xmeans2[i]),
                            xycoords='data',
                            xy=(xmeans2[i],min(ymeans2[i]*rescale,0.5)),
                            textcoords='offset points',
                            xytext=(xmeans2[i]+lly[i]*np.cos(np.pi/180*rot),ymeans2[i]+lly[i]*np.sin(np.pi/180*rot)),
                            arrowprops=dict(arrowstyle='-', color='black'),
                            rotation= rot,
                            fontsize = 14)
            axs[1].axvline(x = xmeans2[i],ymax = ymeans2[i]/max(count_max2)*rescale,ls = ":",c = colors[i],label='_nolegend_')
            
    # plot p means
    for i in range(len(paths)):
        rot = 65
        if includep[i]:
            axs[2].annotate("{:.1f} px".format(xmeans3[i]),
                            xycoords='data',
                            xy=(xmeans3[i],min(ymeans3[i]*rescale,0.5)),
                            textcoords='offset points',
                            xytext=(xmeans3[i]+llp[i]*np.cos(np.pi/180*rot),ymeans3[i]+llp[i]*np.sin(np.pi/180*rot)),
                            arrowprops=dict(arrowstyle='-', color='black'),
                            rotation= rot,
                            fontsize = 14)
            axs[2].axvline(x = xmeans3[i],ymax = ymeans3[i]/max(count_max3)*rescale,ls = ":",c = colors[i],label='_nolegend_')
        
    axs[0].set_xlim([0,cutoff_error[0]])
    axs[0].set_ylim([0,np.max(count_max1)])
    axs[1].set_xlim([0,cutoff_error[1]])
    axs[1].set_ylim([0,np.max(count_max2)])
    axs[2].set_xlim([0,cutoff_error[2]])
    axs[2].set_ylim([0,np.max(count_max3)])
    
    axs[0].set_yticks([])
    axs[1].set_yticks([])
    axs[2].set_yticks([])
    axs[0].xaxis.set_tick_params(labelsize=14)
    axs[1].xaxis.set_tick_params(labelsize=14)
    axs[2].xaxis.set_tick_params(labelsize=14)

    
    axs[0].set_ylabel("Relative frequency", fontsize = 18)
    axs[0].set_xlabel("a.) Cross-camera x-error (ft)", fontsize = 24)
    axs[1].set_xlabel("b.) Cross-camera y-error (ft)", fontsize = 24)
    axs[2].set_xlabel("c.) Cross-camera pixel error", fontsize = 24)


    plt.subplots_adjust(wspace=0.1, hspace=0)

    axs[0].legend(legend1,fontsize = 14)
    axs[1].legend(legend2,fontsize = 14)
    axs[2].legend(legend3,fontsize = 14)

    
    plt.savefig("histogram.pdf",bbox_inches="tight")
    plt.show()

def inlier_idx(array, ratio = 3):
    keep = [0]
    for i in range(len(array)-2):       
        if np.abs((array[i+2]- array[i]))/np.abs(array[i+1] - array[i]) < ratio:
            keep.append(i)
            
    return keep

#plot_proj_error([x_error,y_error,x_error_post,y_error_post],names = ["x", "y","x corrected", "y corrected"],cutoff_error = 5)

    
#%%
    
def plot_trajectories_unified(anns,lane = [82,98],
                                  selected_idx = 4,
                                  smooth = 1,
                                  v_range = [0,140],
                                  a_range =[-20,20],
                                  theta_range = [-25,25]):
     
    fig, axs = plt.subplots(2,4,figsize = (25,5*len(anns)),sharex= True)
    t0 = min(list(anns[0].all_ts[0].values()))   
    for a_idx,ann in enumerate(anns):
        ax1,ax2,ax3,ax4 = axs[a_idx,0:4]

        all_x = []
        all_y = []
        all_time = []
        all_ids = []
        all_lengths = []
        
        all_v = []
        all_a = []
        all_theta = []
        
       
        
        for obj_idx in range(ann.get_unused_id()):
            x = []
            y = []
            time = []
        
            for cam_idx, camera in enumerate(ann.cameras):
                cam_name = camera.name
                for frame in range(0,len(ann.data)):
                    key = "{}_{}".format(cam_name,obj_idx)
                    item = ann.data[frame].get(key)
                    if item is not None:
                        y_test = ann.safe(item["y"])
                        if y_test > lane[0] and y_test < lane[1]:
                            x.append(ann.safe(item["x"]))
                            y.append(ann.safe(item["y"]))
                            time.append(ann.safe(item["timestamp"]))
                            length = item["l"]
            
            
               
            if len(time) == 0:
                continue
                
            # sort by time     
            x = np.array(x)
            y = np.array(y)
            time = np.array(time) # - t0
            
            order = np.argsort(time)
            
            x = x[order]
            y = y[order]
            time = time[order]
            

            keep = [0]
            for i in range(1,len(time)):
                if time[i] >= time[i-1] + 0.01:
                    keep.append(i)
                    
            x = x[keep]
            y = y[keep]
            time = time[keep]
            
            
                        
            # estimate derivative qualities
            if len(time) > 1:
                
                
                try:
                    vel_spline = ann.splines[obj_idx][0].derivative()
                    v = vel_spline(time)
                    
                except:
                    
                    # finite difference velocity estimation
                    v = [(x[i] - x[i-1]) / (time[i] - time[i-1] + 1e-08) for i in range(1,len(x))] 
                    v += [v[-1]]
                    v  = np.array(v)
                    #v = np.convolve(v,np.hamming(smooth),mode = "same")
                v *= -1
                
                vy = [(y[i] - y[i-1]) / (time[i] - time[i-1] + 1e-08) for i in range(1,len(y))] 
                vy += [vy[-1]]
                vy = np.array(vy)
                vy = np.convolve(vy,np.hamming(smooth),mode = "same")

                try:
                    a_spline = ann.splines[obj_idx][0].derivative().derivative()
                    a = a_spline(time)
                    
                except:
                    a = [(v[i] - v[i-1]) / (time[i] - time[i-1] + 1e-08) for i in range(1,len(v))]
                    a += [a[-1]]
                    a = np.array(a)
                    a = np.convolve(a,np.hamming(smooth),mode = "same")

                theta = np.arctan2(vy,v) * 180/np.pi
                theta = np.convolve(theta,np.hamming(smooth),mode = "same")

                
                # store aggregate traj data
                all_time.append(time)
                all_v.append(v)
                all_x.append(x)
                all_y.append(y)
                all_a.append(a)
                all_theta.append(theta)
                all_lengths.append(length)
                all_ids.append(obj_idx)
            
        ax1.set_ylim([0,1800])
        ax2.set_ylim(v_range)
        ax3.set_ylim(a_range)
        ax4.set_ylim(theta_range)

        ax1.set_xlim([0,60])
        ax2.set_xlim([0,60])
        ax3.set_xlim([0,60])
        ax4.set_xlim([0,60])
        
        ax1.set_ylabel("x-position (ft)",fontsize = 20)
        ax2.set_ylabel("Velocity (ft/s)",fontsize = 20)
        ax3.set_ylabel("Acceleration ($ft/s^2$)",fontsize = 20)
        ax4.set_ylabel("Heading angle (deg)",fontsize = 20)
       
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax3.tick_params(axis='both', which='major', labelsize=14)
        ax4.tick_params(axis='both', which='major', labelsize=14)


        
        if a_idx == len(anns) -1:
            ax1.set_xlabel("Time (s)",fontsize = 20)
            ax2.set_xlabel("Time (s)",fontsize = 20)
            ax3.set_xlabel("Time (s)",fontsize = 20)
            ax4.set_xlabel("Time (s)",fontsize = 20)
        # else:
        #     ax1.set_xticks([])
        #     ax2.set_xticks([])
        #     ax3.set_xticks([])
        #     ax4.set_xticks([])
        

        if a_idx == 0:
            ax1.set_title("a.) All Vehicle Trajectories",fontsize = 24)
            ax2.set_title("b.) Selected Velocity",fontsize = 24)
            ax3.set_title("c.) Selected Acceleration",fontsize = 24)
            ax4.set_title("d.) Selected Heading Angle",fontsize = 24)
        
        gcolor = np.array([0,0,1])
        vcolor = np.array([1,0,0])
        acolor = np.array([1,0.4,0])
        tcolor = np.array([1,1,0])
        
        
        for i in range(len(all_x)):
            all_time[i] -= t0
        
            i_color = np.random.rand(3)*0.5 + 0.5
            i_color[2] = 0
            i_color[1] = i_color[1] - i_color[1]*0.2
            i_color = [0,0,0]
            if i== selected_idx:
                i_color = gcolor
            
            # plot single trajectory
            # else:
            #     continue
        
            #plot velocity
            points = np.array([all_time[i],all_v[i]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            vmask = np.logical_and(np.max(segments[:,:,1],axis = 1) < v_range[1],np.min(segments[:,:,1],axis = 1) > v_range[0])
            if i == selected_idx:
                clr = np.ones([len(segments),3]) * vcolor
                clr[vmask] = gcolor
                lc = LineCollection(segments, colors = clr)
                ax2.add_collection(lc)
            
            # plot acceleration
            points = np.array([all_time[i],all_a[i]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            amask = np.logical_and(np.max(segments[:,:,1],axis = 1) < a_range[1],np.min(segments[:,:,1],axis = 1) > a_range[0])
            if i == selected_idx:
                clr = np.ones([len(segments),3]) * acolor
                clr[amask] = gcolor
                lc = LineCollection(segments, colors = clr)
                ax3.add_collection(lc)
            
            # plot theta
            points = np.array([all_time[i],all_theta[i]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            tmask = np.logical_and(np.max(segments[:,:,1],axis = 1) < theta_range[1],np.min(segments[:,:,1],axis = 1) > theta_range[0])
            if i == selected_idx:
                clr = np.ones([len(segments),3]) * tcolor
                clr[tmask] = gcolor
                lc = LineCollection(segments, colors = clr)
                ax4.add_collection(lc)
            
            # ax1.plot(all_time[selected_idx],all_x[selected_idx],color = [0,0,0],linewidth = 3)
            # ax1.plot(all_time[selected_idx],all_x[selected_idx],color = [0,0,1],linewidth = 1)
            
            #plot position
            lw = 4
            # if i == selected_idx:
            #     lw = 5
            points = np.array([all_time[i],all_x[i]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            clr = np.ones([len(segments),3]) * i_color
            clr[np.invert(amask)] = acolor
            clr[np.invert(tmask)] = tcolor
            clr[np.invert(vmask)] = vcolor
            
            lc = LineCollection(segments, colors = clr, linewidth = lw)
            ax1.add_collection(lc)
            
            # plot radar data
            # keep = []
            # for j in range(len(all_x[i])):
            #     if all_x[i][j] % 300 < 1:
            #         keep.append(j)
            # tsub = all_time[i][keep]
            # xsub = all_x[i][keep]
            # vsub = all_v[i][keep]
            # colors = np.zeros([len(vsub),3])
            # colors [:,0] = 1 - vsub/30
            # colors [:,1] = 0 + vsub/60
            # colors = np.clip(colors,0,1)
            # ax1.scatter(tsub,xsub, color = colors)
            
     
    plt.subplots_adjust(wspace=0.25, hspace=0.05)
    plt.savefig("trajectories.pdf",bbox_inches="tight")
        # for i in range(len(all_x)):
        #         all_time[i] -= t0
        #         color = np.random.rand(3)*0.5 + 0.5
        #         color[2] = 0
        #         color[1] = color[1] - color[1]*0.2
               
        #         # axs[0].scatter(all_time[i],all_x[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
        #         # axs[1].scatter(all_time[i],all_v[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
        #         # axs[2].scatter(all_time[i],all_y[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
                
    
        #         ax1.plot(all_time[i],all_x[i],color = color)#/(i%1+1))
        #         #all_x2 = [all_lengths[i] + item for item in all_x[i]]
        #         #ax1.fill_between(all_time[i],all_x[i],all_x2,color = color)

        
        
        #ax2.plot(all_time[selected_idx],all_v[selected_idx],color = [0,0,1])
        #ax3.scatter(all_time[selected_idx],all_a[selected_idx],color = [0,0,1])
        #ax4.scatter(all_time[selected_idx],all_theta[selected_idx],color = [0,0,1])
            
        
       

        
    plt.show()  
    
def calculate_total_variation(ann):
        x_var = []
        x_range = []
        
        for obj_idx in range(ann.get_unused_id()):
            x = []
            time = []
        
            for cam_idx, camera in enumerate(ann.cameras):
                cam_name = camera.name
                for frame in range(0,len(ann.data)):
                    key = "{}_{}".format(cam_name,obj_idx)
                    item = ann.data[frame].get(key)
                    if item is not None:
                        x.append(ann.safe(item["x"]))
                        time.append(ann.safe(item["timestamp"]))
            
            
               
            if len(time) == 0:
                continue
                
            # sort by time     
            x = np.array(x)
            time = np.array(time) # - t0
            
            order = np.argsort(time)
            
            x = x[order]
            time = time[order]
            

            keep = [0]
            for i in range(1,len(time)):
                if time[i] >= time[i-1] + 0.01:
                    keep.append(i)
                    
            x = x[keep]            
            ran = (max(x) - min(x))
            variation = sum([np.abs(x[i] - x[i-1]) for i in range(1,len(x))])
            
            x_range.append(ran)
            x_var.append(variation)
        
        print("Total vs True variation: {}/{}   ({}x)".format(sum(x_var),sum(x_range),sum(x_var)/sum(x_range)))
        
        return x_var,x_range
    
def calculate_feasibility(ann):
    v_range = [0,140]
    a_range =[-20,20]
    theta_range = [-25,25]
    
    f_percentage = [] # one value per object
    for obj_idx in range(ann.get_unused_id()):
            x = []
            y = []
            time = []
        
            for cam_idx, camera in enumerate(ann.cameras):
                cam_name = camera.name
                for frame in range(0,len(ann.data)):
                    key = "{}_{}".format(cam_name,obj_idx)
                    item = ann.data[frame].get(key)
                    if item is not None:
                        x.append(ann.safe(item["x"]))
                        y.append(ann.safe(item["y"]))
                        time.append(ann.safe(item["timestamp"]))
                        length = item["l"]
            
            
               
            if len(time) == 0:
                continue
                
            # sort by time     
            x = np.array(x)
            y = np.array(y)
            time = np.array(time) # - t0
            
            order = np.argsort(time)
            
            x = x[order]
            y = y[order]
            time = time[order]
            

            keep = [0]
            for i in range(1,len(time)):
                if time[i] >= time[i-1] + 0.01:
                    keep.append(i)
                    
            x = x[keep]
            y = y[keep]
            time = time[keep]
                        
            # estimate derivative qualities
            if len(time) > 1:
                
                # finite difference velocity estimation
                v = [(x[i] - x[i-1]) / (time[i] - time[i-1] + 1e-08) for i in range(1,len(x))] 
                v += [v[-1]]
                v  = np.array(v)
                #v = np.convolve(v,np.hamming(smooth),mode = "same")
                v *= -1
                
                vy = [(y[i] - y[i-1]) / (time[i] - time[i-1] + 1e-08) for i in range(1,len(y))] 
                vy += [vy[-1]]
                vy = np.array(vy)

                
                a = [(v[i] - v[i-1]) / (time[i] - time[i-1] + 1e-08) for i in range(1,len(v))]
                a += [a[-1]]
                a = np.array(a)

                theta = np.arctan2(vy,v) * 180/np.pi
            
            
                points = np.array([time,v]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                vmask = np.logical_and(np.max(segments[:,:,1],axis = 1) < v_range[1],np.min(segments[:,:,1],axis = 1) > v_range[0])

            
                points = np.array([time,a]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                amask = np.logical_and(np.max(segments[:,:,1],axis = 1) < a_range[1],np.min(segments[:,:,1],axis = 1) > a_range[0])
           
            
                points = np.array([time,theta]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                tmask = np.logical_and(np.max(segments[:,:,1],axis = 1) < theta_range[1],np.min(segments[:,:,1],axis = 1) > theta_range[0])
                
                total_mask = np.logical_and(tmask,vmask,amask)
                
                percent_feasible = np.sum(total_mask.astype(int)) / total_mask.shape[0]
                f_percentage.append(percent_feasible)
    return f_percentage

def annotator_rmse(ann):
        mses = []
        
        i = 0
        obj_annotations = []
        cam = None
        while i < ann.get_unused_id():
            for frame_data in ann.data:
                for obj in frame_data.values():
                    if obj["id"] == i:
                        obj_annotations.append(obj)
                        cam = obj["camera"]
            if i%5 == 4:
                boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"]]).float() for obj in obj_annotations])
                im_boxes = ann.hg.state_to_im(boxes,name = cam)
                
                im_pos = torch.mean(im_boxes[:,[2,3],:],dim = 1)
                mean = torch.mean(im_pos,dim = 0)
                
                rmse = (((im_pos[:,0] - mean[0]).pow(2) + (im_pos[:,1] - mean[1]).pow(2)).sum(dim = 0)/im_pos.shape[0]).sqrt()
                mses.append(rmse)
                
                obj_annotations = []
            i+= 1
        
        rmse = torch.sqrt(sum(mses)/len(mses))
        print("RMSE: {}".format(rmse))
                

def plot_deltas(ann,cam = "p1c5"):
    
    all_ts = []
    for frame in ann.data:        
        for item in frame:
            if cam in item:    
                all_ts.append(frame[item]["timestamp"])
                break
        
    deltas = [all_ts[i] - all_ts[i-1] for i in range(1,len(all_ts))]
    plt.plot(deltas)
    plt.ylim([-0.01,0.05])
    plt.xlim([0,250])
    plt.show
        
#%%    

if __name__ == "__main__":
    
    try:
    #if True:
        parser = argparse.ArgumentParser()
        parser.add_argument('-scene', type = int, default =0)
        parser.add_argument('-spline_metric', type = str, default = "ape")
        parser.add_argument('-ts_metric', type = str, default = "ape")
        parser.add_argument('--correct_bias', action='store_true')
        parser.add_argument('--re', action='store_true')
        parser.add_argument('-n_frames', type = int, default = 2700)
        parser.add_argument('-max_shift',type = float, default = 0.0433)
        args = parser.parse_args()
        
        scene_id = args.scene
        spl_met = args.spline_metric
        ts_met = args.ts_metric
        correct_bias = args.correct_bias
        n_frames = args.n_frames
        max_shift= args.max_shift
        re  = args.re
    
    except:    
        print("Argument Error")
        # scene_id = 4
        # spl_met = "mpe"
        # ts_met = "mpe"
        # correct_bias = False #True
        # n_frames = 500
        # max_shift= 0.01
        # re = False
    
    #plot_histograms2()
    
    overwrite = False
    
    scene_id = 6
    directory = "/home/derek/Data/dataset_beta/sequence_{}".format(scene_id)
    if scene_id == 0:
                exclude_p3c6 = True
    else:
                exclude_p3c6 = False
    
    # ann = Annotator(directory,scene_id = scene_id,exclude_p3c6 = exclude_p3c6)  
    #ann.run()
    # error = ann.port_homography()
    # ann.save2()
    # ann.reload_chg_data()
    #for active_cam in [0,2,4,6,8,10,12,14,15]:
    ann = Annotator(directory,scene_id = scene_id,exclude_p3c6 = exclude_p3c6)  
    ann.fill_buffer(500)
    ann.run()
    #ann.port_homography()
    #ann.save2()
    #ann.run()
    
    #plot_proj_error(ann)
    
    
    # for i in range(100):
    #      ann.plot_trajectory(i)
    # raise Exception
    
    ann.unbias_timestamps()
    ann.get_splines()
    ann.adjust_ts_with_trajectories()
    ann.unbias_timestamps()
    ann.get_splines()
    #ann.run()
    
    # if False:
    #     for scene_id in [6]:
    #         lf = {0:2700,4:1800,6:1800}
    #         directory = "/home/worklab/Data/dataset_beta/sequence_{}".format(scene_id)
    #         if scene_id == 0:
    #                     exclude_p3c6 = True
    #         else:
    #                     exclude_p3c6 = False
    #         ann = Annotator(directory,scene_id = scene_id,exclude_p3c6 = exclude_p3c6)  
    #         with open("cached_final_ann_{}.cpkl".format(scene_id),"rb") as f:
    #                         [ann.data,ann.all_ts,ann.splines] = pickle.load(f)
    #         ann.last_frame = lf[scene_id]
    #         ann.output_vid()
            
    #         plot_histograms2()
   
    
    # if False:
    #     totals = None
    #     for scene_id in [0,4,6]:
    #         directory = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k"
    #         directory = "/home/worklab/Data/dataset_beta/sequence_{}".format(scene_id)
    
            
    #         if scene_id == 0:
    #             exclude_p3c6 = True
    #         else:
    #             exclude_p3c6 = False
    #         ann = Annotator(directory,scene_id = scene_id,exclude_p3c6 = exclude_p3c6)  
                    
            
    #         with open("cached_final_ann_{}.cpkl".format(scene_id),"rb") as f:
    #                 [ann.data,ann.all_ts,ann.splines] = pickle.load(f)
    #         #ann.fill_buffer(200)
    #         #ann.run()
            
    #         counts = ann.count_extended_data()
    #         if totals is None:
    #             totals = counts
    #             totals["p3c6"] = 0
    #         else:
    #             for key in counts:
    #                 totals[key] += counts[key]

    #     print("Extended box counts:")
    #     _ = [print("{}:{} boxes".format(key,totals[key])) for key in totals]
    #     print("Total number of added boxes: {}".format(sum([totals[key] for key in totals])))
    
    # if False: #plot extended trajectories
    #     directory = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k"
    #     directory = "/home/worklab/Data/dataset_beta/sequence_{}".format(scene_id)

        
    #     if scene_id == 0:
    #         exclude_p3c6 = True
    #     else:
    #         exclude_p3c6 = False
    #     ann = Annotator(directory,scene_id = scene_id,exclude_p3c6 = exclude_p3c6)  
                
        
    #     with open("cached_final_ann_{}.cpkl".format(scene_id),"rb") as f:
    #             [ann.data,ann.all_ts,ann.splines] = pickle.load(f)
        
    #     px_correction  = 3      
    #     ann.adjust_boxes_with_trajectories(max_shift_x = px_correction,max_shift_y = px_correction)
        
    #     ann.fill_buffer(10)
    #     ann.run()
    
    # if False:
    #     xv = []
    #     xv_corrected = []
    #     xr = []
    #     for scene_id in [0,4,6]:
    #         directory = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k"
    #         directory = "/home/worklab/Data/dataset_beta/sequence_{}".format(scene_id)

            
    #         if scene_id == 0:
    #             exclude_p3c6 = True
    #         else:
    #             exclude_p3c6 = False
            
            
            
    #         ann = Annotator(directory,scene_id = scene_id,exclude_p3c6 = exclude_p3c6)  
            
    #         x_var,x_ran = calculate_total_variation(ann)
    #         xv += x_var
    #         xr += x_ran
            
    #         with open("linear_spacing_splines_{}.cpkl".format(scene_id),"rb") as f:
    #                 [ann.data,ann.all_ts,ann.splines] = pickle.load(f)
            
    #         px_correction  = 3      
    #         ann.adjust_boxes_with_trajectories(max_shift_x = px_correction,max_shift_y = px_correction)
    #         x_var_new,_ = calculate_total_variation(ann)
    #         xv_corrected += x_var_new
        
    #     print("Total vs True variation before corrections: {}/{}   ({}x)".format(sum(xv),sum(xr),sum(xv)/sum(xr)))
    #     print("Total vs True variation after  corrections: {}/{}   ({}x)".format(sum(xv_corrected),sum(xr),sum(xv_corrected)/sum(xr)))
        
    
    # if False:
    #     #plot_histograms()
    
    #     directory = "/home/worklab/Data/dataset_beta/sequence_{}".format(scene_id)
    #     x_error = []
    #     y_error = []
        
    #     if scene_id == 0:
    #         exclude_p3c6 = True
    #     else:
    #         exclude_p3c6 = False
        
        
        
    #     ann = Annotator(directory,scene_id = scene_id,exclude_p3c6 = exclude_p3c6)  
        
    #     # ann.fill_buffer(1000)
    #     # ann.run()
        
    #     ann2 = Annotator(directory,scene_id = scene_id,exclude_p3c6 = exclude_p3c6)  
    #     with open("linear_spacing_splines_{}.cpkl".format(scene_id),"rb") as f:
    #         [ann2.data2,ann2.all_ts,ann2.splines] = pickle.load(f)
    #     #ann2.get_splines(plot = True)
    #     ann2.gen_trajectories()
    #     ann2.data = ann2.spline_data        
    #     # plot_trajectories_unified([ann,ann2],smooth = 1,selected_idx = 7)
    
    #     #px_correction = 5
    #     #ann2.adjust_boxes_with_trajectories(max_shift_x = px_correction,max_shift_y = px_correction)
    #     plot_trajectories_unified([ann,ann2],smooth = 1,selected_idx = 7)

    
    
    
    
    # if False:
    #     for scene_id in [0,4,6]:
    #         directory = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k"
    #         directory = "/home/worklab/Data/dataset_beta/sequence_{}".format(scene_id)
    #         x_error = []
    #         y_error = []
    #         print("On Scene {}".format(scene_id))
            
    #         if scene_id == 0:
    #             exclude_p3c6 = True
    #         else:
    #             exclude_p3c6 = False
            
            
            
    #         ann = Annotator(directory,scene_id = scene_id,exclude_p3c6 = exclude_p3c6)  
            
    #         with open("cached_final_ann_{}.cpkl".format(scene_id),"rb") as f:
    #                 [ann.data,ann.all_ts,ann.splines] = pickle.load(f)
                   
    #         px_correction = 3
    #         ann.adjust_boxes_with_trajectories(max_shift_x = px_correction,max_shift_y = px_correction)
    #         ann.get_splines(plot = False,metric = spl_met)
            
    #         with open("cached_respline_final_ann_{}.cpkl".format(scene_id),"wb") as f:
    #             pickle.dump([ann.data,ann.all_ts,ann.splines],f)    
            
            
    
    
    
    
    # if False:
    #     for px_correction in [5]:
    #         for scene_id in [0,4,6]:
    #             directory = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k"
    #             directory = "/home/worklab/Data/dataset_beta/sequence_{}".format(scene_id)
    #             x_error = []
    #             y_error = []
    #             ps_all = []
    #             print("On Scene {}, {}pixel max".format(scene_id,px_correction))
                
    #             if scene_id == 0:
    #                 exclude_p3c6 = True
    #             else:
    #                 exclude_p3c6 = False
                
                
                
    #             ann = Annotator(directory,scene_id = scene_id,exclude_p3c6 = exclude_p3c6)  
                
    #             with open("cached_final_ann_{}.cpkl".format(scene_id),"rb") as f:
    #                     [ann.data,ann.all_ts,ann.splines] = pickle.load(f)
                        
    #             pixel_shifts = ann.adjust_boxes_with_trajectories(max_shift_x = px_correction,max_shift_y = px_correction)
    #             ps_all += pixel_shifts
                
    #             # get projection error
    #             #[err_x,err_y,err_p] = ann.estimate_projection_error()
    #             #x_error += err_x
    #             #y_error += err_y 
           
    #         print("With max pixel correction {} px, {}px average shift".format(px_correction, sum(ps_all)/len(ps_all)))
    #         # # save x error y error and title for plotting later 
    #         # with open("PE_{}_pixels.cpkl".format(px_correction),"wb") as f:
    #         #         pickle.dump([x_error,y_error,"{} pixel correction"],f)   
            
    #         ann.run()
    #         del ann
        
    # if False: # get roadway error
    #     x_error = []
    #     y_error = []
    #     p_error = []
    #     for scene_id in [0,4,6]:
    #         directory = "/home/worklab/Data/dataset_beta/sequence_{}".format(scene_id)
    #         print("On Scene {}".format(scene_id))
            
    #         if scene_id == 0:
    #             exclude_p3c6 = True
    #         else:
    #             exclude_p3c6 = False
                
    #         ann = Annotator(directory,scene_id = scene_id,exclude_p3c6 = exclude_p3c6)  
    #         [err_x,err_y,err_p] = ann.estimate_transform_reprojection_error2(use_curvature = False)
    #         x_error += err_x
    #         y_error += err_y 
    #         p_error += err_p
        
    #     print("CCDE x: {}".format(sum(x_error)/len(x_error)))
    #     print("CCDE y: {}".format(sum(y_error)/len(y_error)))
    #     print("CCPE: {}".format(sum(p_error)/len(p_error)))
    #     #save x error y error and title for plotting later 
    #     with open("PE_hg.cpkl","wb") as f:
    #             pickle.dump([x_error,y_error,p_error,None,None],f)    
            
    
    # if False: # get proj error histogram
        
    #     for px_correction in [0,1,2,3]:
    #         print("\n\nPixel Correction {}px".format(px_correction))
    #         x_error = []
    #         y_error = []
    #         p_error = []
            
    #         var_total = []
    #         ran_total = []
    #         f_total   = []
    #         for scene_id in [0,4,6]:
    #             directory = "/home/worklab/Data/dataset_beta/sequence_{}".format(scene_id)
    #             print("On scene {}".format(scene_id))
                
    #             if scene_id == 0:
    #                 exclude_p3c6 = True
    #             else:
    #                 exclude_p3c6 = False
    #             ann = Annotator(directory,scene_id = scene_id,exclude_p3c6 = exclude_p3c6)  
                
    #             # do change
    #             #ann.unbias_timestamps()
    #             with open("linear_spacing_splines_{}.cpkl".format(scene_id),"rb") as f:
    #                     [ann.data,ann.all_ts,ann.splines] = pickle.load(f)
                
    #             ann.gen_trajectories()
    #             ann.data = ann.spline_data     
    #             #ann.adjust_boxes_with_trajectories(max_shift_x = px_correction,max_shift_y = px_correction)

                
    #             # get projection error
    #             [err_x,err_y,err_p] = ann.estimate_projection_error(reverse_curve_offset = False)
    #             print(sum(err_x)/len(err_x),sum(err_y)/len(err_y),sum(err_p)/len(err_p))
    #             x_error += err_x
    #             y_error += err_y 
    #             p_error += err_p 
                
    #             # get total variation
    #             x_var,x_ran =  calculate_total_variation(ann)
    #             var_total += x_var
    #             ran_total += x_ran
                
    #             # get percent feasibility
    #             f_percent = calculate_feasibility(ann)
    #             f_total += f_percent
    
    #         print("Summary of metrics:")
    #         print("CCDE x: {}".format(sum(x_error)/len(x_error)))
    #         print("CCDE y: {}".format(sum(y_error)/len(y_error)))
    #         print("CCPE: {}".format(sum(p_error)/len(p_error)))
    #         print("V total: {}".format(sum(var_total)/sum(ran_total)))
    #         print("Percent feasible: {}".format(sum(f_total)/len(f_total)))

    #         # save x error y error and title for plotting later 
    #         # with open("PE_{}px.cpkl".format(px_correction),"wb") as f:
    #         #         pickle.dump([x_error,y_error,p_error,f_total,var_total,ran_total],f)    
    

    
    # if False: # run to do final spline fitting and timestamp adjustment
    #     for scene_id in [0,4,6]:
    #         directory = "/home/derek/Data/cv/video/ground_truth_video_06162021/segments_4k"
    #         directory = "/home/derek/Data/dataset_beta/sequence_{}".format(scene_id)
    
            
    #         if scene_id == 0:
    #             exclude_p3c6 = True
    #         else:
    #             exclude_p3c6 = False
                
                
                
    #         ann = Annotator(directory,scene_id = scene_id,exclude_p3c6 = exclude_p3c6)  
    #         ann.unbias_timestamps()
    #         ann.last_frame = n_frames
    #         ann.get_splines(plot = False,metric = spl_met)
            
    #         for i in range(1):
    #             ann.adjust_ts_with_trajectories(max_shift = max_shift,use_running_error = False,verbose = False,metric = ts_met)
    #             err_x,err_y,err_p = ann.estimate_projection_error()
    #             print(sum(err_x)/len(err_x),sum(err_y)/len(err_y))
    #             ann.get_splines(plot = False,metric = spl_met)
        
    #         with open("linear_spacing_splines_{}.cpkl".format(scene_id),"wb") as f:
    #             pickle.dump([ann.data,ann.all_ts,ann.splines],f)    
        
