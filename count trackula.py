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


import torch.multiprocessing as mp
ctx = mp.get_context('spawn')

# personal modules and packages
from i24_rcs import I24_RCS
from i24_database_api import DBClient
from nvc_buffer import NVC_Buffer

detector_path = os.path.join("retinanet")
sys.path.insert(0,detector_path)
from retinanet.model import resnet50 


import pyproj
from pyproj import Proj, transform
import pandas as pd
import numpy as np
import torch


def clockify(polygon, clockwise = True):
    """
    polygon - [n_vertices,2] tensor of x,y,coordinates for each convex polygon
    clockwise - if True, clockwise, otherwise counterclockwise
    returns - [n_vertices,2] tensor of sorted coordinates 
    """
    
    # get center
    center = torch.mean(polygon.float(), dim = 0)
    
    # get angle to each point from center
    diff = polygon - center.unsqueeze(0).expand([polygon.shape[0],2])
    tan = torch.atan(diff[:,1]/diff[:,0])
    direction = (torch.sign(diff[:,0]) - 1)/2.0 * -np.pi
    
    angle = tan + direction
    sorted_idxs = torch.argsort(angle)
    
    
    if not clockwise:
        sorted_idxs.reverse()
    
    polygon = polygon[sorted_idxs.detach(),:]
    return polygon

class Annotator:
    """ 
    
    """
    
    def __init__(self,im_directory,include_cameras,save_dir):
        
        
           
            
           
            
        self.camera_names = include_cameras
        # 1. Get multi-thread frame-loader object
        self.b = NVC_Buffer(im_directory,include_cameras,ctx,buffer_lim = 3601)
        
        
        self.data = dict([(name,[]) for name in self.camera_names])
        # reload data if it exists
        
        for name in self.camera_names:
            save_file = os.path.join(save_dir,name) + ".cpkl"
            try:
                with open(save_file,"rb") as f:
                    self.data[name] = pickle.load(f)
            except:
                pass
        self.save_dir = save_dir
        
        
        self.frame_idx = 0


        self.buffer(3600)
        
        
        #### Initialize a bunch of other stuff for tool management
        self.cont = True
        self.new = None
        self.clicked = False
        self.clicked_idx = None
        self.MASK = False

        self.PLAY_PAUSE = False
        self.active_command = "ADD"
        
        self.label_buffer = copy.deepcopy(self.data)
        self.colors =  np.random.rand(2000,3)
        
        self.active_region = {"start":None,
                              "end"  :None,
                              "points":[],
                              "static":False }

        self.active_idx = 0
        
        self.stride = 1
    
    def quit(self):      
        self.cont = False
        cv2.destroyAllWindows()
            
        self.save()
        
        
    def save(self):
        
        for name in self.camera_names:
            save_file = os.path.join(self.save_dir,name) + ".cpkl"
            try:
                with open(save_file,"wb") as f:
                    pickle.dump(self.data[name],f)
                print("Saved annotations at {}".format(save_file))
            except:
                print("Error saving {}".format(save_file))
                pass
        
       
        
        
        
    def buffer(self,n):
        self.b.fill(n)
        #while len(self.b.frames[self.frame_idx]) == 0:
        #self.next(stride = n-1)
            
    def safe(self,x):
        """
        Casts single-element tensor as an variable, otherwise does nothing
        """
        try:
            x = x.item()
        except:
            pass
        return x

        
    # def toggle_cams(self,dir):
    #     """dir should be -1 or 1"""
        
    #     if self.active_cam + dir < len(self.camera_names) -1 and self.active_cam + dir >= 0:
    #         self.active_cam += dir
    #         self.plot()
       
    #     if self.toggle_auto:
    #         self.AUTO = True
               
    def next(self,stride = 1):
        """
        We assume all relevant frames are pre-buffered, so all we have to do is 
        Check whether there's another frame available to advance, advance the index,
        and then assign the indexed frame and timestamps
        """        
        
        

        if self.frame_idx+stride < len(self.b.frames):
            self.frame_idx += stride
        else:
            print("On last frame")
    
    
    
    def prev(self,stride = 1):
        
        
        if self.frame_idx-stride >= 0 and len(self.b.frames[self.frame_idx-stride]) > 0:
            self.frame_idx -= stride     


        else:
            print("Cannot return to previous frame. First frame or buffer limit")
                        
    def plot(self):        
        plot_frames = []
        #ranges = self.ranges
        
        
        for i in range(4):
           frame = self.b.frames[self.frame_idx][i]
           frame = frame.copy()
           blur = frame.copy()
           
           cur_time = self.b.ts[self.frame_idx][i]
           
           # plot regions
           for region in self.data[self.camera_names[i]]:
               if region["static"] or (cur_time >= region["start"] and cur_time <= region["end"]):
                   color = (0,40,100)
                   if region["static"]: color = (0,0,255)
                   pts = np.array(region["points"]).reshape(-1,1,2)
                   
                   
                   if self.MASK:
                       transparency = np.ones(frame.shape)
                       transparency = cv2.fillPoly(
                           transparency, [pts],  (0,0,0), lineType=cv2.LINE_AA)
                       
                       blur = cv2.blur(blur,(25,25))
                       
                       frame = (transparency.astype(
                           float) * frame.astype(float)).astype(np.uint8) + ((1-transparency).astype(
                               float) * blur.astype(float)).astype(np.uint8)
                   
                   # plot outline
                   frame = cv2.polylines(frame,[pts],True,color,1)
                   
                   
                  
                           
                           
                   

           if self.active_idx == i and len(self.active_region["points"]) > 2:
               pts = np.array(self.active_region["points"]).reshape(-1,1,2)
               frame = cv2.polylines(frame,[pts],True,(200,150,0),1)
               

               
           
           # if self.MASK:
           #     gps_boxes = []
           #     gps_ids = []
           #     frame_ts = self.b.ts[self.frame_idx][i]
           #     for key in self.gps.keys():
           #         gpsob = self.gps[key]
           #         #print(gpsob["start"].item(),gpsob["end"].item())
           #         if gpsob["start"] < frame_ts and gpsob["end"] > frame_ts:
                       
           #             # iterate through timestamps to find ts directly before and after current ts
           #             for t in range(1,len(gpsob["ts"])):
           #                 if gpsob["ts"][t] > frame_ts:
           #                     break
                       
           #             x1 = gpsob["x"][t-1]
           #             x2 = gpsob["x"][t]
           #             y1 = gpsob["y"][t-1]
           #             y2 = gpsob["y"][t]
           #             t1 = gpsob["ts"][t-1]
           #             t2 = gpsob["ts"][t]
           #             f1 = (t2-frame_ts)/(t2-t1)
           #             f2 = (frame_ts-t1)/(t2-t1)
                       
           #             x_interp =  x1*f1 + x2*f2
           #             y_interp =  y1*f1 + y2*f2
                       
           #             l = gpsob["l"]
           #             w = gpsob["w"]
           #             h = gpsob["h"]
                       
           #             gps_box = torch.tensor([x_interp,y_interp,l,w,h,torch.sign(y_interp)])
           #             gps_boxes.append(gps_box)
           #             gps_ids.append(key)
            
           #     # plot labels
           #     if self.TEXT:
           #         times = [item["timestamp"] for item in ts_data]
           #         classes = [item["class"] for item in ts_data]
           #         ids = [item["id"] for item in ts_data]
           #         directions = [item["direction"] for item in ts_data]
           #         directions = ["WB" if item == -1 else "EB" for item in directions]
           #         camera.frame = Data_Reader.plot_labels(None,frame,im_boxes,boxes,classes,ids,None,directions,times)
                   
                
           # if self.MASK:
           #     mask_im = self.mask_ims[camera.name]/255
           #     blur_im = cv2.blur(frame,(17,17))
           #     frame = frame*mask_im + blur_im * (1-mask_im)*0.7
           
           if True:
               font =  cv2.FONT_HERSHEY_SIMPLEX
               header_text = "{} frame {}: {:.3f}s".format(self.camera_names[i],self.frame_idx,self.b.ts[self.frame_idx][i])
               frame = cv2.putText(frame,header_text,(30,30),font,1,(255,255,255),1)
               
           plot_frames.append(frame)
       
        # concatenate frames
        n_ims = len(plot_frames)
        n_row = int(np.round(np.sqrt(n_ims)))
        n_col = int(np.ceil(n_ims/n_row))
        
        cat_im = np.zeros([1080*n_row,1920*n_col,3]).astype(float)
        for i in range(len(plot_frames)):
            im = plot_frames[i]
            row = i % n_row
            col = i // n_row
            
            cat_im[col*1080:(col+1)*1080,row*1920:(row+1)*1920,:] = im
            
        # view frame and if necessary write to file
        cat_im /= 255.0
        
        
        self.plot_frame = cat_im
    
   
    
    
       

    def return_to_first_frame(self):
        #1. return to first frame in buffer
        for i in range(0,len(self.b.frames)):
            if len(self.b.frames[i]) > 0:
                break
            
        self.frame_idx = i
        self.label_buffer = copy.deepcopy(self.data)

        
        
 
        
        
    def add(self,point): 
    
       if len(self.active_region["points"]) == 0:
           self.active_idx = self.clicked_idx
           
           self.active_region["start"] = self.b.ts[self.frame_idx][self.active_idx]
           
       # clip point
       point[0] = max(min(1920,point[0]),0)
       point[1] = max(min(1080,point[1]),0)
       
       self.active_region["points"].append(point)
       
       points = torch.tensor(self.active_region["points"])
       self.active_region["points"] = clockify(points).tolist()
        
    def terminate_region(self):
        
       
       print("Terminated region in camera {}".format(self.camera_names[self.active_idx]))

       
       self.active_region["end"] = self.b.ts[self.frame_idx][self.active_idx]
       self.data[self.camera_names[self.active_idx]].append(self.active_region)

       self.active_region = {"start":None,
                              "end"  :None,
                              "points":[],
                              "static":False }
               
       self.active_idx    = 0
       
            
    def delete(self,cam_name,region_idx):
        """

        """
        if region_idx is None:
            return 
        
        else:
             del self.data[cam_name][region_idx]
        
    def on_mouse(self,event, x, y, flags, params):
       if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
         self.clicked = True 
       elif event == cv.EVENT_LBUTTONUP and self.clicked:
            
            
            self.clicked_idx = 0
            if x > 1920:
                self.clicked_idx += 1
                x -= 1920
            if y > 1080:
                self.clicked_idx += 2
                y -= 1080
                
            self.new = np.array([x,y])
            self.clicked = False
            
            #print(x,y)
            
            
    
    def find_region(self,point):
        
        return_idx = None
        point = point.copy()
        
        cam_name = self.camera_names[self.clicked_idx]
        
        for r_idx, region in enumerate(self.data[cam_name]):
            
            # check time 
            cur_time = self.b.ts[self.frame_idx][self.clicked_idx]
            if region["static"] or (cur_time >= region["start"] and cur_time <= region["end"]):
                
            
                # check whether clicked point falls inside region
                # get angle to each point
                xpts = np.array([pt[0] for pt in region["points"]])
                ypts = np.array([pt[1] for pt in region["points"]])
                
                xdisp = xpts - point[0]
                ydisp = ypts - point[1]
                angle = np.arctan2(ydisp,xdisp)*180/np.pi
                #angle = (angle + 360)%360
                spread = np.max(angle) - np.min(angle)
                if spread > 180:
                    return_idx = r_idx
        
        return return_idx
    
    
    
    def undo(self):
        if self.label_buffer is not None:
            self.data = self.label_buffer
            self.label_buffer = None
            self.plot()
            
        else:
            print("Can't undo")
    


    def run(self):
        """
        Main processing loop
        """
        
        cv2.namedWindow("window")
        cv.setMouseCallback("window", self.on_mouse, 0)
        self.plot()
        self.cont = True
        
        while(self.cont): # one frame
            
           ### handle click actions
           
           if self.new is not None:
                # buffer one change
                self.label_buffer = copy.deepcopy(self.data)
                
                # Add and delete objects
                if self.active_command   == "DELETE":
                    region_idx = self.find_region(self.new)
                    cam_name = self.camera_names[self.clicked_idx]
                    if region_idx is not None:
                        self.delete(cam_name,region_idx)
                    
                elif self.active_command == "ADD":
                    self.add(self.new)
                    
                elif self.active_command  == "TOGGLE STATIC":
                    region_idx = self.find_region(self.new)
                    cam_name = self.camera_names[self.clicked_idx]
                    if region_idx is not None:
                        self.data[cam_name][region_idx]["static"] = not self.data[cam_name][region_idx]["static"]
                    
                self.new = None
                self.plot()
          
                
           
           ### Show frame
                
           #self.cur_frame = cv2.resize(self.cur_frame,(1920,1080))
           cv2.imshow("window", self.plot_frame)
           title = "{}     Frame {}, Cameras {}".format( self.active_command,self.frame_idx,self.camera_names[0:4])
           cv2.setWindowTitle("window",str(title))
           
           
           ### Handle keystrokes 
           key = cv2.waitKey(1)

           if key == ord(" "):
               self.PLAY_PAUSE = not self.PLAY_PAUSE
               
           if key == ord("-"):
               self.stride -= 1
           elif key == ord("="):
                self.stride += 1
           if self.PLAY_PAUSE:
               self.next(stride = self.stride)
               self.plot()
               continue

           if key == ord('9'):
                self.next()
                self.plot()
           elif key == ord('8'):
                self.prev()  
                self.plot()
           
           elif key == ord("q"):
               self.quit()
           elif key == ord("w"):
               self.save()
           elif key == ord("u"):
               self.undo()
               self.plot()
               
           elif key == ord("f"):
                self.return_to_first_frame()
                self.plot()
           elif key == ord("t"):
               self.terminate_region()
               self.plot()
           elif key == ord("m"):
               self.MASK = not self.MASK
           
                
                
         
           # toggle commands
           elif key == ord("a"):
               self.active_command = "ADD"
           elif key == ord("d"):
               self.active_command = "DELETE"
           elif key == ord("s"):
               self.active_command = "TOGGLE STATIC"
               
               
         
           
            
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
           

        
#%%    

if __name__ == "__main__":
    directory = "/home/derek/Data/1hz"
    save_dir = "redaction"
    
    existing_save_files = os.listdir(save_dir)


    ## this logic should eventually get moved outside of object
    # get list of all cameras available
    camera_names = os.listdir(directory)
    camera_names.sort()
    camera_names.reverse()
    
    include_cameras = []
    for camera in camera_names:
        if ".pts" in camera: continue
        p = int(camera.split("P")[-1].split("C")[0])
        c = int(camera.split("C")[-1].split(".")[0])
        shortname = camera.split(".")[0]
        
        if p > 40: continue
        if c > 6:  continue 
    
        if shortname + ".cpkl" not in existing_save_files: 
            include_cameras.append(shortname)
        
        if len(include_cameras) == 4:
            break
    print(include_cameras)
    
    #Override
    include_cameras = ["P03C04","P03C03","P03C02","P03C01"]
    
    ann = Annotator(directory,include_cameras,save_dir)  
    ann.run()