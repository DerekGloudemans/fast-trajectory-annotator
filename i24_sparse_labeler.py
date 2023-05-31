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

class Annotator:
    """ 
    """
    
    def __init__(self,im_directory,hg_path,save_file = None):
        
        #### Initialize frame array

        
        # get list of all cameras available
        camera_names = os.listdir(im_directory)
        camera_names.sort()
        camera_names.reverse()
        
        include_cameras = []
        for camera in camera_names:
            if ".pts" in camera: continue
            p = int(camera.split("P")[-1].split("C")[0])
            c = int(camera.split("C")[-1].split(".")[0])
            shortname = camera.split(".")[0]
            
            if (c == 4) and p%2 == 0 and ".h264" in camera and p< 41 and p > 2 :
#            if c == 4 and p in [22,24,26,28]:

                include_cameras.append(shortname)
           
        self.camera_names = include_cameras
        # 1. Get multi-thread frame-loader object
        self.b = NVC_Buffer(im_directory,include_cameras,ctx,buffer_lim = 300)
        
        
        # frame indexed data array since we'll plot by frames
        self.data = [{} for i in range(5000)] # each frame 
        
        
        self.frame_idx = 0
        self.toggle_auto = True
        self.AUTO = True

        # dictionary with dimensions, source camera, number of annotations, current lane, and sink camera for each object
        self.objects = {} 

        self.buffer(100)
        
        
        #### get homography
        self.hg = I24_RCS(save_file = hg_path,downsample = 2)
        self.hg_path = hg_path
        self.extents = self.get_hg_extents()
    
        #### Initialize data structures for storing annotations, and optionally reload
        

       
        
           
        try:
            with open(save_file,"rb") as f:
                [self.data,self.objects] = pickle.load(f)
        except:
            pass
        self.save_file = save_file
        
       
        self.next_object_id = 0
        for id in self.objects.keys():
            if id > self.next_object_id:
                self.next_object_id = id + 1

                
        #### Initialize a bunch of other stuff for tool management
        self.cont = True
        self.new = None
        self.clicked = False
        self.clicked_camera = None
        self.TEXT = True
        self.LANES = True
        self.MASK = True
        
        self.active_command = "DIMENSION"
        self.right_click = False
        self.copied_box = None
        
        self.label_buffer = copy.deepcopy(self.data),copy.deepcopy(self.objects)
    
        self.colors =  np.random.rand(2000,3)
        
        loc_cp = "/home/derek/Documents/i24/fast-trajectory-annotator/data/localizer_april_112.pt"
        self.detector = resnet50(num_classes=8)
        cp = torch.load(loc_cp)
        self.detector.load_state_dict(cp) 
        self.detector.cuda()
        
        
        
        self.plot_idx = 0        
        self.active_cam = 0
    
        
        self.frame_gaps = np.zeros(len(self.camera_names))
        self.prev_cam_label_frame = 0
    
    
        # load GPS data
        gps_data_cache = "./data/GPS6.cpkl"
        try:
            with open(gps_data_cache,"rb") as f:
                self.gps = pickle.load(f)
        except:
            self.load_gps_data()
            with open(gps_data_cache,"wb") as f:
                pickle.dump(self.gps,f)
    
        self.find_furthest_gps(direction = -1)
        print("Loaded annotator")
   
   
    def get_hg_extents(self):
        # 2. convert all extent points into state coordinates
        data = {}
        for key in self.hg.correspondence.keys():
            pts = (torch.from_numpy(self.hg.correspondence[key]["corr_pts"])/2.0).unsqueeze(1)
            data[key] = self.hg.im_to_state(pts,name = [key for _ in range(len(pts))], heights = 0, refine_heights = False)
            
        # 3. Find min enclosing extents for each camera
        
        extents = {}
        for key in data.keys():
            key_data = data[key]
            minx = torch.min(key_data[:,0]).item()
            maxx = torch.max(key_data[:,0]).item()
            miny = torch.min(key_data[:,1]).item()
            maxy = torch.max(key_data[:,1]).item()
            
            extents[key] = [minx,maxx,miny,maxy]
            
        return extents
   
    def load_gps_data(self):
        """
        Load GPS file and convert to rcs. 
        
        
        self.gps - dict of dicts, each with CIRCLES_id,dimensions, and array of x,y,time
        """
        
        # collection = "637517698b5b68fc4fd40c77_CIRCLES_GPS"
        # db_param = {
        #       "host":"10.80.4.91",
        #       "port":27017,
        #       "username": "mongo-admin",
        #       "password": "i24-data-access",
        #       "database_name": "trajectories",      
        #       }
        
        
        # prd   = DBClient(**db_param,collection_name = collection)
        # preds = list(prd.read_query(None))
        def WGS84_to_TN(points):
            """
            Converts GPS coordiantes (WGS64 reference) to tennessee state plane coordinates (EPSG 2274).
            Transform is expected to be accurate within ~2 feet
            
            points array or tensor of size [n_pts,2]
            returns out - array or tensor of size [n_pts,2]
            """
            
            wgs84=pyproj.CRS("EPSG:4326")
            tnstate=pyproj.CRS("epsg:2274")
            out = pyproj.transform(wgs84,tnstate, points[:,0],points[:,1])
            out = np.array(out).transpose(1,0)
            
            if type(points) == torch.Tensor:
                out = torch.from_numpy(out)
                
            return out
        
        start_ts = 1668600000
        end_ts = start_ts + 60*60*4
        #gps_data_file = "/home/derek/Data/CIRCLES_GPS/CIRCLES_GPS_ALL.csv"
        #gps_data_file = "/home/derek/Data/CIRCLES_GPS/mvt_11_18_gps_vins.csv"
        gps_data_file = "/home/derek/Data/CIRCLES_GPS/mvt_11_14_to_11_18_gps_vins.csv"

        feet_per_meter = 3.28084
        y_valid = [-150,150]
        x_valid = [0,23000]
        ms_cutoff =  1000000000000


        # 2. Load raw GPS data and assemble into dict with one key per object
        # each entry will be array of time, array of x, array of y, array of acc, vehID
        vehicles = {}

        # TODO - get data from file
        dataframe = pd.read_csv(gps_data_file,delimiter = ",")

        ts   = dataframe["systime"].tolist()
        ts = [(item/1000 if item > ms_cutoff else item) for item in ts]
        lat  = dataframe["latitude"].tolist()
        long = dataframe["longitude"].tolist()
        vel  = dataframe["velocity"].tolist()
        acc  = dataframe["acceleration"].tolist()
        vehid   = dataframe["veh_id"].tolist()
        acc_setting = dataframe["acc_speed_setting"].tolist()

        ts = np.array(ts) #- (6*60*60) # convert from ms to s, then do UTC to CST offset
        lat = np.array(lat)
        long = np.array(long)
        vel = np.array(vel) * feet_per_meter
        acc = np.array(acc) * feet_per_meter
        vehid = np.array(vehid)
        # stack_data
        data = np.stack([ts,lat,long,vel,acc,acc_setting,vehid]).transpose(1,0)

        # sort by timestamp
        data = data[data[:,0].argsort(),:]

        # get unique vehids
        ids = np.unique(data[:,6])

        # assemble into dictionary
        vehicles = dict([(id,[]) for id in ids])
        for row in data:
            if row[0] < start_ts or row[0] > end_ts:
                continue
            id = row[6]
            vehicles[id].append(row)

            
        # lastly, stack each vehicle 
        new_vehicles = {}
        for key in vehicles.keys():
            if len(vehicles[key]) == 0:
                continue
            data = np.stack(vehicles[key])
            new_data = {
                "ts":data[:,0],
                "lat":data[:,1],
                "long":data[:,2],
                "vel":data[:,3],
                "acc":data[:,4]
                }
            new_vehicles[key] = new_data


        vehicles = new_vehicles

        # Nissan Rogue 183″ L x 72″ W x 67″ H
        l = 183/12
        w = 72/12
        h = 67/12

        # 3. Convert data into roadway coordinates
        trunc_vehicles = {}
        for vehid in vehicles.keys():
            
            #print("Converting vehicle {}".format(vehid))
            
            data = vehicles[vehid]
            vehid = int(vehid)
            # get data as roadway coordinates
            gps_pts  = torch.from_numpy(np.stack([data["lat"],data["long"]])).transpose(1,0)
            deriv_data = torch.from_numpy(np.stack([data["vel"],data["acc"],data["ts"]])).transpose(1,0)
            
            state_pts = WGS84_to_TN(gps_pts)
            state_pts = torch.cat((state_pts,torch.zeros(state_pts.shape[0],1)),dim = 1).unsqueeze(1)
            roadway_pts = self.hg.space_to_state(state_pts)
            
            veh_counter = 0
            cur_veh_data = [],[] # for xy and tva
            for r_idx in range (len(roadway_pts)):
                row = roadway_pts[r_idx]
                deriv_row = deriv_data[r_idx]
                
                if row[0] > x_valid[0] and row[0] < x_valid[1] and row[1] > y_valid[0] and row[1] < y_valid[1]:
                    cur_veh_data[0].append(row)
                    cur_veh_data[1].append(deriv_row)
                
                else:
                    # break off trajectory chunk
                    if len(cur_veh_data[0]) > 30:
                        this_road = torch.stack(cur_veh_data[0])
                        this_deriv = torch.stack(cur_veh_data[1])
                        
                        sub_key = "{}_{}".format(vehid,veh_counter)
                        trunc_vehicles[sub_key] =   {
                         "x": this_road[:,0],
                         "y": this_road[:,1],
                         "vel":this_deriv[:,0],
                         "acc":this_deriv[:,1],
                         "ts" :this_deriv[:,2],
                         "start":this_deriv[0,2],
                         "end":this_deriv[-1,2],
                         "id" :vehid,
                         "run":veh_counter,
                         "l":l,
                         "w":w,
                         "h":h,
                         }
                        
                        veh_counter += 1
                    cur_veh_data = [],[]
                           
                    
        self.gps = trunc_vehicles

    def load_gps_data2(self):
        """
        Load GPS file and convert to rcs. 
        
        
        self.gps - dict of dicts, each with CIRCLES_id,dimensions, and array of x,y,time
        """
        
        # collection = "637517698b5b68fc4fd40c77_CIRCLES_GPS"
        # db_param = {
        #       "host":"10.80.4.91",
        #       "port":27017,
        #       "username": "mongo-admin",
        #       "password": "i24-data-access",
        #       "database_name": "trajectories",      
        #       }
        
        
        # prd   = DBClient(**db_param,collection_name = collection)
        # preds = list(prd.read_query(None))
        def WGS84_to_TN(points):
            """
            Converts GPS coordiantes (WGS64 reference) to tennessee state plane coordinates (EPSG 2274).
            Transform is expected to be accurate within ~2 feet
            
            points array or tensor of size [n_pts,2]
            returns out - array or tensor of size [n_pts,2]
            """
            
            wgs84=pyproj.CRS("EPSG:4326")
            tnstate=pyproj.CRS("epsg:2274")
            out = pyproj.transform(wgs84,tnstate, points[:,0],points[:,1])
            out = np.array(out).transpose(1,0)
            
            if type(points) == torch.Tensor:
                out = torch.from_numpy(out)
                
            return out
        
        start_ts = 1668600000
        end_ts = start_ts + 60*60*4
        #gps_data_file = "/home/derek/Data/CIRCLES_GPS/CIRCLES_GPS_ALL.csv"
        gps_data_file = "/home/derek/Data/CIRCLES_GPS/gps_message_raw.csv"

        feet_per_meter = 3.28084
        y_valid = [-150,150]
        x_valid = [0,23000]
        ms_cutoff =  1000000000000


        # 2. Load raw GPS data and assemble into dict with one key per object
        # each entry will be array of time, array of x, array of y, array of acc, vehID
        vehicles = {}

        # TODO - get data from file
        dataframe = pd.read_csv(gps_data_file,delimiter = ",")

        ts   = dataframe["Systime"].tolist() 
        ts = [(item/1000 if item > ms_cutoff else item) for item in ts]
        lat  = dataframe["Lat"].tolist()
        long = dataframe["Long"].tolist()
        vehid   = dataframe["vin"].tolist()

        ts = np.array(ts)# - (6*60*60) # convert from ms to s, then do UTC to CST offset
        lat = np.array(lat)#.astype(float)
        long = np.array(long)#.astype(float)
        vehid = np.array(vehid)#.astype(int)
        # stack_data
        
        data = []
        for i in range(len(ts)):
            if i% 100000 == 0: print(i, i/len(ts))
            
            try:
                data.append(np.array([float(ts[i]), float(lat[i]),float(long[i]),float(vehid[i])]))
            except: 
                pass
            
            #if i > 10000: break
        data = np.stack(data)
        

        # sort by timestamp
        data = data[data[:,0].argsort(),:]
        # get unique vehids
        ids = np.unique(data[:,-1]).astype(int)

        # assemble into dictionary
        vehicles = dict([(id,[]) for id in ids])
        for row in data:
            try:
                if row[0] < start_ts or row[0] > end_ts:
                    continue
                #print("Got one")
                id = int(row[-1])
                vehicles[id].append(row)
            except:
                print("Bad Row")
                continue

            
        # lastly, stack each vehicle 
        new_vehicles = {}
        for key in vehicles.keys():
            if len(vehicles[key]) == 0:
                continue
            data = np.stack(vehicles[key])
            new_data = {
                "ts":data[:,0],
                "lat":data[:,1],
                "long":data[:,2]
                }
            new_vehicles[key] = new_data


        vehicles = new_vehicles

        # Nissan Rogue 183″ L x 72″ W x 67″ H
        l = 183/12
        w = 72/12
        h = 67/12

        # 3. Convert data into roadway coordinates
        trunc_vehicles = {}
        for vehid in vehicles.keys():
            
            #print("Converting vehicle {}".format(vehid))
            
            data = vehicles[vehid]
            vehid = int(vehid)
            # get data as roadway coordinates
            gps_pts  = torch.from_numpy(np.stack([data["lat"],data["long"]])).transpose(1,0)
            deriv_data = torch.from_numpy(np.stack([data["ts"]])).transpose(1,0)
            
            state_pts = WGS84_to_TN(gps_pts)
            state_pts = torch.cat((state_pts,torch.zeros(state_pts.shape[0],1)),dim = 1).unsqueeze(1)
            roadway_pts = self.hg.space_to_state(state_pts)
            
            veh_counter = 0
            cur_veh_data = [],[] # for xy and tva
            for r_idx in range (len(roadway_pts)):
                row = roadway_pts[r_idx]
                deriv_row = deriv_data[r_idx]
                
                if row[0] > x_valid[0] and row[0] < x_valid[1] and row[1] > y_valid[0] and row[1] < y_valid[1]:
                    cur_veh_data[0].append(row)
                    cur_veh_data[1].append(deriv_row)
                
                else:
                    # break off trajectory chunk
                    if len(cur_veh_data[0]) > 30:
                        this_road = torch.stack(cur_veh_data[0])
                        this_deriv = torch.stack(cur_veh_data[1])
                        
                        sub_key = "{}_{}".format(vehid,veh_counter)
                        trunc_vehicles[sub_key] =   {
                         "x": this_road[:,0],
                         "y": this_road[:,1],
                         "ts" :this_deriv[:,0],
                         "start":this_deriv[0,0],
                         "end":this_deriv[-1,0],
                         "id" :vehid,
                         "run":veh_counter,
                         "l":l,
                         "w":w,
                         "h":h,
                         }
                        
                        veh_counter += 1
                    cur_veh_data = [],[]
                           
                    
        self.gps = trunc_vehicles    

    def quit(self):      
        self.cont = False
        cv2.destroyAllWindows()
            
        self.save()
        
        
    def save(self):
        with open(self.save_file,"wb") as f:
            pickle.dump([self.data,self.objects],f)
        print("Saved annotations at {}".format(self.save_file))
        
        self.recount_objects()
        
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

        
    def toggle_cams(self,dir):
        """dir should be -1 or 1"""
        
        if self.active_cam + dir < len(self.camera_names) -1 and self.active_cam + dir >= 0:
            self.active_cam += dir
            self.plot()
       
        if self.toggle_auto:
            self.AUTO = True
               
    def next(self,stride = 1):
        """
        We assume all relevant frames are pre-buffered, so all we have to do is 
        Check whether there's another frame available to advance, advance the index,
        and then assign the indexed frame and timestamps
        """        
        self.label_buffer = None
        
        if self.toggle_auto:
            self.AUTO = True

        if self.frame_idx+stride < len(self.b.frames):
            self.frame_idx += stride
            self.label_buffer = copy.deepcopy(self.data),copy.deepcopy(self.objects)

        else:
            print("On last frame")
    
    
    
    def prev(self,stride = 1):
        self.label_buffer = None
        
        if self.toggle_auto:
            self.AUTO = True

        
        if self.frame_idx-stride >= 0 and len(self.b.frames[self.frame_idx-stride]) > 0:
            self.frame_idx -= stride     
            self.label_buffer = copy.deepcopy(self.data),copy.deepcopy(self.objects)


        else:
            print("Cannot return to previous frame. First frame or buffer limit")
                        
    def plot(self):        
        plot_frames = []
        #ranges = self.ranges
        
        
        for i in range(self.active_cam, self.active_cam+2):
           frame = self.b.frames[self.frame_idx][i]
           frame = frame.copy()
           
           
           # # get frame objects
           # # stack objects as tensor and aggregate other data for label
           positions = list(self.data[self.frame_idx].values())
           ids = list(self.data[self.frame_idx].keys())
           
           # stack object x,y,l,w,h,direction into state tensor
           boxes = []
           for p in range(len(positions)):
               pos = positions[p]
               id = ids[p]
               obj = self.objects[id]
               datum = torch.tensor([pos[0],pos[1],obj["l"],obj["w"],obj["h"],torch.sign(pos[1])])
               boxes.append(datum)
           
           #convert to image
           #im_boxes = self.hg.state_to_im(boxes,name = [self.camera_names[self.active_cam]])
           
           
           # remove all boxes entirely out of image ?
           # TODO maybe if necessary 

           for j in range(len(ids)):
              id = ids[j]
              if self.objects[id] is not None and self.objects[id]["gps_id"] is not None:
                   ids[j] = str(id) + " ({})".format(self.objects[id]["gps_id"])
           # plot boxes with ids
           if len(ids) > 0:
               boxes = torch.stack(boxes)
               self.hg.plot_state_boxes(frame,boxes,labels = ids,thickness = 1,name = [self.camera_names[i] for _ in boxes])
           
           
           
           if self.MASK:
               gps_boxes = []
               gps_ids = []
               frame_ts = self.b.ts[self.frame_idx][i]
               for key in self.gps.keys():
                   gpsob = self.gps[key]
                   #print(gpsob["start"].item(),gpsob["end"].item())
                   if gpsob["start"] < frame_ts and gpsob["end"] > frame_ts:
                       
                       # iterate through timestamps to find ts directly before and after current ts
                       for t in range(1,len(gpsob["ts"])):
                           if gpsob["ts"][t] > frame_ts:
                               break
                       
                       x1 = gpsob["x"][t-1]
                       x2 = gpsob["x"][t]
                       y1 = gpsob["y"][t-1]
                       y2 = gpsob["y"][t]
                       t1 = gpsob["ts"][t-1]
                       t2 = gpsob["ts"][t]
                       f1 = (t2-frame_ts)/(t2-t1)
                       f2 = (frame_ts-t1)/(t2-t1)
                       
                       x_interp =  x1*f1 + x2*f2
                       y_interp =  y1*f1 + y2*f2
                       
                       l = gpsob["l"]
                       w = gpsob["w"]
                       h = gpsob["h"]
                       
                       gps_box = torch.tensor([x_interp,y_interp,l,w,h,torch.sign(y_interp)])
                       gps_boxes.append(gps_box)
                       gps_ids.append(key)
              
               if len(gps_boxes) > 0:
                   gps_boxes = torch.stack(gps_boxes)
                   self.hg.plot_state_boxes(frame,gps_boxes,labels = gps_ids,thickness = 2,name = [self.camera_names[i] for _ in gps_boxes],color = (0,200,0))     
           #     # plot labels
           #     if self.TEXT:
           #         times = [item["timestamp"] for item in ts_data]
           #         classes = [item["class"] for item in ts_data]
           #         ids = [item["id"] for item in ts_data]
           #         directions = [item["direction"] for item in ts_data]
           #         directions = ["WB" if item == -1 else "EB" for item in directions]
           #         camera.frame = Data_Reader.plot_labels(None,frame,im_boxes,boxes,classes,ids,None,directions,times)
                   
            
           # if self.LANES:
                
           #          for lane in range(-60,60,12):
           #              # get polyline coordinates in space
           #              x_curve = np.linspace(20000,30000,4000)
           #              y_curve = np.ones(x_curve.shape) * lane
           #              zeros = np.zeros(x_curve.shape)
           #              curve = np.stack([x_curve,y_curve,zeros,zeros,zeros,zeros],axis = 1)
           #              curve = torch.from_numpy(curve)
           #              cname = [camera.name for i in range(x_curve.shape[0])]
           #              curve_im = self.hg.state_to_im(curve,name = cname)
           #              #curve_im = curve_im[:,0,:]
                       
           #              mask = ((curve_im[:,:,0] > 0).int() + (curve_im[:,:,0] < 1920).int() + (curve_im[:,:,1] > 0).int() + (curve_im[:,:,1] < 1080).int()) == 4
           #              curve_im = curve_im[mask,:]
                       
           #              curve_im = curve_im.data.numpy().astype(int)
           #              cv2.polylines(frame,[curve_im],False,(255,100,0),1)
                  
           #          for tick in range(20000,30000,10):
           #                      y_curve = np.linspace(-60,60,8)
           #                      x_curve = y_curve *0 + tick
           #                      z_curve = y_curve *0
           #                      curve = np.stack([x_curve,y_curve,z_curve,z_curve,z_curve,z_curve],axis = 1)
           #                      curve = torch.from_numpy(curve)
           #                      cname = [camera.name for i in range(x_curve.shape[0])]
           #                      curve_im = self.hg.state_to_im(curve,name = cname)
                               
           #                      mask = ((curve_im[:,:,0] > 0).int() + (curve_im[:,:,0] < 1920).int() + (curve_im[:,:,1] > 0).int() + (curve_im[:,:,1] < 1080).int()) == 4
           #                      curve_im = curve_im[mask,:]
                               
           #                      curve_im = curve_im.data.numpy().astype(int)
                               
           #                      th = 1
           #                      color = (150,150,150)
           #                      if tick % 200 == 0:
           #                          th = 2
           #                          color = (255,100,0)
           #                      elif tick % 40 == 0:
           #                           th = 2
                                    
           #                      cv2.polylines(frame,[curve_im],False,color,th)
                   
                
           
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
            row = i // n_row
            col = i % n_row
            
            cat_im[col*1080:(col+1)*1080,row*1920:(row+1)*1920,:] = im
            
        # view frame and if necessary write to file
        cat_im /= 255.0
        
        
        self.plot_frame = cat_im
        
    # def output_vid(self):
    #     self.LANES = False
    #     self.MASK = True
    #     #self.active_cam = 6
    #     directory = "video/scene2/{}".format(str(self.active_cam))
    
    #     while self.frame_idx < self.last_frame:
            
    #         # while self.active_cam < len(self.camera_names):
    #         if not os.path.exists(directory):
    #             os.mkdir(directory)
        
    #         self.plot()
    #         #resize_im = cv2.resize(self.plot_frame*255,(3840,2160))
    #         cv2.imwrite("{}/{}.png".format(directory,str(self.frame_idx).zfill(4)),self.plot_frame*255)
            
            
    #         # warning, this de-syncs the cameras and is pretty dangerous for that reason
    #         self.advance_2(camera_idx = self.active_cam)
    #         self.frame_idx += 1
            
    #         if self.frame_idx % 10 == 0:
    #             print("On frame {}".format(self.frame_idx))
                
                
    #     # convert frames to video
    #     all_files = os.listdir(directory)
    #     all_files.sort()
    #     for filename in all_files:
    #         filename = os.path.join(directory, filename)
    #         img = cv2.imread(filename)
    #         height, width, layers = img.shape
    #         size = (width,height)
    #         break
        
    #     n = 0
    #     f_name = os.path.join("/home/derek/Desktop",'{}_{}.mp4'.format(self.cameras[self.active_cam].name,self.cameras[self.active_cam+1].name))
    #     temp_name =  os.path.join("/home/derek/Desktop",'temp.mp4')
        
    #     out = cv2.VideoWriter(temp_name,cv2.VideoWriter_fourcc(*'mp4v'), 8, size)
         
    #     for filename in all_files:
    #         filename = os.path.join(directory, filename)
    #         img = cv2.imread(filename)
    #         out.write(img)
    #         print("Wrote frame {}".format(n))
    #         n += 1
            
            
    #     out.release()
        
    #     os.system("/usr/bin/ffmpeg -i {} -vcodec libx264 {}".format(temp_name,f_name))
        
    def find_furthest_gps(self,direction = -1):
        """
        Finds the GPS vehicle that is furthest along on the roadway (depending on direction of travel)
        Then finds the cameras bookending that object's position.
        Excludes all objects associated with a labeled object that has a sink camera
        """
        
        #1. return to first frame in buffer
        stride = min(self.frame_idx, self.b.buffer_limit)
        self.prev(stride = stride)
        
        #2. sample all object positions that are currently active; filter by direction
        gps_ids = []
        gps_pos = []
        frame_ts = self.b.ts[self.frame_idx][0] # just use first frame's timestamp
        for key in self.gps.keys():
            gpsob = self.gps[key]
            #print(gpsob["start"].item(),gpsob["end"].item())
            if gpsob["start"] < frame_ts and gpsob["end"] > frame_ts and torch.sign(gpsob["y"][0]) == direction:
                
                # iterate through timestamps to find ts directly before and after current ts
                for t in range(1,len(gpsob["ts"])):
                    if gpsob["ts"][t] > frame_ts:
                        break
                
                gps_pos.append(gpsob["x"][t])
                gps_ids.append(key)
    
        #3. filter by associated sink objects
        furthest_id = None
        furthest_pos = -np.inf
        
        for gidx,gpsid in enumerate(gps_ids):
            valid = True
            for obj in self.objects:
                if self.objects[obj]["gps_id"] == gpsid:
                    valid = False
                    break
            if valid:
                if gps_pos[gidx]*direction > furthest_pos:
                    furthest_pos = gps_pos[gidx]*direction
                    furthest_id = gpsid
        
        #4. after selecting object, find relevant camera for current position
        directionstr = "EB" if direction == 1 else "WB"
        furthest_pos *= direction # make positive again
        for cidx in range(len(self.camera_names)):
            cam = self.camera_names[int(-1*direction*cidx)] # need to index in reverse order for EB 
            
            if direction == 1: 
                min_x = self.extents["{}_{}".format(cam,directionstr)][1]
                if furthest_pos < min_x: break
            else:
                min_x = self.extents["{}_{}".format(cam,directionstr)][0]
                if furthest_pos > min_x: break
        
        #5. advance
        self.active_cam = cidx
        print("Next furthest gps vehicle is {}, which will be visible in {} or {}".format(furthest_id,self.camera_names[self.active_cam],self.camera_names[self.active_cam+1]))
        
    def associate(self,id,gps_id):
        try:
            self.objects[id]["gps_id"] = gps_id
            self.active_command == "COPY PASTE"
        except:
            pass
        
        
    def smart_advance(self):
        """ Advance to next camera if there is an annotation for this camera/frame pair, else just advance a frame"""
        
        # get active (copied) object
        if self.copied_box is None:
            self.next()
            self.plot()
            return
      
        obj_id = self.copied_box[0]
        
        
        # if no annotation yet, just advance the frame
        if obj_id not in self.data[self.frame_idx].keys():
            self.next()
            self.plot()
            return
            
        
        # get associated GPS object
        gps_id = self.objects[obj_id]["gps_id"]
        
        
        # get next camera
        if self.active_cam < len(self.camera_names) -1:
            self.toggle_cams(1)
            cam = self.camera_names[self.active_cam]
            direction = torch.sign(self.copied_box[1][1])
            directionstr = "EB" if direction == 1 else "WB"
            
        else:
            #self.active_cam = 0
            return
        
        if gps_id is None: # for objects that don't have a tied GPS
            # advance according to self.frame_gaps for active cam
            self.next(stride = int(self.frame_gaps[self.active_cam]))
            return
        
        
        # get active camera's leading edge from hg
        if direction == 1: 
            min_x = self.extents["{}_{}".format(cam,directionstr)][0]
        else:
            min_x = self.extents["{}_{}".format(cam,directionstr)][1]
        
        
        # find the first frame in which GPS object is past the appropriate time to advance s.t. the GPS object is in the camera's FOV
        for tidx in range(len(self.gps[gps_id]["x"])):
            if direction == 1 and self.gps[gps_id]["x"][tidx] > min_x:
                break
            if direction == -1 and self.gps[gps_id]["x"][tidx] < min_x:
                break
            
        next_time = self.gps[gps_id]["ts"][tidx]
        
        
        # advance to that frame (or give warning if not buffered)
        for f_idx in range(len(self.b.ts)):
            if self.b.ts[f_idx][self.active_cam] > next_time:
                break
        stride = f_idx - self.frame_idx
        self.next(stride = stride)
        
        
        
    def add(self,obj_idx,location):
        
        xy = self.box_to_state(location)[0,:].data.numpy()
        
        # create new object
        # 2022 nissan rogue dimensions : 183″ L x 72″ W x 67″ H
        obj = {
            "l": 183/12,
            "w": 72/12,
            "h": 67/12,
            "class":"midsize",
            "source":self.clicked_camera,
            "sink": None,
            "complete":0,
            "gps_id":None
            }
        
        timestamp  = 0 # TODO
        datum = torch.tensor([float(xy[0]),float(xy[1]),timestamp],dtype = torch.double)
        
        self.objects[obj_idx] = obj
        self.data[self.frame_idx][obj_idx] = datum
        
        self.copied_box = None
        self.copy_paste(location,obj_idx = obj_idx)
        self.active_command = "COPY PASTE"
    
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
            cam = self.camera_names[self.active_cam+1]
            point[0] -= 1920
            point[2] -= 1920
        else:
            cam = self.camera_names[self.active_cam]

        point1 = torch.tensor([point[0],point[1]]).unsqueeze(0).unsqueeze(0).repeat(1,8,1)
        point2 = torch.tensor([point[2],point[3]]).unsqueeze(0).unsqueeze(0).repeat(1,8,1)
        point = torch.cat((point1,point2),dim = 0)
        
        state_point = self.hg.im_to_state(point,name = [cam,cam], heights = torch.tensor([0]))
        
        return state_point[:,:2]
    
        
    def shift(self,obj_idx,box, dx = 0, dy = 0):
        
        item =  self.data[self.frame_idx].get(obj_idx)
        
        if item is None:
            return
        
        
        
        if dx == 0 and dy == 0:
            state_box = self.box_to_state(box)
            dx = state_box[1,0] - state_box[0,0]
            dy = state_box[1,1] - state_box[0,1]
        
        
        if np.abs(dy) > np.abs(dx): # shift y if greater magnitude of change
            item[1] += dy
                
        else:
            item[0] += dx
             
                                
    
    def change_class(self,obj_id,cls):
        self.objects[obj_id]["class"] = cls
    
    def paste_in_2D_bbox(self,box):
        """
        Finds best position for copied box such that the image projection of that box 
        matches the 2D bbox with minimal MSE error
        """
        
        if self.copied_box is None:
            return
        
        obj_id = self.copied_box[0]
        obj = self.objects[obj_id]

        base = self.copied_box[1].clone()
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
                    shift_box = torch.tensor([i,j, obj["l"],obj["w"],obj["h"],torch.sign(base[1])])
                    shifts.append(shift_box)
        
            # convert shifted grid of boxes into 2D space
            shifts = torch.stack(shifts)
            cname = [self.clicked_camera for _ in range(shifts.shape[0])]
            boxes_space = self.hg.state_to_im(shifts,name = cname)
            
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
        base[0] = self.safe(center[0])
        base[1] = self.safe(center[1])
        self.data[self.frame_idx][obj_id] = base
        
    def automate(self,obj_id):
        """
        Crop locally around expected box coordinates based on constant velocity
        assumption. Localize on this location. Use the resulting 2D bbox to align 3D template
        Repeat at regularly spaced intervals until expected object location is out of frame
        """
        # store base box for future copy ops
        cam = self.clicked_camera
        prev_box = self.data[self.frame_idx].get(obj_id)
        
        if prev_box is None:
            return
        obj = self.objects[obj_id]
        for c_idx in range(len(self.camera_names)):
            if self.camera_names[c_idx] == cam:
                break
        
        crop_state = torch.tensor([prev_box[0],prev_box[1],obj["l"],obj["w"],obj["h"],torch.sign(prev_box[0])]).unsqueeze(0)
        boxes_space = self.hg.state_to_im(crop_state,name = [cam])
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
        frame = self.b.frames[self.frame_idx][c_idx].copy()
        
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
    
    def crop_detect(self,frame,crop,ber = 1.5,cs = 112):
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
        
        key = obj_idx
        item =  self.objects[key]
        if item is None:
            return
        
        if dx == 0 and dy == 0:
            state_box = self.box_to_state(box)
            dx = state_box[1,0] - state_box[0,0]
            dy = state_box[1,1] - state_box[0,1]
            dh = -(box[3] - box[1]) * 0.02 # we say that 50 pixels in y direction = 1 foot of change
        else:
            dh = dy
        
        
        try:
            l = self.objects[key]["l"]
            w = self.objects[key]["w"]
            h = self.objects[key]["h"]
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
        
        item[relevant_key] = relevant_change
   
        # also adjust the copied box if necessary
        if self.copied_box is not None and self.copied_box[0] == obj_idx:
             self.objects[obj_idx][relevant_key] = relevant_change
             
    
    def copy_paste(self,point,obj_idx = None):     
        if self.copied_box is None:
            if obj_idx is None:
                obj_idx = self.find_box(point)
            
            if obj_idx is None:
                return
            
            state_point = self.box_to_state(point)[0]
            
            obj =  self.data[self.frame_idx].get(obj_idx)
            
            if obj is None:
                return
            
            base_box = obj.clone()
            
            # save the copied box
            self.copied_box = (obj_idx,base_box,[state_point[0],state_point[1]].copy())
            
        
        else: # paste the copied box
            start = time.time()
            state_point = self.box_to_state(point)[0]
            
            obj_idx = self.copied_box[0]
            new_obj = copy.deepcopy(self.copied_box[1])


            dx = state_point[0] - self.copied_box[2][0] 
            dy = state_point[1] - self.copied_box[2][1] 
            new_obj[0] += dx
            new_obj[1] += dy
            new_obj[0]  = new_obj[0].item()
            new_obj[1]  = new_obj[1].item()
            new_obj[2] = 0 #TODO FIX with real timeself.all_ts[self.frame_idx][self.clicked_camera]

                
            self.data[self.frame_idx][obj_idx] = new_obj
            
            self.frame_gaps[self.active_cam] = self.frame_idx - self.prev_cam_label_frame
            self.prev_cam_label_frame = self.frame_idx
            
            if self.AUTO:
                self.automate(obj_idx)
                self.AUTO = False
            

            
    # def interpolate(self,obj_idx,verbose = True):
        
    #     #self.print_all(obj_idx)
        
    #     for cur_cam in self.cameras:
    #         cam_name = cur_cam.name
        
    #         prev_idx = -1
    #         prev_box = None
    #         for f_idx in range(0,len(self.data)):
    #             frame_data = self.data[f_idx]
                    
    #             # get  obj_idx box for this frame if there is one
    #             cur_box = None
    #             for obj in frame_data.values():
    #                 if obj["id"] == obj_idx and obj["camera"] == cam_name:
    #                     del cur_box
    #                     cur_box = copy.deepcopy(obj)
    #                     break
                    
    #             if prev_box is not None and cur_box is not None:
                    
                    
    #                 for inter_idx in range(prev_idx+1, f_idx):   
                        
    #                     # doesn't assume all frames are evenly spaced in time
    #                     t1 = self.all_ts[prev_idx][cam_name]
    #                     t2 = self.all_ts[f_idx][cam_name]
    #                     ti = self.all_ts[inter_idx][cam_name]
    #                     p1 = float(t2 - ti) / float(t2 - t1)
    #                     p2 = 1.0 - p1                    
                        
                        
    #                     new_obj = {
    #                         "x": p1 * prev_box["x"] + p2 * cur_box["x"],
    #                         "y": p1 * prev_box["y"] + p2 * cur_box["y"],
    #                         "l": prev_box["l"],
    #                         "w": prev_box["w"],
    #                         "h": prev_box["h"],
    #                         "direction": prev_box["direction"],
    #                         "id": obj_idx,
    #                         "class": prev_box["class"],
    #                         "timestamp": self.all_ts[inter_idx][cam_name],
    #                         "camera":cam_name,
    #                         "gen":"Interpolation"
    #                         }
                        
    #                     key = "{}_{}".format(cam_name,obj_idx)
    #                     self.data[inter_idx][key] = new_obj
                
    #             # lastly, update prev_frame
    #             if cur_box is not None:
    #                 prev_idx = f_idx 
    #                 del prev_box
    #                 prev_box = copy.deepcopy(cur_box)
        
    #     if verbose: print("Interpolated boxes for object {}".format(obj_idx))
        
    def correct_homography_Z(self,box):
        dx = self.safe(box[2]-box[0]) 
        if np.abs(dx) > 100:
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
        if direction == 1: self.hg.correspondence[self.clicked_camera + "_EB"]["P"][:,2] *= sign*delta
        else:              self.hg.correspondence[self.clicked_camera + "_WB"]["P"][:,2] *= sign*delta
            
        #self.hg.save(self.hg_save_file)
        
    def sink_active_object(self):
        if self.copied_box is None:
            return
        
        obj_id = self.copied_box[0]
        self.objects[obj_id]["sink"] = self.clicked_camera
        self.copied_box = None
        
        # count number of annotations for this object to ensure there are an appropriate number
        
        count = 0
        for fidx in range(len(self.data)):
            if obj_id in self.data[fidx].keys():
                count += 1
                
        
        source_idx = -1
        sink_idx = -1
        source = self.objects[obj_id]["source"]
        sink = self.objects[obj_id]["sink"]
        print("Assigned sink camera {} to object {}".format(sink,obj_id))

        # get number of cameras between source and sink
        for i in range(len(self.camera_names)):
            if self.camera_names[i] == source:
                source_idx = i
            elif self.camera_names[i] == sink:
                sink_idx = i
                
        probable_count = np.abs(sink_idx - source_idx) + 1
        self.objects[obj_id]["complete"] = 1
        
        if probable_count > count:
            print("Warning: object {} is probably missing an annotation: {} annotations for {} cameras".format(obj_id,count,probable_count))
            self.objects[obj_id]["complete"] = 0.5
            
        # advance to next object
        self.find_furthest_gps()
        self.save()
     
    def recount_objects(self):
        for obj_id in self.objects.keys():

            source_idx = -1
            sink_idx = -1
            source = self.objects[obj_id]["source"]
            sink = self.objects[obj_id]["sink"]
            if sink is None:
                continue
        
            count = 0
            for fidx in range(len(self.data)):
                if obj_id in self.data[fidx].keys():
                    count += 1
                    
            # get number of cameras between source and sink
            for i in range(len(self.camera_names)):
                if self.camera_names[i] == source:
                    source_idx = i
                elif self.camera_names[i] == sink:
                    sink_idx = i
                    
            probable_count = np.abs(sink_idx - source_idx) + 1
            self.objects[obj_id]["complete"] = 1
            
            if probable_count != count:
                print("Warning: object {} is probably missing an annotation: {} annotations for {} cameras".format(obj_id,count,probable_count))
                self.objects[obj_id]["complete"] = 0.5
                
    def hop(self):
        self.next(stride = 3)
            
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
                obj =  self.data[frame_idx].get(obj_idx)
                if obj is not None:
                    del self.data[frame_idx][obj_idx]
            except KeyError:
                pass
            frame_idx += 1
        
        if obj_idx in self.objects.keys():
            del self.objects[obj_idx]
        
    def on_mouse(self,event, x, y, flags, params):
       if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
         self.start_point = (x,y)
         self.clicked = True 
       elif event == cv.EVENT_LBUTTONUP:
            box = np.array([self.start_point[0],self.start_point[1],x,y])
            self.new = box
            self.clicked = False
            
            
            if x > 1920:
                self.clicked_camera = self.camera_names[self.active_cam+1]
                self.clicked_idx = self.active_cam + 1
            else:
                self.clicked_camera = self.camera_names[self.active_cam]
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
            cam = self.camera_names[self.active_cam+1]
            point[0] -= 1920
        else:
            cam = self.camera_names[self.active_cam]

        point = torch.tensor([point[0],point[1]]).unsqueeze(0).unsqueeze(0).repeat(1,8,1)
        state_point = self.hg.im_to_state(point,name = [cam], heights = torch.tensor([0])).squeeze(0)
        
        min_dist = np.inf
        min_id = None
        
        for b_id  in self.data[self.frame_idx].keys():
            box = self.data[self.frame_idx][b_id]
            dist = (box[0]- state_point[0] )**2 + (box[1] - state_point[1])**2
            if dist < min_dist:
                min_dist = dist
                min_id = b_id
        
        return min_id

    def keyboard_input(self):
        keys = ""
        letters = string.ascii_lowercase + string.digits + "_"
        while True:
            key = cv2.waitKey(1)
            for letter in letters:
                if key == ord(letter):
                    keys = keys + letter
            if key == ord("\n") or key == ord("\r"):
                break
        return keys    
      

    
    def undo(self):
        if self.label_buffer is not None:
            self.data = self.label_buffer[0]
            self.objects = self.label_buffer[1]
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
        
            for obj_idx in range(self.next_object_id):
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
            
            for cam in self.camera_names:
                
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
        boxes_im = self.hg.state_to_im(boxes.float(),name = cameras)
        
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
    
    
    # def gen_trajectories(self):
    #     spline_data = copy.deepcopy(self.data)
    #     for idx in range(self.get_unused_id()):
    #         if self.splines is None:
    #             t,x_spline,y_spline,ape,mpe = self.create_trajectory(idx)
    #         else:
    #             x_spline,y_spline = self.splines[idx]
    #         if x_spline is None: 
    #             continue
    #             print("skip")
                
    #         for f_idx in range(len(spline_data)):
    #             for cam in self.camera_names:
    #                 key = "{}_{}".format(cam,idx)
                    
    #                 box = spline_data[f_idx].get(key)
    #                 if box is not None:
    #                     box["x"] = x_spline(box["timestamp"]).item()
    #                     box["y"] = y_spline(box["timestamp"]).item()
            
    #     self.spline_data = spline_data
      
    # def get_splines(self,plot = True,metric = "ape"):
    #     splines = []
    #     apes = []
    #     ases = []
    #     for idx in range(self.get_unused_id()):
    #         t,x_spline,y_spline,ase,ape = self.create_trajectory(idx,plot = plot,metric = metric)
    #         splines.append([x_spline,y_spline])
    #         if ape is not None:
    #             ases.append(ase)
    #             apes.append(ape)
                
    #     print("Spline errors: {}ft ase, {}px ape".format(sum(ases)/len(ases),sum(apes)/len(apes)))    
    #     self.splines = splines 
     
    # def adjust_boxes_with_trajectories(self,max_shift_x = 2,max_shift_y = 2,verbose = False):
    #     """
    #     Adjust each box by up to max_shift pixels in x and y direction towards the best-fit spline
    #     """               
        
    #     print("NOT IMPLEMENTED")
        
    #     pixel_shifts = []
    #     try:
    #         self.splines
    #     except:
    #         print("Splines not yet fit - cannot adjust boxes using splines")
    #         return
        
    #     for f_idx,frame_data in enumerate(self.data):
            
    #         if f_idx > self.last_frame:
    #             break
    #         if len(self.data[f_idx]) == 0:
    #             continue
            
    #         ids = [obj["id"] for obj in frame_data.values()]
    #         cameras = [obj["camera"] for obj in frame_data.values()]
    #         boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"],obj["timestamp"]],dtype = torch.double) for obj in frame_data.values()])

                
    #         boxes_im = self.hg.state_to_im(boxes.float(),name = cameras)
    #         boxes = boxes.data.numpy()
            
    #         # get feet per pixel ratio for y
    #         y_diff = torch.sum(torch.pow(torch.mean(boxes_im[:,[0,2],:] - boxes_im[:,[1,3],:],dim = 1),2),dim = 1).sqrt()        
    #         y_lim = (boxes[:,3] / y_diff * max_shift_y).data.numpy()
            
    #         # get feet per pixel ration for x        
    #         x_diff = torch.sum(torch.pow(torch.mean(boxes_im[:,[0,1],:] - boxes_im[:,[2,3],:],dim = 1),2),dim = 1).sqrt()
    #         x_lim = (boxes[:,2] / x_diff * max_shift_x).data.numpy()
            
    #         # for each box
    #         for i in range(len(ids)):
                
    #             if self.splines[ids[i]][0] is not None:
    #                 #get xpos on spline
    #                 ts = boxes[i,6]
    #                 x_spl = self.splines[ids[i]][0](ts)
                    
    #                 # get difference
    #                 x_diff = x_spl - boxes[i,0]
                    
    #                 if x_diff < -x_lim[i]:
    #                     x_diff = -x_lim[i]
    #                 elif x_diff > x_lim[i]:
    #                     x_diff = x_lim[i]
                    
    #                 # move box either to spline or to x_lim
    #                 key = "{}_{}".format(cameras[i],ids[i])
    #                 frame_data[key]["x"] += x_diff
                
    #             if self.splines[ids[i]][1] is not None:
    #                 #get ypos on spline
    #                 ts = boxes[i,6]
    #                 y_spl = self.splines[ids[i]][1](ts)
                    
    #                 # get difference
    #                 y_diff = y_spl - boxes[i,1]
                
    #                 if y_diff < -y_lim[i]:
    #                     y_diff = -y_lim[i]
    #                 elif y_diff > y_lim[i]:
    #                     y_diff = y_lim[i]
                    
    #                 # move box either to spline or to x_lim
    #                 key = "{}_{}".format(cameras[i],ids[i])
    #                 frame_data[key]["y"] += y_diff
                    
    #                 pixel_shifts.append(np.sqrt(x_diff**2 + y_diff**2))
            
    #         if verbose and (f_idx %100 == 0): print("Adusted boxes for frame {}".format(f_idx))
            
    #     return pixel_shifts
           
                        
    # def adjust_ts_with_trajectories(self,max_shift = 1,trials = 101,overwrite_ts_data = False,metric = "ape", use_running_error = False,verbose = True):
        
    #     print("NOT IMPLEMENTED")
        
    #     splines = self.splines
        
    #     running_error = [self.ts_bias[i] for i in range(len(self.camera_names))]
    #     if not use_running_error:
    #         running_error = [0 for item in running_error]

    #     # for each camera, for each frame, get labels
    #     for f_idx,frame_data in enumerate(self.data):
            
    #         if f_idx % 100 == 0:
    #             print("Adjusting ts for frame {}".format(f_idx))
                
    #         if f_idx > self.last_frame and len(self.data[f_idx]) == 0:
    #             break 
    #         for c_idx, cam in enumerate(self.camera_names):
                
    #             # get all frame/camera labels
    #             objs = []
    #             ids = []
    #             for idx in range(self.get_unused_id()):
    #                 key = "{}_{}".format(cam,idx)
    #                 obj = frame_data.get(key)
                    
    #                 if obj is not None:
    #                     objs.append(obj)
    #                     ids.append(idx)
                
    #             id_splines = [splines[id][0] for id in ids]
    #             yid_splines = [splines[id][1] for id in ids]
                
    #             if len(id_splines) == 0:
    #                 continue
                
    #             # get x_weights
    #             boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"],obj["timestamp"]],dtype = torch.double) for obj in objs])
                
    #             # a one foot change in space results in a _ pixel change in image spac         
    #             # convert boxes to im space - n_boxes x 8 x 2 in order: fbr,fbl,bbr,bbl,ftr,ftl,fbr,fbl
    #             cname = [cam for i in range(len(boxes))]
    #             #boxes_im = self.hg.state_to_im(boxes.float(),name = cname)
    #             boxes_im = self.hg.state_to_im(boxes.float(),name = cam)

                
                
    #             # x_weight = length in pixels / legnth in feet
               
                
    #             # remove all objects without valid splines from consideration
    #             keep = [True if item is not None else False for item in id_splines]
    #             keep_splines = []
    #             for i in range(len(keep)):
    #                 if keep[i]:
    #                     keep_splines.append(id_splines[i])
    #             id_splines = keep_splines
                
                
    #             x_weight = 1
    #             if True:
    #                 #x_diff = torch.mean(torch.sum(torch.pow(boxes_im[:,[0,1],:] - boxes_im[:,[2,3],:],2), dim = 2).sqrt(),dim = 1)
    #                 x_diff = torch.sum(torch.pow(torch.mean(boxes_im[:,[0,1],:] - boxes_im[:,[2,3],:],dim = 1),2),dim = 1).sqrt()
    #                 x_weight = x_diff / boxes[:,2]  
    #                 x_weight = x_weight[keep]
                
    #             boxes = boxes[keep]

                
    #             if len(id_splines) == 0:
    #                 continue
                
    #             best_time = copy.deepcopy(self.all_ts[f_idx][cam])
    #             best_error = np.inf
                
    #             #get initial error
    #             xs = torch.tensor([id_splines[i](best_time).item() for i in range(len(id_splines))])
    #             xlabel = boxes[:,0]
    #             if metric == "ape":
    #                     init_error = torch.mean(((xs-xlabel)*x_weight).pow(2)).sqrt()
    #             else:
    #                     init_error =  torch.max(((xs-xlabel)*x_weight).pow(2)).sqrt()            
                
    #             for shift in np.linspace(-max_shift,max_shift,trials):
    #                 if use_running_error:
    #                     shift += running_error[c_idx]
    #                 new_time = self.all_ts[f_idx][cam] + shift 
                
    #                 # for each timestamp shift, compute the error between spline position and label position
    #                 xs = torch.tensor([id_splines[i](new_time).item() for i in range(len(id_splines))])
    #                 xlabel = boxes[:,0]
                    

                    
    #                 if metric == "ape":
    #                     mse = torch.mean(((xs-xlabel)*x_weight).pow(2)).sqrt()
    #                 else:
    #                     mse =  torch.max(((xs-xlabel)*x_weight).pow(2)).sqrt()
                        
    #                 if mse < best_error:
    #                     best_error = mse
    #                     best_time = new_time
                
    #             if verbose:
    #                 print("{} frame {}: shifted time by {:.3f}s --- {:.2f}x initial error".format(cam,f_idx,best_time - self.all_ts[f_idx][cam],best_error/init_error))
                
    #             for idx in range(self.get_unused_id()):
    #                 key = "{}_{}".format(cam,idx)
    #                 obj = frame_data.get(key)
                    
    #                 if obj is not None:
    #                     self.data[f_idx][key]["timestamp"] = best_time
                
    #             if overwrite_ts_data:
    #                 self.all_ts[f_idx][cam] = best_time

    #             if use_running_error:
    #                 running_error[c_idx] += shift
        
        
        
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
        
            for obj_idx in range(self.next_object_id):
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

    # def replace_timestamps(self):
    #     """
    #     Replace timestamps with timestamps at nominal framerate. 
    #     Then reinterpolate boxes based on these timestamps
    #     """
        
    #     # get nominal framerate for each camera
    #     start_ts = [self.all_ts[0][key] for key in self.camera_names]
    #     spans = [self.all_ts[-1][key] - self.all_ts[0][key] for key in self.camera_names]
    #     frame_count = len(self.all_ts)
        
    #     fps_nom = [frame_count/spans[i] for i in range(len(spans))]
    #     print (fps_nom)
        
    #     # modify self.ts for each camera
    #     for i in range(len(self.all_ts)):
    #         for j in range(len(self.camera_names)):
    #             key = self.camera_names[j]
    #             self.all_ts[i][key] = start_ts[j] + 1.0/fps_nom[j]*i
        
    #     # ts bias dictionary
    #     #ts_bias_dict =  dict([ (self.camera_names[i],self.ts_bias[i]) for i in range(len(self.ts_bias))])
        
    #     new_data = []
    #     # overwrite timestamps in all labels
    #     for f_idx in range(len(self.data)):
    #         new_data.append({})
    #         for key in self.data[f_idx]:
    #             cam = self.data[f_idx][key]["camera"]
    #             self.data[f_idx][key]["timestamp"] = self.all_ts[f_idx][cam] #+ ts_bias_dict[cam]
                
    #             if "gen" in self.data[f_idx][key].keys() and self.data[f_idx][key]["gen"] != "Manual":
    #                 continue
    #             new_data[f_idx][key] =  copy.deepcopy(self.data[f_idx][key])
        
        
    #     # delete and reinterpolate boxes as necessary
    #     self.data = new_data
    #     for i in range(self.get_unused_id()):
    #         self.interpolate(i,verbose = False)
        
    #     self.plot_all_trajectories()
    #     print("Replaced timestamps")
        
    
    # def unbias_timestamps(self):
    #     """
    #     Replace timestamps with timestamps at nominal framerate. 
    #     Then reinterpolate boxes based on these timestamps
    #     """
        
    #     # get nominal framerate for each camera
    #     self.estimate_ts_bias()
        
    #     print(self.ts_bias)
    #     print(self.camera_names)
        
    #     # ts bias dictionary
    #     ts_bias_dict =  dict([ (self.camera_names[i],self.ts_bias[i]) for i in range(len(self.camera_names))])
        
    #     new_data = []
    #     # overwrite timestamps in all labels
    #     for f_idx in range(len(self.data)):
    #         new_data.append({})
    #         for key in self.data[f_idx]:
    #             cam = self.data[f_idx][key]["camera"]
    #             try:
    #                 self.data[f_idx][key]["timestamp"] += ts_bias_dict[cam]
    #             except:
    #                 print("KeyError")
                
    #     # overwrite timestamps in self.all_ts
    #     for f_idx in range(len(self.all_ts)):
    #         for i in range(len(self.camera_names)):
    #             key = self.camera_names[i]
    #             try:
    #                 self.all_ts[f_idx][key] += self.ts_bias[i]
    #             except KeyError:
    #                 print("KeyError")
    #                 pass
                
    #     print("Un-biased timestamps")
            
    # def replace_homgraphy(self):
    #     raise Exception ("Dont Call this!")
        
    #     # get replacement homography
    #     hid = 5
    #     with open("EB_homography{}.cpkl".format(hid),"rb") as f:
    #         hg1 = pickle.load(f) 
    #     with open("WB_homography{}.cpkl".format(hid),"rb") as f:
    #         hg2 = pickle.load(f) 
    #     hg_new  = Homography_Wrapper(hg1=hg1,hg2=hg2)    
    
    #     # # copy height scale values from old homography to new homography
    #     # for corr in hg_new.hg1.correspondence.keys():
    #     #     if corr in self.hg.hg1.correspondence.keys():
    #     #         hg_new.hg1.correspondence[corr]["P"][]
    #     # if direction == 1:
    #     #         self.hg.hg1.correspondence[self.clicked_camera]["P"][:,2] *= sign*delta
    #     #     else:   
    #     #         self.hg.hg2.correspondence[self.clicked_camera]["P"][:,2] *= sign*delta
        
    #     # create new copy of data
    #     new_data = []
    #     all_errors = [0]
    #     # for each frame in data
    #     for f_idx,frame_data in enumerate(self.data):
            
    #         if f_idx > self.last_frame:
    #             break
            
    #         new_data.append({})
    #         if f_idx % 100 == 0:
    #             print("On frame {}. Average error so far: {}".format(f_idx,sum(all_errors)/len(all_errors)))
            
    #         # for each camera in frame data
    #         for camera in self.cameras:
    #             cam = camera.name
                
    #             # for each box in camera 
    #             for obj_idx in range(self.get_unused_id()):
    #                key = "{}_{}".format(cam,obj_idx)
    #                if frame_data.get(key): 
    #                    obj = frame_data.get(key)
                       
    #                    # if box was manually drawn
    #                    if obj["gen"] == "Manual":
                           
    #                        base = copy.deepcopy(obj)
                           
    #                        # get old box image coordinates
    #                        old_box = torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"]]).unsqueeze(0)
    #                        old_box_im = self.hg.state_to_im(old_box,name = cam)
                           
    #                        # find new box that minimizes the reprojection error of corner coordinates
    #                        center = obj["x"],obj["y"]
    #                        search_rad = 50
    #                        grid_size = 11
    #                        while search_rad > 1:
    #                            x = np.linspace(center[0]-search_rad,center[0]+search_rad,grid_size)
    #                            y = np.linspace(center[1]-search_rad,center[1]+search_rad,grid_size)
    #                            shifts = []
    #                            for i in x:
    #                                for j in y:
    #                                    shift_box = torch.tensor([i,j, base["l"],base["w"],base["h"],base["direction"]])
    #                                    shifts.append(shift_box)
                                    
    #                            # convert shifted grid of boxes into 2D space
    #                            shifts = torch.stack(shifts)
    #                            boxes_space = hg_new.state_to_im(shifts,name = cam)
                                 
                                
                                
    #                            # compute error between old_box_im and each shifted box footprints
    #                            box_expanded = old_box_im.repeat(boxes_space.shape[0],1,1)  
    #                            error = ((boxes_space[:,:4,:] - box_expanded[:,:4,:])**2).mean(dim = 2).mean(dim = 1)
                                
    #                            # find min_error and assign to center
    #                            min_idx = torch.argmin(error)
    #                            center = x[min_idx//grid_size],y[min_idx%grid_size]
    #                            search_rad /= 5
                                
    #                        # save box
    #                        min_err = error[min_idx].item()
    #                        all_errors.append(min_err)
    #                        base["x"] = self.safe(center[0])
    #                        base["y"] = self.safe(center[1])
                           
    #                        new_data[f_idx][key] = base
                           
    #                        # di = "EB" if obj["direction"] == 1 else "WB"
    #                        # print("Camera {}, {} obj {}: Error {}".format(cam,di,obj_idx,min_err))
                           
        
        
    #     # overwrite self.data with new_data
    #     self.data = new_data
    #     # overwrite self.hg with hg_new
    #     self.hg = hg_new
                
    #     # reinterpolate rest of data
    #     for i in range(self.get_unused_id()):
    #         self.interpolate(i,verbose = False)
        
    #     self.plot_all_trajectories()

            
    # def replace_y(self,reverse  = False):
            
    #     # create new copy of data
    #     new_data = []
        
    #     # for each frame in data
    #     for f_idx,frame_data in enumerate(self.data):
            
    #         if f_idx > self.last_frame and len(self.data[f_idx]) == 0:
    #             break
            
    #         new_data.append({})
    #         if f_idx % 100 == 0:
    #             print("On frame {}.".format(f_idx))
            
    #         # for each camera in frame data
    #         for camera in self.cameras:
    #             cam = camera.name
                
    #             # for each box in camera 
    #             for obj_idx in range(self.get_unused_id()):
    #                key = "{}_{}".format(cam,obj_idx)
    #                if frame_data.get(key): 
    #                    obj = frame_data.get(key)
                       
    #                    # if box was manually drawn
                       
    #                    if "gen" not in obj.keys() or obj["gen"] == "Manual":
                           
    #                        base = copy.deepcopy(obj)
    #                        new_box = self.offset_box_y(base,reverse = reverse)
                           
    #                        new_data[f_idx][key] = new_box
                           
    #     # overwrite self.data with new_data
    #     self.data = new_data
        
    #     # reinterpolate rest of data
    #     for i in range(self.get_unused_id()):
    #         self.interpolate(i,verbose = False) 
            

        
    #     self.plot_all_trajectories()

            
    # def offset_box_y(self,box,reverse = False):
        
    #     camera = box["camera"]
    #     direction = box["direction"]
        
    #     x = box["x"]
        
        
    #     direct =  "_EB" if direction == 1 else"_WB"
    #     key = camera + direct
        
    #     p2,p1,p0 = self.poly_params[key]
        
    #     y_offset = x**2*p2 + x*p1 + p0
        
    #     # if on the WB side, we need to account for the non-zero location of the leftmost line so we don't shift all the way back to near 0
    #     if direction == -1:
    #         y_straight_offset = self.hg.hg2.correspondence[camera]["space_pts"][0][1]
    #         y_offset -= y_straight_offset
            
    #     if not reverse:
    #         box["y"] -= y_offset
    #     else:
    #         box["y"] += y_offset

        
    #     return box
            
        
    # def fit_curvature(self,box,min_pts = 4):   
    #     """
    #     Stores clicked points in array for each camera. If >= min_pts points have been clicked, after each subsequent clicked point the curvature lines are refit
    #     """
        
    #     point = self.box_to_state(box)[0]
    #     direction = "_EB" if point[1] < 60 else "_WB"

    #     # store point in curvature_points[cam_idx]
    #     self.curve_points[self.clicked_camera+direction].append(point)
        
    #     # if there are sufficient fitting points, recompute poly_params for active camera
    #     if len(self.curve_points[self.clicked_camera+direction]) >= min_pts:
    #         x_curve = np.array([self.safe(p[0]) for p in self.curve_points[self.clicked_camera+direction]])
    #         y_curve = np.array([self.safe(p[1]) for p in self.curve_points[self.clicked_camera+direction]])
    #         pparams = np.polyfit(x_curve,y_curve,2)
    #         print("Fit {} poly params for camera {}".format(self.clicked_camera,direction))
    #         self.poly_params[self.clicked_camera+direction] = pparams
            
    #         self.plot()

    
    # def erase_curvature(self,box):
    #     """
    #     Removes all clicked curvature points for selected camera
    #     """
    #     point = self.box_to_state(box)[0]
        
    #     direction = "_EB" if point[1] < 60 else "_WB"
    #     # erase all points
    #     self.curve_points[self.clicked_camera+direction] = []

    #     # reset curvature polynomial coefficients to 0
    #     self.poly_params[self.clicked_camera+direction] = [0,0,0]
        
    #     self.plot()
    
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
                
                for obj_idx in range(self.next_object_id):
                    
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
            
            for obj_idx in range(self.next_object_id):
                
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
                self.label_buffer = copy.deepcopy(self.data),copy.deepcopy(self.objects)
                
                # Add and delete objects
                if self.active_command == "DELETE":
                    obj_idx = self.find_box(self.new) 
                    self.delete(obj_idx)
                    
                elif self.active_command == "ADD":
                    # get obj_idx
                    obj_idx = self.next_object_id
                    self.next_object_id += 1
                    
                    
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
                    
                # # interpolate between copy-pasted frames
                # elif self.active_command == "INTERPOLATE":
                #     obj_idx = self.find_box(self.new)
                #     self.interpolate(obj_idx)  

                # correct vehicle class
                elif self.active_command == "VEHICLE CLASS":
                    obj_idx = self.find_box(self.new)
                    try:
                        cls = (self.keyboard_input())  
                    except:
                        cls = "midsize"
                    self.change_class(obj_idx,cls)
                    
                elif self.active_command == "ASSOCIATE":
                    obj_idx = self.find_box(self.new)
                    pair = self.keyboard_input()
                    self.associate(obj_idx,pair)
                
                # adjust homography
                elif self.active_command == "HOMOGRAPHY":
                    self.correct_homography_Z(self.new)
                
                elif self.active_command == "2D PASTE":
                    self.paste_in_2D_bbox(self.new)
                    
                    
                self.plot()
                self.new = None   
                
                
           
           ### Show frame
                
           #self.cur_frame = cv2.resize(self.cur_frame,(1920,1080))
           cv2.imshow("window", self.plot_frame)
           title = "{} {}     Frame {}, Cameras {} and {}".format("R" if self.right_click else "",self.active_command,self.frame_idx,self.camera_names[self.active_cam],self.camera_names[self.active_cam + 1])
           cv2.setWindowTitle("window",str(title))
           
           
           ### Handle keystrokes 
           key = cv2.waitKey(1)

           
           if key == ord('9'):
                self.next()
                self.plot()
           elif key == ord("b"):
               self.hop()
               self.plot()
                
           elif key == ord('8'):
                self.prev()  
                self.plot()
           elif key == ord("q"):
               self.quit()
           elif key == ord("w"):
               self.save()
               #self.plot_all_trajectories()
           elif key == ord("@"):
               self.toggle_auto = not(self.toggle_auto)
               print("Automatic box pasting: {}".format(self.toggle_auto))
               
           elif key == ord("["):
               self.toggle_cams(-1)
           elif key == ord("]"):
               self.toggle_cams(1)
               
           elif key == ord("u"):
               self.undo()
           # elif key == ord("-"):
           #      [self.prev() for i in range(self.stride)]
           #      self.plot()
           # elif key == ord("="):
           #      [self.next() for i in range(self.stride)]
           #      self.plot()
           elif key == ord("+"):
               print("Filling buffer. Type number of frames to buffer...")
               n = int(self.keyboard_input())  
               self.buffer(n)
               
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

           elif key == ord("."):
                self.sink_active_object()
                self.plot()
                
           elif key == ord(" "):
               self.smart_advance()
               self.plot()
               
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
           elif key == ord("*"):
                self.active_command = "ASSOCIATE"
               
          
           elif self.active_command == "COPY PASTE" and self.copied_box:
               nudge = 0.25
               xsign = torch.sign(self.copied_box[1][1])
               if key == ord("1"):
                   self.shift(self.copied_box[0],None,dx = -nudge*xsign)
                   self.plot()
               if key == ord("5"):
                   self.shift(self.copied_box[0],None,dy =  nudge)
                   self.plot()
               if key == ord("3"):
                   self.shift(self.copied_box[0],None,dx =  nudge*xsign)
                   self.plot()
               if key == ord("2"):
                   self.shift(self.copied_box[0],None,dy = -nudge)
                   self.plot()
            
           elif self.active_command == "DIMENSION" and self.copied_box:
               nudge = 1/6 

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

if __name__ == "__main__":
    directory = "/home/derek/Data/1hz"
    hg_file = "/home/derek/Documents/i24/fast-trajectory-annotator/data/CIRCLES_20_Wednesday_20230530.cpkl"
    save_file = "temp_test.cpkl"
    ann = Annotator(directory,hg_file,save_file=save_file)  
    ann.run()
    