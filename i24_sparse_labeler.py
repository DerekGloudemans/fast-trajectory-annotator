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
        self.active_direction = -1

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
            
            # For WB
            if self.active_direction == -1:
                if (((c == 4) and p%1 == 0  and p< 41 and p > 2  and p != 24) or (c == 3 and p == 24)) and ".h264" in camera:
                #if (((c == 3 or c==5) and p%1 == 0  and p< 41 and p > 34 ))and ".h264" in camera:

                    include_cameras.append(shortname)

            if self.active_direction == 1:
                if (((c == 3) and p%1 == 0  and p< 41 and p > 7  and p != 24) or (c == 4 and p == 24)) and ".h264" in camera:
                    include_cameras.append(shortname)
           
        self.camera_names = include_cameras
        # 1. Get multi-thread frame-loader object
        self.b = NVC_Buffer(im_directory,include_cameras,ctx,buffer_lim = 650)
        
        
        # frame indexed data array since we'll plot by frames
        self.data = [{} for i in range(5000)] # each frame 
        
        
        self.frame_idx = 0
        self.toggle_auto = True
        self.AUTO = True

        # dictionary with dimensions, source camera, number of annotations, current lane, and sink camera for each object
        self.objects = {} 

        self.buffer(1)
        
        
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
        
       
        self.next_object_id = self.get_unused_id()

                
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
        #self.gps_data_cache = "./data/GPS_corrected.cpkl"
        self.gps_data_cache = "./data/GPS_10hz_smooth_CORRECTED.cpkl"

        try:
            with open(self.gps_data_cache,"rb") as f:
                self.gps = pickle.load(f)
        except:
            self.load_gps_data2()
            with open(self.gps_data_cache,"wb") as f:
                pickle.dump(self.gps,f)
    
        self.find_furthest_gps(direction = -1)
        print("Loaded annotator")
        
    def get_unused_id(self):
       i = 0
       while i in self.objects.keys():
           i += 1
       return i
   
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
    
    def write_timestamps(self):
        for fidx in range(3600):
            
            
            if fidx % 50 == 0:
                print("Writing timestamps for frame {}".format(fidx))
            
            try:
                self.b.ts[fidx][1]
            except IndexError:
                print("Buffering more timestamps...")
                self.buffer(1)
                
            
            for id in self.data[fidx]:
                x_pos= self.data[fidx][id][0]
                y_pos = self.data[fidx][id][1]
                direction = torch.sign(y_pos)
                
                if direction == self.active_direction:
                    ds = "_WB" if direction == -1 else "_EB"
                    mindist = np.inf
                    mints = None
                    # find closest camera
                    
                    
                    for cidx in range(len(self.camera_names)):
                        cname = self.camera_names[cidx] + ds
                        dist =  np.abs((self.extents[cname][0] + self.extents[cname][1])/2.0 - x_pos)
                        if dist < mindist:
                            mindist = dist
                            mints = self.b.ts[fidx][cidx]
                            
                    self.data[fidx][id][2] = mints
                    
    def save_gps(self):
        #self.gps_data_cache = "./data/GPS_10hz_working.cpkl"
        with open(self.gps_data_cache,"wb") as f:
            pickle.dump(self.gps,f)            
   
        print("Saved GPS data")
    
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
        gps_data_file = "/home/derek/Data/CIRCLES_GPS/gps_message_raw2a.csv"

        feet_per_meter = 3.28084
        y_valid = [-100,100]
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
                print("Error converting to float, skipping datum...")
                pass
            
            #if i > 100000: break
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
                    if len(cur_veh_data[0]) > 300:
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
                           
                    
        # Remove bad data where GPS misses a point and then tries to correct
        if True:
            min_gap = 0.08
            max_gap = 0.25
            
            for vid in trunc_vehicles:
                veh = trunc_vehicles[vid]
                init_size = len(veh["ts"])
                removals = []
                for t in range(1,len(veh["ts"])-1):
                    if veh["ts"][t+1] - veh["ts"][t] < min_gap:
                        removals.append(t-1)
                        removals.append(t)
                    if veh["ts"][t] - veh["ts"][t-1] > max_gap:
                        removals.append(t+1)
                        removals.append(t)
                #print(removals)        
                removals = list(set(removals))
                mask = np.ones(len(veh["ts"])).astype(bool)
                mask[removals] = False
                #print(mask)
                #print(veh["x"][mask])
                #print(veh["ts"][mask].shape,mask.shape)

                veh["ts"] = veh["ts"][mask]
                veh["x"] = veh["x"][mask]
                veh["y"] = veh["y"][mask]
                
                print("Obj {}: {} -> {} entries \n\n".format(vid,init_size,len(veh["ts"])))
        self.gps = trunc_vehicles    

    def rebase_annotations(self):
        """ 
        When I ported from 1Hz to 10Hz data, the post ids (eg. 56_4) dont match up (e.g. might by 56_3 now)
        This function iterates over all manual labels and for each annotation X_Y,
        searches all GPS X_Z and finds the Z that matches the annotation most closely, then rebases
        the annotation to X_Z.
        """
        obj_mapping = {}
        import time
        
        start = time.time()
        counter = 0
        for key in self.objects:
            
            # timekeeping
            elapsed = time.time() - start
            done_ratio =  counter / len(self.objects) + 1e-06
            eta = (elapsed / done_ratio) - elapsed
            
            print("\rRebasing annotations... ETA {:.1f} sec".format(eta),flush = True, end="\r")

            obj = self.objects[key]
            obj_map_key = obj["gps_id"]
            
            # find a position for that object
            for frame_data in self.data:
                if key in frame_data.keys():
                    x,t = frame_data[key][0],frame_data[key][2] # double check this indexing
                    break
                    
            # find the closest GPS position at a nearby time for the same object
            mindist = np.inf
            min_pass_num = None
            for gps_key in self.gps:
                if gps_key.split("_")[0] == obj_map_key.split("_")[0]:
                    
                    gps_xpos = None
                    # advance time to find a close position
                    for g_idx in range(len(self.gps[gps_key]["ts"])):
                        if np.abs(self.gps[gps_key]["ts"][g_idx] - t) < 1:
                            gps_xpos = self.gps[gps_key]["x"][g_idx]
                            break
                        
                    

                    # this run is not present at the same time
                    if gps_xpos is None:
                        continue
                    
                    dist = np.abs(gps_xpos - x)
                    if dist < mindist:
                        mindist =  dist
                        min_pass_num = gps_key
                
                    
                    
            # record the run / pass number
            if mindist > 100:
                print("Possible bad match for object {}/{}".format(key,min_pass_num))
            self.objects[key]["gps_id"] = min_pass_num
            counter += 1
        print("Done")
        
        

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
   
    
    def find_object(self,obj_id,direction = -1):
        direction = self.active_direction
        last_frame_idx = 0
        last_pos = 0
        first_frame_idx = None
        for i in range(len(self.b.frames)):
            if obj_id in self.data[i].keys():
                last_frame_idx = i
                if first_frame_idx is None:
                    first_frame_idx = i
                last_pos = self.data[i][obj_id][0]
                
        if last_frame_idx < (len(self.b.frames) - self.b.buffer_limit): # in this case the object was last visible in a frame before the buffer starts
            print("Object {} not visible in currently buffered frames".format(obj_id))
            return False
                
        # if len(self.b.frames) - last_frame_idx <= 20: # in this case the object has probably been labeled as far as it can be
        #     return
                
       
            
    
        #4. after selecting object, find relevant camera for current position
        directionstr = "EB" if direction == 1 else "WB"
        last_pos *= direction # make positive again
        for cidx in range(len(self.camera_names)):
            cam = self.camera_names[int(-1*direction*cidx)] # need to index in reverse order for EB 
            
            if direction == 1: 
                min_x = self.extents["{}_{}".format(cam,directionstr)][1]
                if last_pos < min_x: break
            else:
                min_x = self.extents["{}_{}".format(cam,directionstr)][0]
                if last_pos > min_x: break
        
        #5. advance
        self.active_cam = min(cidx,len(self.camera_names)-2)
        self.frame_idx = last_frame_idx
        gid = self.objects[obj_id]["gps_id"]
        print("Found {} (gps {}),  in camera {}, frame {} (first frame is {})".format(obj_id,gid,self.camera_names[self.active_cam],last_frame_idx,first_frame_idx))
        self.plot()
        return True
        
    def find_unfinished_obj(self,direction = -1,remove_last = 80):
        """
        Goes through object dict, finds first object without a sink camera, and finds last camera and frame for said object
        """
        direction = self.active_direction
        
        obj_id = None
        for key in self.objects.keys():
            if self.objects[key]["sink"] is None:
                obj_id = key
            
                if obj_id is not None:
                    last_frame_idx = 0
                    last_pos = 0
                    first_frame_idx = None
                    for i in range(len(self.b.frames)):
                        if obj_id in self.data[i].keys():
                            last_frame_idx = i
                            if first_frame_idx is None:
                                first_frame_idx = i
                            last_pos = self.data[i][obj_id][0]
                            
                    if last_frame_idx < (len(self.b.frames) - self.b.buffer_limit): # in this case the object was last visible in a frame before the buffer starts
                        print("Object {} not visible in currently buffered frames".format(obj_id))
                        continue                            
                    elif last_frame_idx > len(self.b.frames) - remove_last:
                        print("Object {} already labeled to frame {}, skipping...".format(obj_id,last_frame_idx))
                        continue # remove objects that have been labeled all the way to the end already
                            
                   
                        
                
                    directionstr = "EB" if direction == 1 else "WB"
                    last_pos *= direction # make positive again
                    for cidx in range(len(self.camera_names)):
                        if direction == -1:
                            cam = self.camera_names[cidx] # need to index in reverse order for EB 
                        else:
                            cam = self.camera_names[(len(self.camera_names) - 1 - cidx)]
                        
                        if direction == 1: 
                            min_x = self.extents["{}_{}".format(cam,directionstr)][1]
                            if last_pos < min_x: break
                        else:
                            min_x = self.extents["{}_{}".format(cam,directionstr)][0]
                            if last_pos > min_x: break
                    
                    #5. advance
                    if direction == 1: cidx = (len(self.camera_names) - 1 - cidx)
                    
                    #5. advance
                    self.active_cam = min(cidx,len(self.camera_names)-2)
                    self.frame_idx = last_frame_idx
                    gid = self.objects[obj_id]["gps_id"]
                    print("Found {} (gps {}),  in camera {}, frame {} (first frame is {})".format(obj_id,gid,self.camera_names[self.active_cam],last_frame_idx,first_frame_idx))
                    self.plot()
                    break
       

    def return_to_first_frame(self):
        #1. return to first frame in buffer
        for i in range(0,len(self.b.frames)):
            if len(self.b.frames[i]) > 0:
                break
            
        self.frame_idx = i
        self.label_buffer = copy.deepcopy(self.data),copy.deepcopy(self.objects)
        self.AUTO = True

        
        
    def find_furthest_gps(self,direction = -1,SINK = False):
        """
        Finds the GPS vehicle that is furthest along on the roadway (depending on direction of travel)
        Then finds the cameras bookending that object's position.
        Excludes all objects associated with a labeled object that has a sink camera
        """
        
        #1. return to first frame in buffer
        # stride = min(self.frame_idx, self.b.buffer_limit)
        # self.prev(stride = stride)
        direction = self.active_direction
        self.return_to_first_frame()
        
        
        while True: # iteratively step forward until there's an object
            
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
                        if not SINK: 
                            valid = False
                            break
                        else:
                            if self.objects[obj]["sink"] is not None:
                                valid = False
                                break
                if valid:
                    if gps_pos[gidx]*direction > furthest_pos:
                        furthest_pos = gps_pos[gidx]*direction
                        furthest_id = gpsid
            
            if furthest_id is None:
                self.next(stride = 10)
                if self.frame_idx >= len(self.b.frames) - 10:
                    print("All gps vehicles visible in buffered frames have been labeled at least once")
                    break
            else:
                break
        
        #4. after selecting object, find relevant camera for current position
        directionstr = "EB" if direction == 1 else "WB"
        furthest_pos *= direction # make positive again
        for cidx in range(len(self.camera_names)):
            if direction == -1:
                cam = self.camera_names[cidx] # need to index in reverse order for EB 
            else:
                cam = self.camera_names[(len(self.camera_names) - 1 - cidx)]
            
            if direction == 1: 
                min_x = self.extents["{}_{}".format(cam,directionstr)][1]
                if furthest_pos < min_x: break
            else:
                min_x = self.extents["{}_{}".format(cam,directionstr)][0]
                if furthest_pos > min_x: break
        
        #5. advance
        if direction == 1: cidx = (len(self.camera_names) - 1 - cidx)
        self.active_cam = min(cidx,len(self.camera_names)-2)
        print("Next furthest gps vehicle is {}, which will be visible in {} or {}".format(furthest_id,self.camera_names[self.active_cam],self.camera_names[self.active_cam+1]))
        self.smart_advance(gps_id = furthest_id)
        self.plot()
        
    def associate(self,id,gps_id):
        try:
            self.objects[id]["gps_id"] = gps_id
            self.active_command = "COPY PASTE"
        except:
            pass
        
        
    def smart_advance(self,gps_id = None):
        """ Advance to next camera if there is an annotation for this camera/frame pair, else just advance a frame"""
        
        if gps_id is None:
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
            if self.active_cam < len(self.camera_names) -1 and self.active_cam > 0:
                self.toggle_cams(-self.active_direction)
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
        
        else: # use dummy direction
            cam = self.camera_names[self.active_cam]
            direction = self.active_direction
            directionstr = "EB" if direction == 1 else "WB"
            
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
            "gps_id":None,
            "direction":self.active_direction
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
        
        self.recount_objects(f1 = len(self.b.frames))
            
        # advance to next object
        self.find_furthest_gps()
        self.save()
     
    def recount_objects(self,f0 = 0,f1 = 1200,tolerance = 40):
        for obj_id in self.objects.keys():
            if self.objects[obj_id]["direction"] != self.active_direction:
                continue

            source_idx = -1
            sink_idx = -1
            source = self.objects[obj_id]["source"]
            sink = self.objects[obj_id]["sink"]
            if sink is not None:
        
                all_x_pos = []
                last_frame_idx = -1
                for fidx in range(len(self.data)):
                    if obj_id in self.data[fidx].keys():
                        all_x_pos.append(self.data[fidx][obj_id][0])
                        last_frame_idx = fidx
                    
                # get number of cameras between source and sink
                for i in range(len(self.camera_names)):
                    if self.camera_names[i] == source:
                        source_idx = i
                    elif self.camera_names[i] == sink:
                        sink_idx = i
                
                missing_cameras = []      
                for cidx in range(source_idx,sink_idx+1):
                    cam = self.camera_names[cidx]
                    ANN = False
                    dirstr = "EB" if self.active_direction == 1 else "WB"
                    ex = self.extents["{}_{}".format(cam,dirstr)][0:2]
                    
                    for pos in all_x_pos:
                        if pos > (ex[0]-20) and pos < (ex[1] + 20):
                            ANN = True
                            break
                    
                    if not ANN:
                        missing_cameras.append(cam)
                
                if len(missing_cameras) > 0:
                    print("BAD : Object {} ({}): Has sink but is missing cameras: {}".format(obj_id,self.objects[obj_id]["gps_id"],missing_cameras))
                else:
                    print("GOOD: Object {} ({}): Has sink and no missing cameras".format(obj_id,self.objects[obj_id]["gps_id"]))
                
            elif sink is None:
                all_x_pos = []
                last_frame_idx = -1
                for fidx in range(len(self.data)):
                    if obj_id in self.data[fidx].keys():
                        all_x_pos.append(self.data[fidx][obj_id][0])
                        last_frame_idx = fidx
                
                if f1 - last_frame_idx >= 20 :
                    print("OKAY: Object {} ({}): Last annotation is near buffer limit (frame {}), so probably still active".format(obj_id,self.objects[obj_id]["gps_id"],last_frame_idx))
                    
                else:
                    print("BAD : Object {} ({}): Missing annotations, last annotation on frame {}".format(obj_id,self.objects[obj_id]["gps_id"],last_frame_idx))
                
                
                
        """
        Rewrite for more info. Possible things that can happen
        - Object has source and sink and there is at least one box visible in each camera between, inclusive (object is done)
        - Same as above but object is missing at least one annotation
        - Object does not have sink, but has been labeled in all consecutive cameras and was labeled within tolerance of end of range (i.e. is still active)
        - Object does not have sink and has not been labeled up to within tolerance of end of range.
        """
                
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
        
        if obj_idx in self.objects.keys() and n_frames == -1:
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
    
    
    def plot_traj(self, highlight_id = None):
        
        """ Plot timespace diagram for all GPS tracks. Then add annotation points"""
        
        durations = []
        for gid in self.gps:
            duration = (max(self.gps[gid]["ts"]) - min(self.gps[gid]["ts"]))/60
            durations.append(duration)
            
        #print(durations)
        print("Duration stats")
        print(sum(durations)/len(durations))
        print(min(durations),max(durations))
        
        lengths = []
        for gid in self.gps:
            length = (max(self.gps[gid]["x"]) - min(self.gps[gid]["x"]))
            lengths.append(length)
            
        #print(lengths)
        print("Length stats")
        print(sum(lengths)/len(lengths))
        print(min(lengths),max(lengths))
        
        plt.figure(figsize = (30,10))
        colors = np.random.rand(200,3)
        
        for gid in self.gps:
            if self.gps[gid]["y"][0] * self.active_direction > 0:
                plt.plot(self.gps[gid]["ts"],self.gps[gid]["x"],color = colors[int(gid.split("_")[0])])#,marker = "x")
            
        if highlight_id is not None:
            gid = highlight_id
            if self.gps[gid]["y"][0] * self.active_direction > 0:
                plt.plot(self.gps[gid]["ts"],self.gps[gid]["x"],color = colors[int(gid.split("_")[0])], linewidth = 5)
                
        for frame_data in self.data:
            for did in frame_data:
                datum = frame_data[did]
                gid = int(self.objects[did]["gps_id"].split("_")[0])
                
                if datum[1] *self.active_direction > 0:
                    plt.scatter(datum[2],datum[0],color = colors[gid])
            
        plt.show()
        
        plt.figure(figsize = (30,10))
        colors = np.random.rand(200,3)
        
        for gid in self.gps:
            plt.plot(self.gps[gid]["x"],self.gps[gid]["y"],color = colors[int(gid.split("_")[0])])
            
        for frame_data in self.data:
            for did in frame_data:
                datum = frame_data[did]
                gid = int(self.objects[did]["gps_id"].split("_")[0])
                

                plt.scatter(datum[0],datum[1],color = colors[gid])
            
        plt.show()


    def examine_error(self):
        
        bad = []
        good = []
        cutoff = 3
        
        for id in self.objects:
           x_err = []
           y_err = []
           gps_id = self.objects[id]["gps_id"]
            
           # for each annotation
           for fidx in range(3600):
               if id in self.data[fidx].keys():
                    obj_pos = self.data[fidx][id]
                    frame_ts = obj_pos[2]
                   
                    gpsob = self.gps[gps_id]
                   
            
        
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
                
                    x_error = x_interp - obj_pos[0]
                    y_error = y_interp - obj_pos[1]
                    x_err.append(x_error)
                    y_err.append(y_error)
                    
           # ravel error statistics
           x_err = np.array(x_err)
           y_err = np.array(y_err)
            
           xmean = np.mean(x_err)
           xstd = np.std(x_err)
           ymean = np.mean(y_err)
           ystd = np.std(y_err)
           
           if xstd > 1 or xmean > 1 or ymean > 1 or ystd > 1:
               print("For object {}/{}, x error: {:.2f}ft ({:.2f}ft std) ---- y error {:.2f}ft ({:.2f}ft std)".format(id,gps_id,xmean,xstd,ymean,ystd))
           if xstd > cutoff:
               bad.append(xstd)
           else:
               good.append(xstd)
               
        print("{:.3f}% under {} ft standard deviation x error".format(len(good)/(len(bad)+len(good)), cutoff))
    
    def mean_shift_x(self):
       for id in self.objects:
          x_err = []
          y_err = []
          gps_id = self.objects[id]["gps_id"]
           
          # for each annotation
          for fidx in range(3600):
              if id in self.data[fidx].keys():
                   obj_pos = self.data[fidx][id]
                   frame_ts = obj_pos[2]
                  
                   gpsob = self.gps[gps_id]
                  
           
       
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
               
                   x_error = x_interp - obj_pos[0]
                   y_error = y_interp - obj_pos[1]
                   x_err.append(x_error)
                   y_err.append(y_error)
                   
          # ravel error statistics
          x_err = np.array(x_err)
          y_err = np.array(y_err)
           
          xmean = np.mean(x_err)
          xstd = np.std(x_err)
          ymean = np.mean(y_err)
          ystd = np.std(y_err)
           
       
          if True: 
              for i in range(0,len(gpsob["x"])):
                  gpsob["x"][i] -= xmean
                  
    def mean_shift_ts(self):
        """ 
        For each trajectory, find the time-shift (one per tracklet) that minimizes the error between 
        """
        
        
    
        for id in self.objects:
            
           
           gps_id = self.objects[id]["gps_id"]
           gpsob = self.gps[gps_id]
 
           best_x_err = np.inf
           best_list = None
           best_shift = 0
           
           for timeshift in  np.arange(-2,2,0.05):
               gpsob["ts"] += timeshift
               x_err = []
               y_err = []
                
               # for each annotation
               for fidx in range(3600):
                   if id in self.data[fidx].keys():
                        obj_pos = self.data[fidx][id]
                        frame_ts = obj_pos[2]
                       
                        
                
            
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
                    
                        x_error = x_interp - obj_pos[0]
                        y_error = y_interp - obj_pos[1]
                        x_err.append(x_error)
                        y_err.append(y_error)
                        
               gpsob["ts"] -= timeshift
               
               std = np.std((np.array(x_err)))
               #print(timeshift,std)
               if std < best_x_err:
                   best_list = x_err
                   best_x_err = std
                   best_shift = timeshift
               
           # ravel error statistics
           x_err = np.array(best_list)
           #y_err = np.array(y_err)
            
           xmean = np.mean(x_err)
           xstd = np.std(x_err)
           #ymean = np.mean(y_err)
           #ystd = np.std(y_err)
            
           print("For object {}/{}, best timeshift {}s, {:.2f}/{:.2f} x err".format(id,gps_id,best_shift,xmean,xstd))
           gpsob["ts"] += best_shift
                   
    
    
    def rolling_shift_x(self):
        """ 
        Between labeled points n and n-1, linearly interpolate the x-offset between the x
        offsets at n and n-1. At endpoints, use simply the offset for n or n-1
        """
        
        print("\n\n applying rolling shift x correction")
        # 1 create list of offsets for each labeled point 
        
        for id in self.objects:
            offsets  = []
            times = []
            gps_id = self.objects[id]["gps_id"]

            # for each annotation
            for fidx in range(3600):
                if id in self.data[fidx].keys():
                     obj_pos = self.data[fidx][id]
                     frame_ts = obj_pos[2]
                    
                     gpsob = self.gps[gps_id]
                    
             
         
                     # iterate through timestamps to find ts directly before and after current ts
                     for t in range(1,len(gpsob["ts"])):
                         if gpsob["ts"][t] > frame_ts:
                             break
                     
                     x1 = gpsob["x"][t-1]
                     x2 = gpsob["x"][t]
                     #y1 = gpsob["y"][t-1]
                     #y2 = gpsob["y"][t]
                     t1 = gpsob["ts"][t-1]
                     t2 = gpsob["ts"][t]
                     f1 = (t2-frame_ts)/(t2-t1)
                     f2 = (frame_ts-t1)/(t2-t1)
                     
                     x_interp =  x1*f1 + x2*f2
                     #y_interp =  y1*f1 + y2*f2
                 
                     x_error = x_interp - obj_pos[0]
                     #y_error = y_interp - obj_pos[1]
                     offsets.append(x_error)
                     times.append(frame_ts)
                     #y_err.append(y_error)
                     
            offsets = np.array(offsets)
            times = np.array(times)
            
        
            # now interpolate the other direction
            gps_offsets = []
            gpsob = self.gps[gps_id]
            for i in range(len(gpsob["x"])):
                ts = gpsob["ts"][i]
                x = gpsob["x"][i]

                if ts < times[0]:
                    gps_offsets.append(offsets[0])
                    gpsob["x"][i] -= offsets[0]
                elif ts > times[-1]:
                    gps_offsets.append(offsets[-1])
                    gpsob["x"][i] -= offsets[-1]
                else: # interpolate
                
                    tidx = 0
                    while times[tidx] < ts:
                        tidx += 1
                    
                    t1 = times[tidx-1]
                    t2 = times[tidx]
                    x1 = offsets[tidx-1]
                    x2 = offsets[tidx]
                    f1 = (t2-ts)/(t2-t1)
                    f2 = (ts-t1)/(t2-t1)
                    
                    gps_off =  x1*f1 + x2*f2
                    gps_offsets.append(gps_off)
                    gpsob["x"][i] -= gps_off
                    
    
    def interpolate_y(self):
        print("\n\n applying rolling shift y correction")
        # 1 create list of offsets for each labeled point 
        
        for id in self.objects:
            positions  = []
            times = []
            # gps_id = self.objects[id]["gps_id"]

            # # for each annotation,ravel all annotations
            for fidx in range(3600):
                if id in self.data[fidx].keys():
                     obj_pos = self.data[fidx][id]
                     frame_ts = obj_pos[2]
                     y_pos = obj_pos[1]
                     positions.append(y_pos)
                     times.append(frame_ts)
                
                     
            positions = np.array(positions)
            times = np.array(times)
            
        
            # # now interpolate the other direction
            # gps_offsets = []
            gps_id = self.objects[id]["gps_id"]
            gpsob = self.gps[gps_id]
            for i in range(len(gpsob["y"])):
                ts = gpsob["ts"][i]
                y = gpsob["y"][i]

                if ts < times[0]:
                    gpsob["y"][i] = positions[0]
                elif ts > times[-1]:
                    gpsob["y"][i] = positions[-1]
                else: # interpolate
                
                    tidx = 0
                    while times[tidx] < ts:
                        tidx += 1
                    
                    t1 = times[tidx-1]
                    t2 = times[tidx]
                    y1 = positions[tidx-1]
                    y2 = positions[tidx]
                    f1 = (t2-ts)/(t2-t1)
                    f2 = (ts-t1)/(t2-t1)
                    
                    gps_off =  y1*f1 + y2*f2
                    gpsob["y"][i] = gps_off

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
                    ki = self.keyboard_input()
                    if len(ki) == 0:
                        n_frames = -1
                    else:
                        n_frames = int(ki)
                    self.delete(obj_idx,n_frames)
                    
                elif self.active_command == "ADD":
                    # get obj_idx
                    
                    

                    self.add(self.next_object_id,self.new)
                    self.next_object_id = self.get_unused_id()
                    
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
               
           elif key == ord("f"):
               self.find_furthest_gps()
           elif key == ord("g"):
                self.find_unfinished_obj()
           elif key == ord("j"):
                self.return_to_first_frame()
                self.active_cam = 0
                self.plot()
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
               
           elif key == ord("s"):
                try:
                    obj_id = int(self.keyboard_input())
                    self.find_object(obj_id)
                except:
                    pass
                
                
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
                   self.shift(self.copied_box[0],None,dx = -nudge*xsign* -self.active_direction)
                   self.plot()
               if key == ord("5"):
                   self.shift(self.copied_box[0],None,dy =  nudge)
                   self.plot()
               if key == ord("3"):
                   self.shift(self.copied_box[0],None,dx =  nudge*xsign* -self.active_direction)
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
    #hg_file = "/home/derek/Documents/i24/fast-trajectory-annotator/data/CIRCLES_20_Wednesday_20230530.cpkl"
    hg_file = "/home/derek/Documents/i24/i24_track/data/homography/CIRCLES_20_Wednesday.cpkl"

    save_file = "labeled_data_sandbox.cpkl"
    ann = Annotator(directory,hg_file,save_file=save_file)  
    #ann.write_timestamps()
    #ann.plot_traj(highlight_id = "16_1")
    #ann.recount_objects(f1 = 1500)   
   
    
    
    # ann.mean_shift_x()
    # ann.mean_shift_ts()
    # ann.rolling_shift_x()
    # ann.interpolate_y()
    #ann.examine_error()
    

    # ann.save_gps()
    #ann.rebase_annotations()
    ann.plot_traj()
    # ann.frame_idx = 90
    #ann.run()