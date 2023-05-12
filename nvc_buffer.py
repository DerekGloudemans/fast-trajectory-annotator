import torch
import PyNvCodec as nvc
import PytorchNvCodec as pnvc



import os
import time
import re
import numpy as np
import torch

from torchvision.transforms import functional as F

    
class NVC_Buffer():
    """
    Loads multiple video files in parallel with PTS timestamp decoding and 
    directory - overall file buffer
    """
    
    def __init__(self,directory,include_cams,ctx,resize = (1920,1080)):
     
        
               
        self.cameras_per_device = np.zeros([torch.cuda.device_count()])
        self.loaders = {}
        self.camera_order = include_cams
       
        # instead of getting individual files, sequence is a directorie (1 per camera)
        cam_sequences = {}        
        for camera_dir in os.listdir(directory):
            cam_name = re.search("P\d\dC\d\d",camera_dir).group(0)
            if cam_name in include_cams:
            
                # get device
                for idx in range(len(self.cameras_per_device)):
                    min_cams = self.cameras_per_device.min()
                    if self.cameras_per_device[idx] == min_cams:
                        break
        
                # initialize loader (one per camera)            
                loader = GPUBackendFrameGetter(camera_dir,idx,ctx,resize = resize)
                self.loaders[cam_name].append(loader)
            
        self.frames = []
        self.timestamps = []
        

        
    def fill(self,n_frames):
        for i in range(n_frames):
            if i % 10 == 0:
                print("Buffering frame {} of {}".format((i,n_frames)))
                
            frames,ts = self.get_frames()
            
            self.frames.append(frames)
            self.ts.append(ts)
        
        
        
    def get_frames(self,target_time = None, tolerance = 1/60):
        # accumulators
        frames = []
        timestamps = []
        
        for cam in self.camera_order:
            
            frame,ts = next(self.loaders[cam])
            frames.append(frame)
            timestamps.append(ts)

                
        # stack each accumulator list
        out = []
        for lis in frames:
            if len(lis) == 0: # occurs when no frames are mapped to a GPU
                out.append(np.empty(0))
            else:
                out.append(np.stack(lis))
                
        return out,timestamps
    
    
    
class GPUBackendFrameGetter:
    def __init__(self,directory,device,ctx,buffer_size = 5,resize = (1920,1080),start_time = None):
        
        # create shared queue
        self.queue = ctx.Queue()
        self.frame_idx = -1
        self.device = device  
        
        self.directory = directory
        # instead of a single file, pass a directory, and a start time
        self.worker = ctx.Process(target=load_queue_continuous_vpf, args=(self.queue,directory,device,buffer_size,resize,start_time))
        self.worker.start()   
            

    def __len__(self):
        """
        Description
        -----------
        Returns number of frames in the track directory
        """
        
        return 1000000
    
    
    def __next__(self):
        """
        Description
        -----------
        Returns next frame and associated data unless at end of track, in which
        case returns -1 for frame num and None for frame

        Returns
        -------
        frame_num : int
            Frame index in track
        frame : tuple of (tensor,tensor,tensor)
            image, image dimensions and original image

        """
        
        
        frame = self.queue.get(timeout = 10)
        ts = frame[1] / 10e8
        im = frame[0].data.numpy()
        
        return im,ts
        
        # if False: #TODO - implement shutdown
        #     self.worker.terminate()
        #     self.worker.join()
        #     return None
        
def load_queue_continuous_vpf(q,directory,device,buffer_size,resize,start_time):
    
    
    resize = (resize[1],resize[0])
    gpuID = device
    device = torch.cuda.device("cuda:{}".format(gpuID))
    
    
    
    # GET FIRST FILE
    # sort directory files (by timestamp)
    files = os.listdir(directory)
    
    # filter out non-video_files and sort video files
    files = list(filter(  (lambda f: True if ".mkv" in f else False) ,   files))
    files.sort()
    
    # select next file that comes sequentially after last_file
    for fidx,file in enumerate(files):
        try:
            ftime = float(         file.split("_")[-1].split(".mkv")[0])
            nftime= float(files[fidx+1].split("_")[-1].split(".mkv")[0])
            if nftime >= start_time:
                break
        except:
            break # no next file so this file should be the one
    
    last_file = file
    while True:
        
        file = os.path.join(directory,file)
        
        # initialize Decoder object
        nvDec = nvc.PyNvDecoder(file, gpuID)
        target_h, target_w = nvDec.Height(), nvDec.Width()
    
        to_rgb = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpuID)
        to_planar = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, gpuID)
    
        cspace, crange = nvDec.ColorSpace(), nvDec.ColorRange()
        if nvc.ColorSpace.UNSPEC == cspace:
            cspace = nvc.ColorSpace.BT_601
        if nvc.ColorRange.UDEF == crange:
            crange = nvc.ColorRange.MPEG
        cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)
        
        
        # get frames from one file
        while True:
            if q.qsize() < buffer_size:
                pkt = nvc.PacketData()
                
                # advance frames until reaching start_time
                # Double check this math, pkt.pts is in nanoseconds I believe
                if start_time is not None and start_time > pkt.pts:
                    continue
                    
                
                # Obtain NV12 decoded surface from decoder;
                raw_surface = nvDec.DecodeSingleSurface(pkt)
                if raw_surface.Empty():
                    break
    
                # Convert to RGB interleaved;
                rgb_byte = to_rgb.Execute(raw_surface, cc_ctx)
            
                # Convert to RGB planar because that's what to_tensor + normalize are doing;
                rgb_planar = to_planar.Execute(rgb_byte, cc_ctx)
            
                # likewise, end of video file
                if rgb_planar.Empty():
                    break
                
                # Create torch tensor from it and reshape because
                # pnvc.makefromDevicePtrUint8 creates just a chunk of CUDA memory
                # and then copies data from plane pointer to allocated chunk;
                surfPlane = rgb_planar.PlanePtr()
                surface_tensor = pnvc.makefromDevicePtrUint8(surfPlane.GpuMem(), surfPlane.Width(), surfPlane.Height(), surfPlane.Pitch(), surfPlane.ElemSize())
                surface_tensor.resize_(3, target_h,target_w)
                
                try:
                    surface_tensor = torch.nn.functional.interpolate(surface_tensor.unsqueeze(0),resize).squeeze(0)
                except:
                    raise Exception("Surface tensor shape:{} --- resize shape: {}".format(surface_tensor.shape,resize))
            
                # This is optional and depends on what you NN expects to take as input
                # Normalize to range desired by NN. Originally it's 
                surface_tensor = surface_tensor.type(dtype=torch.cuda.FloatTensor)/255.0
                
                
                # apply normalization
                surface_tensor = F.normalize(surface_tensor,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                frame = (surface_tensor,pkt.pts)
                q.put(frame)
            
        ### Get next file if there is one 
        # sort directory files (by timestamp)
        files = os.listdir(directory)
        
        # filter out non-video_files and sort video files
        files = list(filter(  (lambda f: True if ".mkv" in f else False) ,   files))
        files.sort()
        
        # select next file that comes sequentially after last_file
        NEXTFILE = False
        for file in files:
            if file > last_file:
                last_file = file
                NEXTFILE = True           
                break

        
        if not NEXTFILE:
            raise Exception("Reached last file for directory {}".format(directory))
            