#
from dev.ai.xBase.detect import XDetector
#
import numpy as np
import torch
#
from .net.default_arg import DEFAULT_ARGS
from .net.slowfast.config.defaults import assert_and_infer_cfg
from .net.slowfast.utils.parser import load_config
from .net.slowfast.utils import checkpoint as cu
from .net.slowfast.models import build_model
from .net.slowfast.visualization.utils import process_cv2_inputs
#
from dev.ai.xFire.callback import XTestCallback
from dev.ai.xFire.capture import XFireDetectorCapture
from dev.ai.xFire.annotator import XFireAnnotator
from dev.ai.common.util import Sampler
from dev.util.util import getEncodeFrame
#
from queue import Queue
import time

class xFireDetectorTest(XDetector):
    def __init__(self, mode, model_path, path, config):
        super().__init__()
        self.mode = mode
        self.path = path
        self.cfg = self.init_config({'model_path': model_path, 'config': config})
        
        self.result = np.empty(0)
        self.init_model()
        self.init_annotator()
        self.init_callback()
        self.init_source()
        
    def init_config(self, kwargs):
        DEFAULT_ARGS.model_dir = kwargs['model_path']
        cfg = load_config(DEFAULT_ARGS, kwargs['config'])
        cfg = assert_and_infer_cfg(cfg)
        return cfg
        
    def init_model(self):
        self.initFireDetector()
    
    def init_callback(self):
        self.callback = XTestCallback(self.annotator.labelMap)
        
    def init_source(self):
        self.capture = XFireDetectorCapture(self.mode, self.path, self.annotator.meta)
        self.annotator.initCanvas()
        
    def init_annotator(self):
        self.annotator = XFireAnnotator()
        
    def initFireDetector(self):
        self.fire_detector = build_model(self.cfg)
        cu.load_test_checkpoint(self.cfg, self.fire_detector)
        self.fire_detector.eval()
        self.device = torch.cuda.current_device()
        self.sampler = Sampler(self.cfg)
        self.ts = []
        self.tp = {}
        self.q = Queue()
        self.display_frames = 30
        
    def runnable(self, sample, device):
       with torch.no_grad():
           inputs = process_cv2_inputs(sample, self.cfg)
           inputs = [input.cuda(device=device, non_blocking=True) for input in inputs] # 0.05
           preds = self.fire_detector(inputs, None) # 0.005
           torch.cuda.synchronize()
           preds = preds.detach().cpu()
           torch.cuda.synchronize()
           return torch.cat(torch.topk(preds, 3), dim=0).T.numpy()
    
    def predict_video(self):
        output = { 'frame_list': [], 'result_list': [], 'pref_list': []}
        device = torch.device(self.device)
        self.callback.onPredictStart()
        self.is_fire = False
        for frame_count, (frame, im) in enumerate(self.capture):
            print(f"{frame_count}/{self.capture.meta.frame_count}\r")
            if frame_count > 5400: break
            frame_count += 1
            ret, sample, _ = self.sampler.getSample(im) ## [N, H, W, C]
            if ret:
                self.is_fire = True
                self.result = self.runnable(sample, device)
            if self.is_fire:
                self.display_frames -= 1
            annotated_frame,onCallback = self.visualize(frame)
            output['frame_list'].append(self.annotator.bgr2rgb(annotated_frame))
        return output
                
    def predict_stream(self):
        event_threshold = 0.95
        self.is_fire = False
        device = torch.device(self.device)
        for frame_count,(frame,im) in enumerate(self.capture):
            frame_count += 1
            ret, sample, _ = self.sampler.getSample(im) ## [N,H,W,C]
            if ret:
                self.is_fire = True
                self.createThread(sample,device)
            if self.is_fire:
                if not self.q.empty():
                    self.result = self.q.get()
                self.display_frames -= 1
            annotated_frame,onCallback = self.visualize(frame)
            yield annotated_frame
    
    def visualize(self, image):
        self.annotator.init_image(image)
        self.annotator.begin()
        
        if self.display_frames > 0 and self.is_fire and len(self.result) and self.result[0, 1] != 0:
            self.annotator.drawAlert(self.result[self.result[:,1]>0])
            self.annotator.end(['Alert'])
            return self.annotator.result(), False
        if self.display_frames <= 0 and self.is_fire and len(self.result) and self.result[0, 1] != 0:
            self.is_fire = False
            self.display_frames = 30
            self.annotator.end(['Alert'])
            return self.annotator.result(),True
        self.annotator.end([])
        return self.annotator.result(),False
        
    def getVideo(self):
        output = self.predict_video()
        self.callback.onPredictEnd(output['frame_list'], output['perf_list'], output['result_list'])
        
    def getStream(self):
        for detect in self.predict_stream():
            yield getEncodeFrame(detect)
            time.sleep(1/30)
            
    def getImage(self):
        raise ValueError("No Supported Detect Type")
        
def createXFireDetectorTest(model_path, path, mode, config):
    detector = xFireDetectorTest(mode, model_path, path, config)
    return detector