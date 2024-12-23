import os
import glob
from typing import Literal
#
from dev.ai.xBase.detect import XDetector
#
from dev.ai.xObject.callback import XObjectDetectorCallback
from dev.ai.xObject.capture import XObjectDetectorCapture
from dev.ai.xObject.annotator import XObjectAnnotator
#
from dev.schema.model import IDetect
#
from dev.util.util import getEncodeFrame
#
from ultralytics import YOLO
import numpy as np
#

class XObjectDetector(XDetector):
    def __init__(self,request:IDetect,override:dict):
        self.request = request
        self.override = override
        self.init_model()
        self.init_annotator()
        self.init_callback()
        self.init_source()
        
        
    def init_model(self):
        self.initObjectDetector()
        
    def init_callback(self):
        self.callback = XObjectDetectorCallback(self.request,self.annotator.labelMap)
        
    def init_source(self):
        self.capture = XObjectDetectorCapture(self.request.source_type,self.request.target_path,self.annotator.meta)
        
    def initObjectDetector(self):
        model_file = glob.glob(os.path.join(self.request.model_path,"**","**","*.pt"))[0]
        self.net = YOLO(model=model_file,task='detect')
    
    def init_annotator(self):
        self.annotator = XObjectAnnotator()
    
    def predict(self,im,stream):
        return self.net.predict(source=im,stream=stream,**self.override)
    
    def predict_image(self):
        self.callback.onPredictStart()
        event_threshold = 0.85
        #
        frame,im = next(self.capture)
        result = self.predict(im,stream=False)[0].cpu().numpy()
        speed = sum(result.speed.values()) # type: ignore
        boxes = result.boxes.data # type: ignore
        self.annotator.rescale_box(boxes)
        event_box = boxes[boxes[:,4] > event_threshold] # type: ignore
        if len(event_box):
            annotate_frame = self.visualize(frame,event_box)
            self.callback.onPredictEpochEnd(annotate_frame,speed,event_box,None)
            return annotate_frame
        return frame

    def predict_video(self):
        output = {'frame_list':[],'result_list':[],'perf_list':[]}
        event_threshold = 0.95
        for i,(frame,im) in enumerate(self.capture):
            frame_num = i+1
            result = next(self.predict(im,stream=True)).cpu().numpy() # type: ignore (Results)
            speed = sum(result.speed.values()) # preprocess + inference + postprocess
            output['perf_list'].append(speed)
            boxes = result.boxes.data
            self.annotator.rescale_box(boxes)
            event_box = boxes[boxes[:,4] > event_threshold]
            if len(event_box):
                annotate_frame = self.visualize(frame,event_box)
                topResult = self.callback.onPredictEpochEnd(annotate_frame,speed,event_box,frame_num)
                output['result_list'].append(topResult)
                output['frame_list'].append(self.annotator.bgr2rgb(annotate_frame))
            else:
                output['result_list'].append(np.empty((0,6)))
                output['frame_list'].append(self.annotator.bgr2rgb(frame))
        return output

    def predict_stream(self):
        event_threshold = 0.7
        for frame,im in self.capture:
            result = next(self.predict(im,stream=True)).cpu().numpy() # type: ignore
            boxes = result.boxes.data
            self.annotator.rescale_box(boxes)
            event_box = boxes[boxes[:,4] > event_threshold]
            yield self.visualize(frame,event_box)

    def visualize(self,image,result):
        self.annotator.init_image(image)
        self.annotator.begin()
        self.annotator.build(['BBox'])
        for data in result:
            self.annotator.drawBBox(data)
        self.annotator.end(['BBox'])
        return self.annotator.result()


    def getVideo(self):
        self.callback.onPredictStart()
        output = self.predict_video()
        self.callback.onPredictEnd(output['frame_list'],output['perf_list'],output['result_list'])

    def getStream(self):
        for detect in self.predict_stream():
            yield getEncodeFrame(detect)
    
    def getImage(self):
        detect = self.predict_image()
        frame = getEncodeFrame(detect)
        return frame

def createXObjectDetector(request:IDetect,mode,class_list):
    override = dict(device='0',project=request.model_path,verbose=False,classes=class_list)
    #
    detector = XObjectDetector(request,override=override)
    #
    return detector

