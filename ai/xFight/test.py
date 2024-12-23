#
from dev.ai.xBase.detect import XDetector
#
import paddle
from .net.config import *
from .net.video_action_infer import *
from .net.video_action_preprocess import *
from .net.datacollector import *
from .net.utils import *
from .net.preprocess import ShortSizeScale
#
from dev.ai.xFight.callback import *
from dev.ai.xFight.capture import XFightDetectorCapture
from dev.ai.xFight.annotator import XFightAnnotator
#
from dev.util.util import getEncodeFrame

class XFightDetectModel(VideoActionRecognizer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.scale = ShortSizeScale(self.short_size)
    
    def postprocess(self, output):
        output = output.flatten()  # numpy.ndarray
        output = softmax(output)
        classes = np.argpartition(output, -self.top_k)[-self.top_k:]
        classes = classes[np.argsort(-output[classes])]
        scores = output[classes]
        return classes,scores
    
    def predict(self, input):
        classes,scores = super().predict(input)
        return np.append(classes,scores)
    

class XFightDetectorTest(XDetector):
    def __init__(self,mode,model_path,path,config):
        super().__init__()
        paddle.enable_static()
        self.mode = mode
        self.path = path
        self.cfg = self.init_config({'model_path':model_path,'config':config})
        
        self.boxes = np.empty(0)
        self.fight = np.array([0,1],dtype=np.float32)
        self.is_fight = False
        self.sample_list = []
        
        self.init_model()
        self.init_annotator()
        self.init_callback()
        self.init_source()
        
    def init_config(self,args):
        FLAGS = get_pipeline_cfg(args)
        return merge_cfg(FLAGS)
    
    def init_model(self):
        self.initHumanDetector()
        self.initFightDetector()
    
    def init_callback(self):
        self.callback = XTestCallback(self.annotator.labelMap)
    
    def init_source(self):
        self.capture = XFightDetectorCapture(self.mode,self.path,self.annotator.meta)
    
    def initHumanDetector(self):
        from ultralytics import YOLO
        self.mot_predictor = YOLO("yolo11m.pt",verbose=False)
        self.performance.addTask('human_detect',precision=4)
        
    def initFightDetector(self):
        self.with_video_action = self.cfg.get('VIDEO_ACTION',False)['enable'] if self.cfg.get('VIDEO_ACTION',False) else False
        if self.with_video_action:
            cfg_fight = self.cfg['VIDEO_ACTION']
            model_dir = cfg_fight['model_dir']
            batch_size = cfg_fight['batch_size']
            short_size = cfg_fight['short_size']
            target_size = cfg_fight['target_size']
            self.fight_detector = XFightDetectModel(
                model_dir,
                batch_size=batch_size,
                device='GPU',
                short_size=short_size,
                target_size=target_size)
            self.frame_len = self.cfg["VIDEO_ACTION"]["frame_len"] ## desired sample length
            self.sample_freq = self.cfg["VIDEO_ACTION"]["sample_freq"] ##
            self.display_frames = 30
            self.performance.addTask('fight_detect',precision=4)

    def init_annotator(self):
        self.annotator = XFightAnnotator()
    
    def predict_video(self):
        event_threshold = 0.75
        output = {'frame_list':[],'result_list':[],'perf_list':[]}
        for frame_count,(frame,im) in enumerate(self.capture):
            frame_count += 1
            #TOTAL 0.021s
            ## Detect Human 0.0089s
            res = next(self.mot_predictor(im,verbose=False,stream=True,conf=0.8)).cpu().numpy()# type: ignore
            self.boxes = res.boxes.data
            self.performance.setTask('human_detect',list(res.speed.values()),'ms')
            ### boxes: (N,6[xmin,ymin,xmax,ymax,conf,cls])
            if frame_count % self.sample_freq == 0:
                scaled_img = self.fight_detector.scale(im) ## 0.0014s
                self.sample_list.append(scaled_img)
            if len(self.sample_list) == self.frame_len:
                ## [0], [0.xxxx]
                self.fight = self.fight_detector.predict(self.sample_list) ## 0.025s 0: None, 1; Fight
                self.performance.setTask('fight_detect',list(self.fight_detector.recognize_times.report().values())[:3],'s')
                self.sample_list.clear()
                self.is_fight = self.fight[0] == 1 and self.fight[1] > event_threshold
            if self.is_fight:
                self.display_frames -= 1

            self.annotator.rescale_box(self.boxes)# 0.000022
            ## Visualize 0.009s
            perf = self.performance.getTaskAll('all')
            annotated_frame,onCallback = self.visualize(frame)
            if onCallback and not self.is_fight:
                self.callback.onPredictEpochEnd(annotated_frame,perf,self.fight)

            output['perf_list'].append(perf)
            output['frame_list'].append(self.annotator.bgr2rgb(annotated_frame))
            output['result_list'].append(self.fight)
        return output

    def predict_stream(self):
        event_threshold = 0.75
        for frame_count,(frame,im) in enumerate(self.capture):
            frame_count += 1
            #TOTAL 0.021s
            ## Detect Human 0.0089s
            res = next(self.mot_predictor(im,verbose=False,stream=True,conf=0.8)).cpu().numpy()# type: ignore
            self.boxes = res.boxes.data
            self.performance.setTask('human_detect',list(res.speed.values()),'ms')
            ### boxes: (N,6[xmin,ymin,xmax,ymax,conf,cls])
            if frame_count % self.sample_freq == 0:
                scaled_img = self.fight_detector.scale(im) ## 0.0014s
                self.sample_list.append(scaled_img)
            if len(self.sample_list) == self.frame_len:
                ## [0], [0.xxxx]
                self.fight = self.fight_detector.predict(self.sample_list) ## 0.025s 0: None, 1; Fight
                self.performance.setTask('fight_detect',list(self.fight_detector.recognize_times.report().values())[:3],'s')
                self.sample_list.clear()
                self.is_fight = self.fight[0] == 1 and self.fight[1] > event_threshold
            if self.is_fight:
                self.display_frames -= 1

            self.annotator.rescale_box(self.boxes)# 0.000022
            ## Visualize 0.009s
            perf = self.performance.getTaskAll('all')
            annotated_frame,onCallback = self.visualize(frame)
            if onCallback and not self.is_fight:
                self.callback.onPredictEpochEnd(annotated_frame,perf,self.fight,frame_count)
            ###########
            yield annotated_frame

    def visualize(self,image):
        self.annotator.init_image(image)
        self.annotator.begin()
        self.annotator.build(['BBox'])
        self.annotator.drawUserCount(len(self.boxes))
        for data in self.boxes:
            self.annotator.drawBBox(data)
        if self.display_frames > 0 and self.is_fight:
            self.annotator.drawAlert()
            self.annotator.end(['BBox','Alert','UserCount'])
            return self.annotator.result(),False
        if self.display_frames <=0 and self.is_fight:
            self.is_fight = False
            self.display_frames = 30
            self.annotator.end(['BBox','Alert','UserCount'])
            return self.annotator.result(),True
        self.annotator.end(['BBox','UserCount'])
        return self.annotator.result(),False
        
    def getVideo(self):
        self.callback.onPredictStart()
        output = self.predict_video()
        self.callback.onPredictEnd(output['frame_list'],output['perf_list'],output['result_list'])

    def getStream(self):
        for detect in self.predict_stream():
            yield getEncodeFrame(detect)
    
    def getImage(self):
        raise ValueError("No Supported Detect Type")

def createXFightDetectorTest(model_path,path,mode,config):
    detector = XFightDetectorTest(mode,model_path,path,config)
    return detector

