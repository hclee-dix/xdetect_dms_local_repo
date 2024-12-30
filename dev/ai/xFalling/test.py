#
from dev.ai.xBase.detect import XDetector
#
import paddle
from .net.config import *
from .net.datacollector import *
from .net.mot_sde_infer import *
from .net.keypoint_infer import *
from .net.keypoint_postprocess import translate_to_ori_images
from .net.action_infer import *
from .net.action_utils import *
from .net.utils import *
from .net.preprocess import ShortSizeScale
#
from dev.ai.xFalling.callback import *
from dev.ai.xFalling.capture import XFallingDetectorCapture
from dev.ai.xFalling.annotator import XFallingAnnotator
from dev.ai.common.util import ChunkBuffer,softmax
#
from dev.util.util import getEncodeFrame

class XFallingDetectModel(SkeletonActionRecognizer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    def setMeta(self,meta):
        self.meta = meta
        
    def predict_skeleton(self, skeleton_list, run_benchmark=False, repeats=1):
        return np.array(super().predict_skeleton(skeleton_list, run_benchmark, repeats))
    
    def predict_skeleton_with_mot(self, skeleton_with_mot, run_benchmark=False):
        skeleton_list = skeleton_with_mot["skeleton"]
        mot_id = skeleton_with_mot["mot_id"]
        act_res = self.predict_skeleton(skeleton_list, run_benchmark, repeats=1)
        if mot_id.shape[0] == 0 or act_res.shape[0] == 0:
            return np.empty(0)
        act_res = softmax(act_res) ## softmax적용(-inf~+inf)=>(0~1) 합이 1
        pos = np.argpartition(act_res,-1)
        classes = np.argmax(pos,axis=1)
        scores = act_res[pos.astype(bool)]
        return np.concatenate([np.array(mot_id),classes,scores]).reshape(3,-1).T[::-1]
    def postprocess(self, inputs, result):
        return result['output'][0]
        
class XFallingKeypointStore(ChunkBuffer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        
    def add(self,kpts):
        ## kpts: (N,4+1+17,3[x,y,conf])
        ## mots: (N,7[xmin,ymin,xmax,ymax,track_id,conf,cls])
        #NOTE track_id별로 버퍼 업데이트(max_size:100)
        provided_uids = set(kpts[:, 4,0])  # 제공된 UID
        existing_uids = set(self.uids[self.uids != -1])  # 현재 등록된 UID
        uids_to_remove = existing_uids - provided_uids  # 제거해야 할 UID
        # 제거 작업
        for uid in uids_to_remove:
            index = self.uid_to_index[uid]
            self.uids[index] = -1
            self.current_lengths[index] = 0
            self.empty_slots = np.append(self.empty_slots, index)
            del self.uid_to_index[uid]
        
        for kpt in kpts:
            # kpt (22,3)
            uid = int(kpt[4,0])
            
            index = self._get_index(uid)
            if self.current_lengths[index]<self.max_length:
                self.buffers[index, self.current_lengths[index]] = kpt
                self.current_lengths[index] += 1
                self.uids[index] = uid  # UID 저장
                if self.current_lengths[index] >= self.max_length:
                    self.state = True
        
    def retrieve_and_clear(self,uid=None):
        #NOTE 꽉찬 track_id에 대해서 리턴후 버퍼에서 제거
        return super().retrieve_and_clear(uid)

class XFallingDetectorTest(XDetector):
    def __init__(self,mode,model_path,path,config):
        super().__init__()
        paddle.enable_static()
        self.mode = mode
        self.path = path
        self.cfg = self.init_config({'model_path':model_path,'config':config})
        
        self.result = np.empty(0)
        self.is_fall = False
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
        self.initFallingDetector()
    
    def init_callback(self):
        self.callback = XTestCallback(self.annotator.labelMap)
    
    def init_source(self):
        self.capture = XFallingDetectorCapture(self.mode,self.path,self.annotator.meta)
    
    def initHumanDetector(self):
        from ultralytics import YOLO
        self.mot_predictor = YOLO("yolo11l-pose.pt",verbose=False)
        self.performance.addTask('human_detect',precision=4)
        
    def initFallingDetector(self):
        self.with_skeleton_action = self.cfg.get('SKELETON_ACTION',False)['enable'] if self.cfg.get('SKELETON_ACTION',False) else False
        cfg_skeleton_action = self.cfg['SKELETON_ACTION']
        skeleton_model_dir = cfg_skeleton_action['model_dir']
        skeleton_batch_size = cfg_skeleton_action['batch_size']
        self.display_frames = cfg_skeleton_action['display_frames']
        self.coord_size = cfg_skeleton_action['coord_size']
        self.coord_size = np.array(self.coord_size).reshape(1,2,1,1)
        skeleton_action_frames = cfg_skeleton_action['max_frames']
        self.skeleton_action_predictor = XFallingDetectModel(skeleton_model_dir,device='GPU',batch_size=skeleton_batch_size)
        
        self.kpt_buff = XFallingKeypointStore(skeleton_action_frames,(22,3))
        self.performance.addTask('falling_detect',precision=4)
    
    def init_annotator(self):
        self.annotator = XFallingAnnotator()
    
    def predict_video(self):
        event_threshold = 0.95
        output = {'frame_list':[],'result_list':[],'perf_list':[]}
        for frame_count,(frame,im) in enumerate(self.capture):
            frame_count += 1
            #TOTAL 0.021s
            ## Detect Human 0.0089s
            res = next(self.mot_predictor.track(im,verbose=False,stream=True,conf=0.8)).cpu().numpy()# type: ignore
            self.performance.setTask('human_detect',list(res.speed.values()),'ms')
            ### boxes: (N,7[xmin,ymin,xmax,ymax,track_id,conf,cls])
            ### kpts: (N,17,3[x,y,conf])
            boxes,kpts = res.boxes.data,res.keypoints.data
            if boxes.shape[0] != 0:
                ## Detect Action 0.003935s
                ### kpts: (N, 22, 3)
                kpts = merge_mot_kpt_result(boxes,kpts) # 0.000110
                self.result = kpts.copy() # 0.000002
                self.kpt_buff.add(kpts) # 0.000026
                ##
                if self.kpt_buff.is_full():
                    collected_keypoint = self.kpt_buff.retrieve_and_clear() # 0.000051
                    skeleton_action_input = parse_keypoint(collected_keypoint,self.coord_size) # 0.000065
                    skel = self.skeleton_action_predictor.predict_skeleton_with_mot(skeleton_action_input) # 0.0036
                    self.performance.setTask('falling_detect',list(self.skeleton_action_predictor.det_times.report().values())[:3],'s')
                    merge_kpt_skel_result(self.result,skel) # 0.000061
                ##
                self.annotator.rescale_box(self.result)
                # rescale_box(self.result,self.capture.video_meta.rescale) # 0.000022
            ###########
            ## Visualize 0.009s
            perf = self.performance.getTaskAll('all')
            if np.any(self.result[:,3,1] == 0) and np.any(self.result[:,3,2] > event_threshold):
                self.is_fall = True
            if self.is_fall:
                self.display_frames -= 1

            annotated_frame,onCallback = self.visualize(frame)

            if onCallback and not self.is_fall:
                self.callback.onPredictEpochEnd(annotated_frame,perf,self.result)
            output['perf_list'].append(perf)
            output['frame_list'].append(self.annotator.bgr2rgb(annotated_frame))
            output['result_list'].append(self.result)
        return output

    def predict_stream(self):
        event_threshold = 0.95
        for frame_count,(frame,im) in enumerate(self.capture):
            frame_count += 1
            #TOTAL 0.021s
            ## Detect Human 0.0089s
            res = next(self.mot_predictor.track(im,verbose=False,stream=True,conf=0.8)).cpu().numpy()# type: ignore
            self.performance.setTask('human_detect',list(res.speed.values()),'ms')
            ### boxes: (N,7[xmin,ymin,xmax,ymax,track_id,conf,cls])
            ### kpts: (N,17,3[x,y,conf])
            boxes,kpts = res.boxes.data,res.keypoints.data
            if boxes.shape[0] != 0:
                ## Detect Action 0.003935s
                ### kpts: (N, 22, 3)
                kpts = merge_mot_kpt_result(boxes,kpts) # 0.000110
                self.result = kpts.copy() # 0.000002
                self.kpt_buff.add(kpts) # 0.000026
                ##
                self.performance.resetTask('falling_detect')
                if self.kpt_buff.is_full():
                    collected_keypoint = self.kpt_buff.retrieve_and_clear() # 0.000051
                    skeleton_action_input = parse_keypoint(collected_keypoint,self.coord_size) # 0.000065
                    skel = self.skeleton_action_predictor.predict_skeleton_with_mot(skeleton_action_input) # 0.0036
                    self.performance.setTask('falling_detect',list(self.skeleton_action_predictor.det_times.report().values())[:3],'s')
                    merge_kpt_skel_result(self.result,skel) # 0.000061
                ##
                self.annotator.rescale_box(self.result)
                # rescale_box(self.result,self.capture.video_meta.rescale) # 0.000022
            ## Visualize 0.009s
            perf = self.performance.getTaskAll('all')
            if np.any(self.result[:,3,1] == 0) and np.any(self.result[:,3,2] > event_threshold):
                self.is_fall = True
            if self.is_fall:
                self.display_frames -= 1
            
            annotated_frame,onCallback = self.visualize(frame)

            if onCallback and not self.is_fall:
                self.callback.onPredictEpochEnd(annotated_frame,perf,self.result)
            ###########
            yield annotated_frame

    def visualize(self,image):
        self.annotator.init_image(image)
        self.annotator.begin()
        self.annotator.build(['BBox'])
        self.annotator.drawUserCount(len(self.result))
        for data in self.result:
            self.annotator.drawBBox(data)
        if self.display_frames > 0 and self.is_fall:
            self.annotator.drawAlert()
            self.annotator.end(['BBox','Alert','UserCount'])
            return self.annotator.result(),False
        if self.display_frames <= 0 and self.is_fall:
            self.is_fall = False
            self.display_frames = self.cfg['SKELETON_ACTION']['display_frames']
            self.annotator.end(['BBox','Alert','UserCount'])
            return self.annotator.result(),True
        self.annotator.end(['BBox','UserCount'])
        return self.annotator.result(),False
        
    def getVideo(self):
        output = self.predict_video()
        self.callback.onPredictEnd(output['frame_list'],output['perf_list'],output['result_list'])

    def getStream(self):
        for detect in self.predict_stream():
            yield getEncodeFrame(detect)
    
    def getImage(self):
        raise ValueError("No Supported Detect Type")

def createXFallingDetectorTest(model_path,path,mode,config):
    detector = XFallingDetectorTest(mode,model_path,path,config)
    return detector

