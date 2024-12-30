import paddle
from .net.config import *
from .net.detector import *
from .net.keypoint_infer import *
from .net.keypoint_postprocess import translate_to_ori_images
from .net.action_infer import *
from .net.action_utils import *
from .net.mot_sde_infer import *
from .net.datacollector import *
from .net.utils import *
#
import torch
import re
from timeit import default_timer
from functools import partial
#
from types import SimpleNamespace
from typing import Literal,Union,overload,Optional,Tuple,Dict,List
from PIL import Image, ImageDraw, ImageFont
from decord import gpu,VideoReader
from dev.util.style import InfoStyle,setColorLevel,add,sub,getVerticalAlign,hex2rgba
from moviepy.editor import ImageSequenceClip


class Capture:
    def __init__(self,path,mode:Literal['cv2','decord']) -> None:
        self.path = path
        self.mode = mode
        self.init_video(path,mode)
    
    def init_video(self,path,mode:Literal['cv2','decord']):
        if mode == 'cv2':
            self.cv2 = cv2.VideoCapture(path)
            width = int(self.cv2.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cv2.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cv2.get(cv2.CAP_PROP_FPS))
            frame_count = int(self.cv2.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_meta = SimpleNamespace(**{'width':width,'height':height,'fps':fps,'frame_count':frame_count, 'scaleX':1/2,'scaleY':1/2, 'rescale':[]})
            self.video_meta.rescale = [1/self.video_meta.scaleX,1/self.video_meta.scaleY]
        elif mode == 'decord':
            self.dc = VideoReader(path,gpu(0))
            self.dc.skip_frames(3000)
     
    def resize(self,source,dsize,fx,fy):
        if self.mode == 'decord':
            return cv2.resize(source[:,:,::-1],dsize,None,fx,fy)
        else:
            return cv2.resize(source,dsize,None,fx,fy)
    
    def retrieve(self):
        if self.mode == 'cv2':
            ret,frame = self.cv2.read()
            if not ret:
                raise StopIteration
            return frame
        else:
            try:
                frame= self.dc.next()
                frame = frame.asnumpy()
            except StopIteration:
                raise StopIteration
            return frame
        
    def __iter__(self):
        return self
    
    def __next__(self):
        frame = self.retrieve()
        im = self.resize(frame,dsize=(0,0),fx=self.video_meta.scaleX,fy=self.video_meta.scaleY)
        return frame,im

class XSmokingAnnotator:
    def __init__(self,meta) -> None:
        self.meta = meta
        
        self.style = InfoStyle(meta)
        
        self.userCount = 0
        
        self.buildFont()
        self.buildAsset()
        self.buildSkeletonUI()
        
    def buildFont(self,ret=False):
        font_style = self.style.text
        font_family = font_style.font_family
        font = ImageFont.truetype(str(font_family),font_style.font_size)
        labelFont = ImageFont.truetype(str(font_family),font_style.label_font_size)
        if ret: return font
        else: 
            self.style.text.font = font
            self.style.text.labelFont = labelFont
            
    def buildAsset(self,asset_path='dev/core/asset/icon/'):
        resourcePathList = glob.glob(asset_path+'*.png')
        icon_style = self.style.default_icon
        resourceMap = {}
        for resourcePath in resourcePathList:
            resource = Image.open(resourcePath)
            resource = resource.resize((icon_style.width,icon_style.height))
            resourceMap.update({os.path.splitext(os.path.basename(resourcePath))[0]:resource})
        self.asset = SimpleNamespace(**resourceMap)
    
    def buildSkeletonUI(self):
        container_style = self.style.container
        self.skeleton_ui = Image.new("RGBA",(container_style.width,container_style.height),container_style.background_color) # type: ignore
        self.ui = ImageDraw.Draw(self.skeleton_ui,"RGBA")
        self.drawSkeleton()
        
    def buildCanvas(self,image):
        if(image.ndim == 3):
            image = cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)
        self.im = image if isinstance(image,Image.Image) else Image.fromarray(image)
        self.layer = Image.new("RGBA",self.im.size,(0,0,0,0)) # type: ignore
        self.topLayer = Image.new("RGBA",self.im.size,(0,0,0,0)) # type: ignore
        self.tcv = ImageDraw.Draw(self.topLayer,"RGBA")
        self.cv = ImageDraw.Draw(self.layer,"RGBA")
        
    def buildInfo(self,image):
        self.buildCanvas(image)
        self.compositeUI()
        ##
        
    def compositeSkeleton(self):
        self.im = Image.alpha_composite(self.im,self.skeleton_ui)
    
    def compositeTopLayer(self):
        self.im = Image.alpha_composite(self.im,self.topLayer)
    
    def compositeUI(self):
        self.im = Image.alpha_composite(self.im,self.layer)
    
    def drawArrowLabel(self,res):
        # res [id,score,xyxy,xyxy]
        boxCoord = add(res[2:6].astype(int),list(res[6:8])*2)
        boxCoord *= np.array(self.meta.rescale*2) # [xmin, ymin, xmax, ymax]
        containerStyle = self.style.arrowLabelContainer
        smokingStyle = self.style.smokingContainer
        arrowStyle = self.style.arrowPoint
        iconStyle = self.style.default_icon
        asset = self.asset
        ## 
        #1 draw label
        containerCenterCoord = [int(boxCoord[0]+(boxCoord[2]-boxCoord[0])/2),int(boxCoord[1]-containerStyle.margin_bottom-containerStyle.height-arrowStyle.height)]
        containerCoord = [int(containerCenterCoord[0]-containerStyle.width/2),containerCenterCoord[1],int(containerCenterCoord[0]+containerStyle.width/2),containerCenterCoord[1]+containerStyle.height]
        arrowCenterCoord = [containerCenterCoord[0],containerCenterCoord[1]+containerStyle.height]
        arrowCoord = [(int(arrowCenterCoord[0]-arrowStyle.width/2),arrowCenterCoord[1]-containerStyle.border_width+self.style.h(1)),(int(arrowCenterCoord[0]+arrowStyle.width/2),arrowCenterCoord[1]-containerStyle.border_width+self.style.h(1)),(arrowCenterCoord[0],arrowCenterCoord[1]+arrowStyle.height-containerStyle.border_width)]
        bottomContainerCoord = [arrowCoord[0][0]+containerStyle.border_width,arrowCoord[0][1],arrowCoord[1][0]-containerStyle.border_width,arrowCoord[1][1]+containerStyle.border_width]
        ## draw box
        if (boxCoord[3]-boxCoord[1] < containerStyle.border_radius) or(boxCoord[2]-boxCoord[0] < containerStyle.border_radius):
            return
        self.cv.rounded_rectangle(boxCoord,containerStyle.border_radius,smokingStyle.background_color,smokingStyle.border_color,containerStyle.border_width)
        self.cv.rounded_rectangle(containerCoord,containerStyle.border_radius,containerStyle.background_color,containerStyle.border_color,containerStyle.border_width)
        ## draw arrow
        self.cv.polygon(arrowCoord,containerStyle.background_color,containerStyle.border_color,containerStyle.border_width)
        ## remove duplicate border
        self.cv.rectangle(bottomContainerCoord,containerStyle.background_color)
        #2 input icon
        innerCoord = sub(containerCoord,[-containerStyle.border_width]*2+[containerStyle.border_width]*2)
        paddingCoord = [-containerStyle.padding_left,-containerStyle.padding_top,containerStyle.padding_right,containerStyle.padding_bottom]
        viewCoord = sub(innerCoord,paddingCoord)
        ### smoking
        smokingIconCoord = getVerticalAlign(viewCoord,asset.female.size)
        #setColor by Score
        smokingScore = res[1]
        smokingIcon = asset.smoking
        # smokingIcon = setColorLevel(smokingIcon,iconStyle.color,iconStyle.levelColor,int((1-smokingScore)*smokingIcon.size[1]))
        smokingIcon = setColorLevel(smokingIcon,iconStyle.color)
        self.layer.paste(smokingIcon,smokingIconCoord,smokingIcon)
        self.drawDefaultText(smokingIconCoord,f'{smokingScore*100:02.0f}%')
    
    def drawSkeleton(self):
        style = self.style.background
        asset = self.asset
        #
        # margin
        marginCoord = [style.margin_left,style.margin_top]*2
        # background
        boxCoord = [style.left,style.top,style.width,style.height]
        backgroundCoord =add(marginCoord,boxCoord)
        self.ui.rounded_rectangle(backgroundCoord,style.border_radius,style.background_color,style.border_color,style.border_width)
        innerCoord = sub(backgroundCoord,[-style.border_width]*2+[style.border_width]*2)
        # padding
        paddingCoord = [-style.padding_left,-style.padding_top,style.padding_right,style.padding_bottom]
        # view
        self.viewCoord = sub(innerCoord,paddingCoord)
        #
        userIconCoord = getVerticalAlign(self.viewCoord,asset.user.size)
        userIcon = setColorLevel(asset.user,self.style.default_icon.color)
        self.drawIcon('ui',userIcon,userIconCoord)
    
    def drawDefaultText(self,refCoord,text):
        textStyle = self.style.text
        textSize = (lambda size:[size[2]-size[0],size[3]-size[1]])(textStyle.labelFont.getbbox(text))
        textView = add(refCoord,[self.asset.user.size[0]+self.style.w(10),0,0,0])
        textViewCoord = getVerticalAlign(textView,textSize)
        self.cv.text(textViewCoord,text,fill=textStyle.font_color,font=textStyle.labelFont,anchor='lt')
    
    def drawUserCount(self):
        countStyle = self.style.countTextView
        textStyle = self.style.text
        userCountText = f"{self.userCount:02d}"
        userCountTextSize = (lambda size:[size[2]-size[0],size[3]-size[1]])(textStyle.font.getbbox(userCountText))
        userCountTextView = add(self.viewCoord,[self.asset.user.size[0]+countStyle.margin_left,0,0,0])
        userCountTextCoord = getVerticalAlign(userCountTextView,userCountTextSize)
        self.tcv.text(userCountTextCoord,userCountText,fill=textStyle.font_color,font=textStyle.font,anchor='lt')

    
    def drawIcon(self,mode:Literal['ui','cv'],asset,pos):
        if mode == 'ui':
            self.skeleton_ui.paste(asset,pos,mask=asset)
        else:
            self.im.paste(asset,pos,mask=asset)
            
    def result(self):
        return np.asarray(self.im)

class XSmokingDetectModel(DetActionRecognizer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.result_history = np.empty((1,6))
    
    def setMeta(self,meta):
        self.meta = meta
        
    def predict(self, images, mot_result,bbox):
        # result [id,score,xmin,ymin,xmax,ymax,w,h,i,j,k]
        if self.skip_frame_cnt == 0 or (not self.check_id_is_same(mot_result)):
            det_result = self.detector.predict_image(images, visual=False)
            result = self.postprocess(det_result, mot_result,bbox)
        else:
            result = self.reuse_result(mot_result)

        self.skip_frame_cnt += 1
        if self.skip_frame_cnt >= self.skip_frame_num:
            self.skip_frame_cnt = 0

        return result
        
    def postprocess(self, det_result, mot_result,bbox):
        ## mot_result [N,11] [id, score, x, y, w, h, w, h, i, j, k]
        box_candidate_list = det_result['boxes'] #boxes [N*M,6] => [cls,score,xmin,ymin,xmax,ymax], M is candidate number
        box_candidate_index = det_result['boxes_num'] # [N,] => [M,...]
        # ids,inds = mot_result[:,0],np.arange(mot_result[:,0]) # [N,]
        current_index = 0
        result = []
        for i,b in enumerate(box_candidate_index):
            a,current_index = box_candidate_list[current_index:current_index + b],current_index+b
            r = a[(a[:, 0] == 0) & (a[:, 1] > self.threshold)] # [m,6] 0<=m<=M
            if r.shape[0] == 0: continue ## no candidate : pass
            r[0][0] = mot_result[i,0]
            res = np.concatenate((r[0],bbox[i]),axis=0)
            ## [id, score, candidate_box[x,y,x,y], crop_box[x,y,x,y]]
            result.append(res) # choose first candidate where cls == smoking(0) and score > 0.4
        self.result_history = np.array(result)
        return self.result_history
    
    def reuse_result(self, mot_result):
        return self.result_history


class XSmokingDetector:
    def __init__(self,*args,**kwargs):
        paddle.enable_static()
        self.with_draw = True
        self.show_info = True
        self.source = args[0]
        self.device = 'GPU'
        self.cfg = self.init_config(kwargs)
        self.result = np.empty(0)
        self.status = {"프레임수":0,"탐지":{'start':0.0,'end':0.0},"흡연":{'start':0.0,'end':0.0},'표시':{'start':0.0,'end':0.0},'전체':0.0}
        self.init_capture()
        self.init_model()
        self.init_annotator()
        
    def init_config(self,args):
        FLAGS = get_pipeline_cfg(args)
        cfg = merge_cfg(FLAGS)
        print_arguments(cfg)
        return cfg
    
    def init_model(self):
        self.initMotDetector()
        # self.testDetector()
        self.initSmokingDetector()
    
    def init_capture(self):
        self.capture = Capture(self.source,'cv2')
    def testDetector(self):
        from ultralytics import YOLO
        self.testmodel = YOLO("yolo11m.pt",verbose=False)
    def initSmokingDetector(self):
        self.with_idbased_detection = self.cfg.get('ID_BASED_DETECTION',False)['enable'] if self.cfg.get('ID_BASED_DETECTION',False) else False
        if self.with_idbased_detection:
            cfg_smoking = self.cfg['ID_BASED_DETECTION']
            model_dir = cfg_smoking['model_dir']
            batch_size = cfg_smoking['batch_size']
            threshold = cfg_smoking['threshold']
            display_frames = cfg_smoking['display_frames']
            skip_frame_num = cfg_smoking.get('skip_frame_num',-1)
            self.smoking_predictor = XSmokingDetectModel(model_dir,batch_size=batch_size,device=self.device,display_frames=display_frames,threshold=threshold,skip_frame_num=skip_frame_num)
            self.smoking_predictor.setMeta(self.capture.video_meta)
    def initMotDetector(self):
        self.with_mot = self.cfg.get('MOT',False)['enable'] if self.cfg.get('MOT',False) else False
        if self.with_mot:
            cfg_mot = self.cfg['MOT']
            cfg_track = cfg_mot['tracker_config']
            model_dir = cfg_mot['model_dir']
            batch_size = cfg_mot['batch_size']
            skip_frame_num = cfg_mot.get('skip_frame_num',-1)
            self.mot_predictor = SDE_Detector(model_dir,cfg_track,self.device,batch_size=batch_size,skip_frame_num=skip_frame_num,threshold=0.8)
        
    def init_annotator(self):
        self.annotator = XSmokingAnnotator(self.capture.video_meta)
    
    def predict(self):
        ##
        
        a = []
        for i,(frame,im) in enumerate(self.capture):
            frame_count = i+1
            ####### yolo
            # [N,7] [xmin ymin xmax ymax id, conf, cls]
            # result = next(self.testmodel.track(im,stream=True,classes=[0],conf=0.7,verbose=False)).boxes.data # type: ignore 
            # if result.shape[0] != 0:
            #     crop_input = crop_image_yolo(im,result)
            #     smoke_res = self.smoking_predictor.predict(crop_input,result)
            #     print(smoke_res)
            
            mot_res = self.mot_predictor.predict_image(
                [copy.deepcopy(im)],
                visual=False,
                frame_count=frame_count,
            )[0] # [N,11] [id, score, x, y, w, h, w, h, i, j, k]
            

            if mot_res.shape[0] != 0:
                crop_input,bbox = crop_image(im, mot_res)
                smoke_res = self.smoking_predictor.predict(crop_input,mot_res,bbox)
                self.result = np.array(smoke_res)
            f = self.visualize(frame,mot_res)
            a.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
            yield(f)
        # print("start clip")
        # clip = ImageSequenceClip(a,fps=30)
        # clip.write_videofile('smoking_demo.mp4')
        # print('done clip')
            
    
    def getStatus(self):
        return f"프레임#{self.status['프레임수']:4d}\n탐지#{self.status['탐지']['end']:.4f}\n흡연#{self.status['흡연']['end']:.4f}\n표시#{self.status['표시']['end']:.4f}\n전체#{self.status['전체']:.4f}"
    
    def visualize(self,image,mot_res):
        im = np.ascontiguousarray(np.copy(image))
        self.annotator.buildCanvas(im)
        if mot_res.shape[0] == 0:
            self.annotator.userCount = 0
            self.annotator.drawUserCount()
            self.annotator.compositeSkeleton()
            self.annotator.compositeTopLayer()
            return self.annotator.result()
        for i,res in enumerate(self.result):
            self.annotator.drawArrowLabel(res)
            # self.annotator.drawBox(res,mot_res[i])

        self.annotator.userCount = mot_res.shape[0]
        self.annotator.drawUserCount()
        self.annotator.compositeUI()
        self.annotator.compositeSkeleton()
        self.annotator.compositeTopLayer()
        return self.annotator.result()
    
    # def visualize(self,image,result,frame_count):
    #     try:
    #         start = default_timer()
    #         self.status['표시']['start'] = start
    #         ## initialize Annotator
    #         im = np.ascontiguousarray(np.copy(image))
    #         annotator = XSmokingAnnotator(
    #             im, 
    #             line_width=None,
    #             pil=True,
    #             font_size=20,
    #             smok_font_size=15,
    #             font='dev/core/NanumSquareNeoB.ttf',
    #         )
    #         ## get result
    #         if result is None:
    #             annotator.drawInfo(self.getStatus())
    #             return annotator.result()
    #         mot_res = copy.deepcopy(result.get('mot'))
    #         smoke_res = copy.deepcopy(result.get('smoke'))
    #         if mot_res is not None:
    #             ids = mot_res['boxes'][:, 0]
    #             scores = mot_res['boxes'][:, 2]
    #             boxes = mot_res['boxes'][:, 3:]
    #             boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    #             boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    #             ref = np.array(ids,dtype=int)
            
    #         if smoke_res is not None:
    #             smoke_res = smoke_res['output']
    #             print(smoke_res)
                
            
    #         ## draw attr
    #         for i, (box,id,box_score,attr) in enumerate(zip(boxes,ids,scores,smoke_res)):
    #             x1,y1,w,h = box
    #             intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
    #             annotator.drawSubInfo(intbox,box_score,0,color=(11, 255, 162))
            
    #         ## draw status
    #         annotator.drawInfo(self.getStatus())
    #         end = default_timer() - start
    #         self.status['표시']['end'] = end
    #         return annotator.result()
    #     except Exception as e:
    #         raise Exception
    
    def getStream(self):
        for detect in self.predict():
            _,encoded = cv2.imencode('.jpg',detect)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encoded) +
                b'\r\n'
            )

    

class DetectCallback:
    ## on detect
    @staticmethod
    def on_fall_detect():
        print("넘어짐 감지",time.strftime("%Y-%m-%dT%H:%M:%S %z"))
        #TODO: firebase

def createXSmokingDetector(name):
    base_path = 'dev/core/asset/video/'
    src = base_path + name
    detector = XSmokingDetector(
        src,
        model_path='/data2/xdetect_storage/temp/model/SMOKE',
        config='dev/ai/common/config/base_smoking_config.yaml')
    return detector