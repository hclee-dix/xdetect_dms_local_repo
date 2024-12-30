import os
import numpy as np
import torch
import cv2
import glob
from types import SimpleNamespace
#####
### init ###
from .net.slowfast.config.defaults import assert_and_infer_cfg
from .net.slowfast.utils.parser import load_config
from .net.slowfast.utils import misc as misc
from .net.slowfast.utils import checkpoint as cu
from .net.slowfast.utils import distributed as du
from .net.slowfast.models import build_model
from .net.default_arg import DEFAULT_ARGS
####

### transform
from .net.slowfast.datasets import cv2_transform
from .net.slowfast.visualization.utils import process_cv2_inputs
#####
### annotator
from PIL import Image, ImageDraw, ImageFont
from dev.util.style import InfoStyle,setColorLevel,add,sub,getVerticalAlign
### save output
from moviepy.editor import ImageSequenceClip
## util
from dev.util.util import t,dt

class Capture:
    def __init__(self,path,cfg):
        self.cfg = cfg
        self.init_video(path)
    
    def init_video(self,path):
        self.cv2 = cv2.VideoCapture(path)
        width = int(self.cv2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cv2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cv2.get(cv2.CAP_PROP_FPS))
        frame_count = int(self.cv2.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_meta = SimpleNamespace(**{'width':width,'height':height,'fps':fps,'frame_count':frame_count, 'scaleX':1/2,'scaleY':1/2, 'rescale':[]})
        self.video_meta.rescale = [1/self.video_meta.scaleX,1/self.video_meta.scaleY]
    
    def resize(self,source,dsize,fx,fy):
        return cv2.resize(source,dsize,None,fx,fy)
    
    def bgr2rgb(self,frame):
        return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    def scale(self,frame):
        return cv2_transform.scale(self.cfg.DATA.TEST_CROP_SIZE,frame)
    
    def retrieve(self):
        ret,frame = self.cv2.read()
        if not ret: raise StopIteration
        return frame
    
    def __iter__(self):
        return self
    
    def __next__(self):
        frame = self.retrieve()
        # im = self.resize(frame,dsize=(0,0),fx=self.video_meta.scaleX,fy=self.video_meta.scaleY)
        im = self.bgr2rgb(frame)
        im = self.scale(im)
        
        return frame,im

class XFireAnnotator:
    def __init__(self,meta) -> None:
        self.meta = meta
        
        self.style = InfoStyle(meta)
        self.buildFont()
        self.buildAsset()
        
        self.labelMap = {0:'Normal',1:'Smoke',2:'Fire'}

    def buildFont(self):
        font = ImageFont.truetype(str(self.style.text.font_family),self.style.text.font_size)
        alertFont = ImageFont.truetype(str(self.style.alertFont.font_family),self.style.alertFont.font_size)
        self.style.text.font = font
        self.style.alertFont.font = alertFont
        
    def buildAsset(self,asset_path='dev/core/asset/icon/'):
        resourcePathList = glob.glob(asset_path+'*.png')
        icon_style = self.style.default_icon
        resourceMap = {}
        for resourcePath in resourcePathList:
            resourceName = os.path.splitext(os.path.basename(resourcePath))[0]
            resource = Image.open(resourcePath)
            if resourceName in ['warning']:
                resource = resource.resize((self.style.alert_icon.width,self.style.alert_icon.height))
            else:
                resource = resource.resize((icon_style.width,icon_style.height))
            resourceMap.update({os.path.splitext(os.path.basename(resourcePath))[0]:resource})
        self.asset = SimpleNamespace(**resourceMap)
    
    def buildCanvas(self,image):
        if(image.ndim == 3):
            image = cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)
        self.im = image if isinstance(image,Image.Image) else Image.fromarray(image)
        self.topLayer = Image.new("RGBA",self.im.size,(0,0,0,0)) # type: ignore
        self.tcv = ImageDraw.Draw(self.topLayer,"RGBA") ## topLayer Canvas
    

    def drawBox(self,BoxCoord,style,label,score):
        innerCoord = sub(BoxCoord,[-self.style.fire.default.border_width]*2+[self.style.fire.default.border_width]*2)
        viewCoord = sub(innerCoord,[-self.style.fire.default_box.padding_left,-self.style.fire.default_box.padding_top,self.style.fire.default_box.padding_right,self.style.fire.default_box.padding_bottom])
        iconCoordLeft = getVerticalAlign(viewCoord,self.asset.warning.size)
        iconCoordRight = add(iconCoordLeft,[viewCoord[2]-viewCoord[0]-self.asset.warning.size[0],0])
        icon = setColorLevel(self.asset.warning,style.icon.color)
        self.tcv.rectangle(BoxCoord,style.box.background_color,style.default.border_color,self.style.fire.default.border_width)
        self.drawIcon(icon,iconCoordLeft)
        self.tcv.text([int((viewCoord[2]+viewCoord[0])/2),int((viewCoord[3]+viewCoord[1])/2)],label,style.text.color,self.style.alertFont.font,"mm")
        self.tcv.text([int((viewCoord[2]+viewCoord[0])/2),viewCoord[3]-self.style.fire.default.border_width],f'{score*100:02.1f}%',style.text.color,self.style.text.font,"mm")
        self.drawIcon(icon,iconCoordRight)
        
    def drawAlert(self,index):
        top1 = index[0]
        smoke = index[index[:,1]==1][0]
        fire = index[index[:,1]==2][0]
        outBoxCoord = [0,0,self.meta.width,self.meta.height]
        inBoxCoord = add(outBoxCoord,[self.style.fire.default.padding_left,self.style.fire.default.padding_top,-self.style.fire.default.padding_right,-self.style.fire.default.padding_bottom])
        ###
        top1Style = self.style.fire.alert if top1[1] == 2 else self.style.fire.warn
        self.tcv.rectangle(outBoxCoord,self.style.fire.default.background_color,top1Style.default.border_color,self.style.fire.default.border_width)
        FireBoxCoord = [inBoxCoord[0],inBoxCoord[3]-self.style.fire.default_box.height,inBoxCoord[0]+self.style.fire.default_box.width,inBoxCoord[3]]
        SmokeBoxCoord = [inBoxCoord[2]-self.style.fire.default_box.width,inBoxCoord[3]-self.style.fire.default_box.height,inBoxCoord[2],inBoxCoord[3]]
        if fire[0] > 0.1: self.drawBox(FireBoxCoord,self.style.fire.alert,'FIRE',fire[0])
        if smoke[0] > 0.1: self.drawBox(SmokeBoxCoord,self.style.fire.warn,'SMOKE',smoke[0])
        
    def drawRatioBar(self,index):
        ...
    
    def drawIcon(self,asset,pos):
        self.topLayer.paste(asset,pos,mask=asset)
    
    def compositeTopLayer(self):
        self.im = Image.alpha_composite(self.im,self.topLayer)

    def result(self):
        return np.asarray(self.im)

class XFireDetector:
    def __init__(self,*args,**kwargs):
        self.source = args[0]
        self.filename = args[1]
        self.cfg = self.init_config(kwargs)
        self.init_capture()
        self.init_model()
        self.init_variable()
        self.init_annotator()
    
    def init_config(self,kwargs):
        DEFAULT_ARGS.model_dir = kwargs['model_path']
        cfg = load_config(DEFAULT_ARGS,kwargs['config'])
        cfg = assert_and_infer_cfg(cfg)
        return cfg
    
    def init_model(self):
        self.initFireDetector()
        
    def init_capture(self):
        self.capture = Capture(self.source,self.cfg)
    
    def init_annotator(self):
        self.annotator = XFireAnnotator(self.capture.video_meta)
    
    def init_variable(self):
        DATA = self.cfg.DATA
        self.N = DATA.NUM_FRAMES * DATA.SAMPLING_RATE
        self.Q = []
        self.result = np.empty(0)
    
    def initFireDetector(self):
        du.init_distributed_training(self.cfg)
        np.random.seed(self.cfg.RNG_SEED)
        torch.manual_seed(self.cfg.RNG_SEED)
        
        self.model = build_model(self.cfg)
        if du.is_master_proc() and self.cfg.LOG_MODEL_INFO:
            misc.log_model_info(self.model, self.cfg, use_train_input=False)
        cu.load_test_checkpoint(self.cfg, self.model)
        self.model.eval()
        self.device = torch.cuda.current_device()
        
    def predict(self):
        a = []
        for frame_id,(frame,im) in enumerate(self.capture):
            frame_id += 1
            ret,sample,_ = self.getSample(im) ## [N,H,W,C]
            if ret:
                inputs = process_cv2_inputs(sample,self.cfg)
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(
                        device=torch.device(self.device), non_blocking=True
                    ) # 0.005
                preds = self.model(inputs,None) # 0.005
                torch.cuda.synchronize()
                preds = preds.detach().cpu() # 0.0004
                torch.cuda.synchronize()
                
                self.result = torch.cat(torch.topk(preds,3),dim=0).T.numpy()
                
            frame = self.visualize(frame)
            a.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            yield frame
        # print("start clip")
        # clip = ImageSequenceClip(a,fps=self.capture.video_meta.fps)
        # clip.write_videofile(f'{self.filename.split(".")[0]}_demo.mp4')
        # print('done clip')

    def visualize(self,image):
        im = np.ascontiguousarray(np.copy(image))
        self.annotator.buildCanvas(im)
        if len(self.result):
            ## self.result [topK,2] 2: score, classes
            ## classes: 0: 정상 1: 연기 2: 불꽃
            if self.result[0,1] != 0: ## top1 result classes
                self.annotator.drawAlert(self.result[self.result[:,1]>0])
                self.annotator.compositeTopLayer()
                
        return self.annotator.result()

    def getSample(self,frame):
        self.Q.append(frame)
        if len(self.Q) < self.N: return False,[],None
        else: return True, self.Q[:], self.Q.clear()
            
    def getStream(self):
        for detect in self.predict():
            _,encoded = cv2.imencode('.jpg',detect)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encoded) +
                b'\r\n'
            )
        
        
            
def createXFireDetector(name):
    base_path = 'dev/core/asset/video/'
    src = base_path + name
    detector = XFireDetector(
        src,
        name,
        model_path='/data2/xdetect_storage/temp/model/FIRE',
        config='dev/ai/common/config/base_fire_config.yaml'
    )
    return detector