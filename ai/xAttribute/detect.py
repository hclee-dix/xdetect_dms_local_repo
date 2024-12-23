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
from timeit import default_timer
#
from types import SimpleNamespace
from typing import Literal,Union,overload,Optional,Tuple,Dict,List
from PIL import Image, ImageDraw, ImageFont
from decord import gpu,VideoReader
from dev.util.style import InfoStyle,setColorLevel,add,sub,getVerticalAlign,hex2rgba
from dev.ai.common.util import EMA
from functools import partial
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
            self.video_meta = SimpleNamespace(**{'width':width,'height':height,'fps':fps,'frame_count':frame_count, 'scaleX':0.5,'scaleY':0.5, 'rescale':[]})
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

class XAttributeType:
    position:Literal['left','top','right','bottom']
    title:List[str]
    score:float
    threshold:float=0.7
    disable:bool=False

class XAttribute:
    def __init__(self,
                 position:Literal['left','top','right','bottom']='right',
                 title:List[str]=['default','기본정보'],
                 score:float=0.0,
                 disable:bool=False,
                 threshold:float=0.7
                 ) -> None:
        self.position = position
        self.title = self.init_title(title)
        self.threshold = threshold
        self.score = score
        self.disable = disable
        self.is_visible = False
    
    def init_title(self,title_list:List[str])->Dict[Literal['en','ko'],str]:
        return {'en':title_list[0],'ko':title_list[1]}
    
    @classmethod
    def create_item(cls,position:Literal['left','top','right','bottom'],title_list:List[List[str]],score_list:List[float],bi=False):
        if bi:
            score = score_list[0]
            titleList = title_list[0 if score>0.5 else 1]
            score = abs(score - 0.5)*2
            instance = cls(position=position,score=score)
            instance.title = instance.init_title(titleList)
            return instance
        score_index,score = np.argmax(score_list),np.max(score_list)
        titleList = title_list[score_index]
        instance = cls(position=position,score=score)
        instance.title = instance.init_title(titleList)
        return instance

"""
    0: Hat  
    1: Glasses  
    2:7: Uppers(ShortSleeve,LongSleeve,UpperStride,UpperLogo,UpperPlaid,UpperSplice)
    8:13: Lowers(LowerStripe,LowerPattern,LongCoat,Trousers,Shorts,Skirt&Dress)
    14: Boots
    15:17: Bags(HandBag, ShoulderBag, Backpack)
    18: HoldObjectsInFront
    19:21: ages(AgeLess18,Age18-60,AgeOver60)
    22: sexuality(Male,Female)
    23:25: direction(Front, Side, Back)
"""
class XAttributeMap:
    def __init__(self,
                 hat:XAttributeType,
                 glasses:XAttributeType,
                 uppers:XAttributeType,
                 lowers:XAttributeType,
                 boots:XAttributeType,
                 bags:XAttributeType,
                 hold:XAttributeType,
                 ages:XAttributeType,
                 sex:XAttributeType,
                 direction:XAttributeType
                 ) -> None:
        self.hat=XAttribute(**hat.__dict__)
        self.glasses=XAttribute(**glasses.__dict__)
        self.uppers=XAttribute(**uppers.__dict__)
        self.lowers=XAttribute(**lowers.__dict__)
        self.boots=XAttribute(**boots.__dict__)
        self.bags=XAttribute(**bags.__dict__)
        self.hold=XAttribute(**hold.__dict__)
        self.ages=XAttribute(**ages.__dict__)
        self.sex=XAttribute(**sex.__dict__)
        self.direction=XAttribute(**direction.__dict__)
    
    def assign(self,**kwargs):
        self.__dict__.update(**kwargs)
        
    def getPosition(self,position:Literal['left','top','right','bottom'])->List[XAttribute]:
        o = []
        for attribute in self.__dict__.values():
            if attribute.position == position:
                o.append(attribute)
        return o
    
    def mergeResult(self,targetList:List[XAttribute],lang:Literal['en','ko']='en'):
        resultList = {'position':[],'score':[],'title':[],'label':[],'is_visible':[]}
        for i,target in enumerate(targetList):
            resultList['position'].append(target.position)
            resultList['score'].append(f'{target.score*100:02.1f}%')
            resultList['title'].append(f'{target.title[lang]:<10}')
            resultList['is_visible'].append(target.score > target.threshold)
        return resultList
    
    @classmethod
    def createAttributeMap(cls):
        return cls(
            XAttributeType(),
            XAttributeType(),
            XAttributeType(),
            XAttributeType(),
            XAttributeType(),
            XAttributeType(),
            XAttributeType(),
            XAttributeType(),
            XAttributeType(),
            XAttributeType(),
        )

class XAttributeAnnotator():
    # def __init__(self,im,line_width=None,font_size=None,attr_font_size=None,font="Arial.ttf",ref_name='abc') -> None:
    def __init__(self,meta) -> None:
        ## ref : video_meta(width,height,fps,frame_count)
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
        containerStyle = self.style.arrowLabelContainer
        arrowStyle = self.style.arrowPoint
        iconStyle = self.style.default_icon
        asset = self.asset
        ## res:: id, score, xmin, ymin,xmax,ymax, width, height
        ## 
        #1 draw label
        containerCenterCoord = [int(res[2]+res[6]/2),int(res[3]-containerStyle.margin_bottom-containerStyle.height-arrowStyle.height)]
        containerCoord = [int(containerCenterCoord[0]-containerStyle.width/2),containerCenterCoord[1],int(containerCenterCoord[0]+containerStyle.width/2),containerCenterCoord[1]+containerStyle.height]
        arrowCenterCoord = [containerCenterCoord[0],containerCenterCoord[1]+containerStyle.height]
        arrowCoord = [(int(arrowCenterCoord[0]-arrowStyle.width/2),arrowCenterCoord[1]-containerStyle.border_width+self.style.h(1)),(int(arrowCenterCoord[0]+arrowStyle.width/2),arrowCenterCoord[1]-containerStyle.border_width+self.style.h(1)),(arrowCenterCoord[0],arrowCenterCoord[1]+arrowStyle.height-containerStyle.border_width)]
        bottomContainerCoord = [arrowCoord[0][0]+containerStyle.border_width,arrowCoord[0][1],arrowCoord[1][0]-containerStyle.border_width,arrowCoord[1][1]+containerStyle.border_width]
        ## draw box
        self.cv.rounded_rectangle(containerCoord,containerStyle.border_radius,containerStyle.background_color,containerStyle.border_color,containerStyle.border_width)
        ## draw arrow
        self.cv.polygon(arrowCoord,containerStyle.background_color,containerStyle.border_color,containerStyle.border_width)
        ## remove duplicate border
        self.cv.rectangle(bottomContainerCoord,containerStyle.background_color)
        #2 input icon
        if res[30] != 0:
            innerCoord = sub(containerCoord,[-containerStyle.border_width]*2+[containerStyle.border_width]*2)
            paddingCoord = [-containerStyle.padding_left,-containerStyle.padding_top,containerStyle.padding_right,containerStyle.padding_bottom]
            viewCoord = sub(innerCoord,paddingCoord)
            ### sexuality
            sexualityIconCoord = getVerticalAlign(viewCoord,asset.female.size)
            #setColor by Score
            sexuality = False if res[30]>0.5 else True ## False: female, True: male
            sexualityScore = abs(res[30] - 0.5)*2
            sexualityIcon = asset.male if sexuality else asset.female
            # sexualityIcon = setColorLevel(sexualityIcon,iconStyle.color,iconStyle.levelColor,int((1-sexualityScore)*sexualityIcon.size[1]))
            sexualityIcon = setColorLevel(sexualityIcon,iconStyle.color)
            self.layer.paste(sexualityIcon,sexualityIconCoord,sexualityIcon)
            self.drawDefaultText(sexualityIconCoord,f'{sexualityScore*100:02.0f}%')
        
    
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

    def drawIcon(self,mode:Literal['ui','cv'],asset,pos):
        if mode == 'ui':
            self.skeleton_ui.paste(asset,pos,mask=asset)
        else:
            self.im.paste(asset,pos,mask=asset)
        
        
    def result(self):
        return np.asarray(self.im)
    
    

class XAttrDetectorModel(AttrDetector):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    
    def postprocess(self, inputs, result):
        return result['output']
        
class XAttributeDetector:
    def __init__(self,*args,**kwargs):
        paddle.enable_static()
        self.with_draw = True
        self.show_info = True
        self.source = args[0]
        self.device = 'GPU'
        self.cfg = self.init_config(kwargs)
        self.is_falling = {}
        self.status = {"프레임수":0,"탐지":{'start':0.0,'end':0.0},"특징":{'start':0.0,'end':0.0},'표시':{'start':0.0,'end':0.0},'전체':0.0}
        self.ema = EMA()
        self.init_model()
        self.init_capture()
        self.init_annotator()
        
    def init_config(self,args):
        FLAGS = get_pipeline_cfg(args)
        cfg = merge_cfg(FLAGS)
        print_arguments(cfg)
        return cfg
    
    def init_model(self):
        self.initMotDetector()
        self.initAttrDetector()
    
    def initAttrDetector(self):
        self.with_attr = self.cfg.get('ATTR',False)['enable'] if self.cfg.get('ATTR',False) else False
        if self.with_attr:
            cfg_attr = self.cfg['ATTR']
            model_dir = cfg_attr['model_dir']
            batch_size = cfg_attr['batch_size']
            self.attr_predictor = XAttrDetectorModel(model_dir,batch_size=batch_size,device=self.device)
            
    def initMotDetector(self):
        self.with_mot = self.cfg.get('MOT',False)['enable'] if self.cfg.get('MOT',False) else False
        if self.with_mot:
            cfg_mot = self.cfg['MOT']
            cfg_track = cfg_mot['tracker_config']
            model_dir = cfg_mot['model_dir']
            batch_size = cfg_mot['batch_size']
            skip_frame_num = cfg_mot.get('skip_frame_num',-1)
            self.mot_predictor = SDE_Detector(model_dir,cfg_track,self.device,batch_size=batch_size,skip_frame_num=skip_frame_num,threshold=0.8)
    
    def init_capture(self):
        self.capture = Capture(self.source,'cv2')
    
    def init_annotator(self):
        self.annotator = XAttributeAnnotator(self.capture.video_meta)
    
    def predict(self):
        a = []
        for i,(frame,im) in enumerate(self.capture):
            frame_count = i+1
            
            mot_res = self.mot_predictor.predict_image(
                [copy.deepcopy(im)],
                visual=False,
                frame_count=frame_count,
            )[0] # [N, 34] [id, score,x,y,x,y,w,h] + [...attr]
            if mot_res.shape[0] != 0:
                crop_input = crop_image(im,mot_res)
                attr_res = self.attr_predictor.predict_image(
                    crop_input,
                    visual=False
                )
                self.ema.update(mot_res[:,0],attr_res)
                mot_res[:,8:] = self.ema.get(mot_res[:,0])[:,:,-1] # [N, 26]
            
            rescale_box(mot_res,self.capture.video_meta.rescale)
            self.result = mot_res
            f = self.visualize(frame)
            a.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
            yield(f)
        # print("start clip")
        # clip = ImageSequenceClip(a,fps=30)
        # clip.write_videofile('attr_demo.mp4')
        # print('done clip')
        
    
        
    
    def getStatus(self,show=False):
        a = f"프레임#{self.status['프레임수']:4d}\n탐지#{self.status['탐지']['end']:.4f}\n특징#{self.status['특징']['end']:.4f}\n표시#{self.status['표시']['end']:.4f}\n전체#{self.status['전체']:.4f}"
        if show:print(a)
        return a
    
    def visualize(self,image):
        im = np.ascontiguousarray(np.copy(image))
        self.annotator.buildCanvas(im)
        if self.result.shape[0] == 0:
            self.annotator.userCount = 0
            self.annotator.drawUserCount()
            self.annotator.compositeSkeleton()
            self.annotator.compositeTopLayer()
            return self.annotator.result()
        
        for res in self.result:
            self.annotator.drawArrowLabel(res)   
        self.annotator.userCount = self.result.shape[0]
        self.annotator.drawUserCount()
        self.annotator.compositeUI()
        self.annotator.compositeSkeleton()
        self.annotator.compositeTopLayer()
        return self.annotator.result()
    
    def getStream(self):
        for detect in self.predict():
            # detect = cv2.resize(detect,(640,360))
            _,encoded = cv2.imencode('.jpg',detect)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encoded) +
                b'\r\n'
            )
        print("done")
    

class DetectCallback:
    ## on detect
    @staticmethod
    def on_fall_detect():
        print("넘어짐 감지",time.strftime("%Y-%m-%dT%H:%M:%S %z"))
        #TODO: firebase

def createXAttrDetector(name):
    base_path = 'dev/core/asset/video/'
    src = base_path + name
    detector = XAttributeDetector(
        src,
        model_path='/data2/xdetect_storage/temp/model/ATTR',
        config='dev/ai/common/config/base_attribute_config.yaml')
    return detector