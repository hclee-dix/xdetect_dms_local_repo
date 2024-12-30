from typing import Literal,List
#
from PIL import Image, ImageDraw
import numpy as np
import cv2
#
from dev.schema.annotator import IAnnotatorMeta
#
from dev.ai.xBase.style import XStyle

class XAnnotator:
    def __init__(self,labelmap,labelColor=None):
        self.meta = IAnnotatorMeta()
        self.style = XStyle(self.meta)
        self.labelMap = labelmap
        self.labelColor = labelColor
        self.initCanvas()

### init
    def initCanvas(self):
        self.initUserCountCanvas()
        self.initAlertCanvas()
    
    def begin(self):
        if self.image.ndim == 3: self.image = cv2.cvtColor(self.image,cv2.COLOR_BGR2BGRA)
        self.im = self.image if isinstance(self.image,Image.Image) else Image.fromarray(self.image)
    
    def build(self,layers:List[Literal['BBox','Top']]):
        if 'BBox' in layers:
            self.initBBoxCanvas()
        if 'Top' in layers:
            self.initTopCanvas()

    def init_image(self,image):
        self.image = np.ascontiguousarray(np.copy(image))

    def initUserCountCanvas(self):
        self.userCountLayer = Image.new("RGBA",self.meta.size,"#00000000") # type: ignore
        self.userCountCanvas = ImageDraw.Draw(self.userCountLayer,"RGBA")
        
    def initTopCanvas(self):
        self.topLayer = Image.new("RGBA",self.im.size,"#00000000") # type: ignore
        self.topCanvas = ImageDraw.Draw(self.topLayer,"RGBA")
    
    def initBBoxCanvas(self):
        self.bboxLayer = Image.new("RGBA",self.im.size,"#00000000") # type: ignore
        self.bboxCanvas = ImageDraw.Draw(self.bboxLayer,"RGBA")
    
    def initAlertCanvas(self):
        self.alertLayer = Image.new("RGBA",self.meta.size,"#00000000") # type: ignore
        self.alertCanvas = ImageDraw.Draw(self.alertLayer,"RGBA")

### method
    def drawUserCount(self,count=0):
        style = self.style.user_count_style
        icon = self.style.icon.src['user']
        marginCoord = [style.margin_left,style.margin_top]*2
        boxCoord = [style.left,style.top,style.width,style.height]
        contaierCoord = self.style.add(marginCoord,boxCoord)
        innerCoord = self.style.sub(contaierCoord,[-style.border_width]*2+[style.border_width]*2)
        paddingCoord = [-style.padding_left,-style.padding_top,style.padding_right,style.padding_bottom]
        viewCoord = self.style.sub(innerCoord,paddingCoord)
        userIconCoord = self.style.getVerticalAlign(viewCoord,icon.size)
        userCountText = f"{count:02d}"
        userCountSize = self.style.getTextSize(self.style.default_text.getbbox(userCountText))
        userCountCoord = self.style.getVerticalAlign(self.style.add(viewCoord,[icon.size[0]+style.margin_left,0,0,0]),userCountSize)
        self.userCountCanvas.rounded_rectangle(contaierCoord,style.border_radius,style.background_color,style.border_color,style.border_width)
        self.drawIcon('UserCount',self.style.setColorLevel(icon,self.style.default_icon_style.color),userIconCoord)
        self.userCountCanvas.text(userCountCoord,userCountText,fill=self.style.default_text_style.font_color,font=self.style.default_text,anchor="lt")

    def drawBBox(self):...
    def drawAlert(self):...
    def drawIcon(self,canvas:Literal['UserCount','Top','BBox','Alert'],icon,coord):
        if canvas == 'UserCount':
            self.userCountLayer.paste(icon,coord,mask=icon)
        elif canvas == 'BBox':
           self.bboxLayer.paste(icon,coord,mask=icon)
        elif canvas == 'Alert':
            self.alertLayer.paste(icon,coord,mask=icon)
        elif canvas == 'Top':
            self.topLayer.paste(icon,coord,mask=icon)
        else:
            raise ValueError("No Assigned Canvas")
        
### post  
    def end(self,layers:List[Literal['UserCount','Top','BBox','Alert']]):
        if 'UserCount' in layers:
            self.mergeCanvas('UserCount')
        if 'BBox' in layers:
            self.mergeCanvas('BBox')
        if 'Alert' in layers:
            self.mergeCanvas('Alert')
        if 'Top' in layers:
            self.mergeCanvas('Top')
            
    def mergeCanvas(self,canvas:Literal['UserCount','Top','BBox','Alert']):
        if canvas == 'UserCount':
            self.im = Image.alpha_composite(self.im,self.userCountLayer)
        elif canvas == 'BBox':
            self.im = Image.alpha_composite(self.im,self.bboxLayer)
        elif canvas == 'Alert':
            self.im = Image.alpha_composite(self.im,self.alertLayer)
        elif canvas == 'Top':
            self.im = Image.alpha_composite(self.im,self.topLayer)
        return self.im
    
    def result(self):
        return np.asarray(self.im)
    
### util
    def rescale_box(self,box):
        box[:,0:4] *= np.array([self.meta.rescale[0],self.meta.rescale[1]]*2)
        
    def bgr2rgb(self,image):
        if image.ndim == 4:
            return cv2.cvtColor(image,cv2.COLOR_BGRA2RGB)
        else:
            return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)