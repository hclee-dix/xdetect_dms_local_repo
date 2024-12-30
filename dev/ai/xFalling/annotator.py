from dev.ai.xBase.annotator import XAnnotator
from dev.ai.xBase.label import FALLING,DEFAULT_COLOR
from dev.ai.xFalling.style import XFallingStyle
#
import numpy as np

TARGET_ROW_RANGE = [np.r_[np.arange(3),np.arange(5,22)]]
class XFallingAnnotator(XAnnotator):
    def __init__(self) -> None:
        super().__init__(FALLING,DEFAULT_COLOR)
        self.style = XFallingStyle(self.meta)
        
    def drawBBox(self,box):
        # box[22,3]
        # 0 [xmin,    ymin,       0]
        # 1 [xmax,    ymax,       0]
        # 2 [width,   height,     0]
        # 3 [id,      skel_class, skel_score]
        # 4 [id,      det_class,  det_conf]
        # 5 [k1_x,    k1_y,       k1_score]
        #   [...]
        # 21[k17_x,   k17_y,      k17_score]
        
        boxCoord = [box[0,0],box[0,1],box[1,0],box[1,1]]
        label = 'HUMAN'
        ## label
        l,t,r,b = self.style.default_text.getbbox(label)
        labelCoord = [boxCoord[0]-1,boxCoord[1]-self.style.bboxLabel.height,boxCoord[0]+r-l+self.style.bboxContainer.border_width+self.style.bboxContainer.border_width+self.style.bboxLabel.padding_left+self.style.bboxLabel.padding_right,boxCoord[1]]
        
        self.bboxCanvas.rounded_rectangle(
            labelCoord,
            self.style.bboxContainer.border_radius,
            self.style.bboxContainer.background_color,
            self.style.bboxContainer.border_width,
            corners=(False,True,False,False)
        )
        
        innerCoord = self.style.sub(labelCoord,[-self.style.bboxContainer.border_width]*2+[self.style.bboxContainer.border_width]*2)
        paddingCoord = [-self.style.bboxLabel.padding_left,-self.style.bboxLabel.padding_top,self.style.bboxLabel.padding_right,self.style.bboxLabel.padding_bottom]
        viewCoord = self.style.sub(innerCoord,paddingCoord)
        ##
        self.bboxCanvas.text(viewCoord,label,fill=self.style.default_text_style.font_color,font=self.style.default_text,anchor='lt')
        ## box
        self.bboxCanvas.rounded_rectangle(
            boxCoord,
            self.style.bboxContainer.border_radius,
            None,
            self.style.bboxContainer.border_color,
            self.style.bboxContainer.border_width,
            corners=(False,True,True,True)) # tl tr br bl
        
    def drawAlert(self):
        # res: [cls, score] 0: None, 1: Falling
        outBoxCoord = [0,0,self.meta.width,self.meta.height]
        inBoxCoord = self.style.add([int(self.meta.width/2),int(self.meta.height/2)]*2,[-self.style.default_alert_box.width,-self.style.default_alert_box.height,self.style.default_alert_box.width,self.style.default_alert_box.height])
        self.alertCanvas.rectangle(outBoxCoord,self.style.default_alert_container.background_color,self.style.default_alert_container.border_color,self.style.default_alert_container.border_width)
        self.alertCanvas.rectangle(inBoxCoord,self.style.default_alert_box.background_color,self.style.default_alert_box.border_color,self.style.default_alert_box.border_width)
        ##
        innerCoord = self.style.sub(inBoxCoord,[-self.style.default_alert_box.border_width]*2+[self.style.default_alert_box.border_width]*2)
        viewCoord = self.style.sub(innerCoord,[-self.style.default_alert_box.padding_left,-self.style.default_alert_box.padding_top,self.style.default_alert_box.padding_right,self.style.default_alert_box.padding_bottom])
        alertIconCoordLeft = self.style.getVerticalAlign(viewCoord,self.style.icon.src['warning'].size)
        alertIconCoordRight = self.style.add(alertIconCoordLeft,[viewCoord[2]-viewCoord[0]-self.style.icon.src['warning'].size[0],0])
        alertIcon = self.style.setColorLevel(self.style.icon.src['warning'],self.style.default_alert_icon_style.color)
        self.drawIcon('Alert',alertIcon,alertIconCoordLeft)
        self.alertCanvas.text([int((viewCoord[2]+viewCoord[0])/2),int((viewCoord[3]+viewCoord[1])/2)],"FALLING",self.style.default_text_alert_style.font_color,self.style.default_alert_text,"mm")
        self.drawIcon('Alert',alertIcon,alertIconCoordRight)
        ##
    
    def rescale_box(self,box):
        # res[N, 22,3]
        # rs : disired rescale position
        #    rs        rs         X
        # 0 [xmin,    ymin,       0]            rs
        # 1 [xmax,    ymax,       0]            rs
        # 2 [width,   height,     0]            rs
        # 3 [id,      skel_class, skel_score]   X
        # 4 [id,      det_conf,   det_class]    X
        # 5 [k1_x,    k1_y,       k1_score]     rs
        #   [...]                               rs
        # 21[k17_x,   k17_y,      k17_score]    rs
        box[:,TARGET_ROW_RANGE,0:2] *= np.array([self.meta.rescale[0],self.meta.rescale[1]])