from dev.ai.xBase.annotator import XAnnotator
from dev.ai.xBase.label import FIGHT,DEFAULT_COLOR
from dev.ai.xFight.style import XFightStyle
#

class XFightAnnotator(XAnnotator):
    def __init__(self) -> None:
        super().__init__(FIGHT,DEFAULT_COLOR)
        self.style = XFightStyle(self.meta)
        
    def drawBBox(self,box):
        boxCoord = box[0:4].astype(int)
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
        # res: [cls, score] 0: None, 1: Fight
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
        self.alertCanvas.text([int((viewCoord[2]+viewCoord[0])/2),int((viewCoord[3]+viewCoord[1])/2)],"FIGHTING",self.style.default_text_alert_style.font_color,self.style.default_alert_text,"mm")
        self.drawIcon('Alert',alertIcon,alertIconCoordRight)
        ##
