from dev.ai.xBase.annotator import XAnnotator
from dev.ai.xBase.label import FIRE, DEFAULT_COLOR
from dev.ai.xFire.style import XFireStyle

class XFireAnnotator(XAnnotator):
    def __init__(self) -> None:
        super().__init__(FIRE,DEFAULT_COLOR)
        self.style = XFireStyle(self.meta)
    
    def drawBBox(self,BoxCoord,iconStyle,containerStyle,textStyle,label,score):
        self.alertCanvas.rectangle(BoxCoord,containerStyle.background_color,containerStyle.border_color,self.style.default_alert_container.border_width)
        ##
        innerCoord = self.style.sub(BoxCoord,[-self.style.default_alert_container.border_width]*2+[self.style.default_alert_container.border_width]*2)
        viewCoord = self.style.sub(innerCoord,[-self.style.alert_box.padding_left,-self.style.alert_box.padding_top,self.style.alert_box.padding_right,self.style.alert_box.padding_bottom])
        iconCoordLeft = self.style.getVerticalAlign(viewCoord,self.style.icon.src['warning'].size)
        iconCoordRight = self.style.add(iconCoordLeft,[viewCoord[2]-viewCoord[0]-self.style.icon.src['warning'].size[0],0])
        icon = self.style.setColorLevel(self.style.icon.src['warning'],iconStyle.color)
        self.drawIcon('Alert',icon,iconCoordLeft)
        self.alertCanvas.text([int((viewCoord[2]+viewCoord[0])/2),int((viewCoord[3]+viewCoord[1])/2)],label,textStyle.font_color,self.style.default_alert_text,"mm")
        self.alertCanvas.text([int((viewCoord[2]+viewCoord[0])/2),viewCoord[3]-self.style.default_alert_container.border_width],f'{score*100:02.1f}%',textStyle.font_color,self.style.default_text,"mm")
        self.drawIcon('Alert',icon,iconCoordRight)
    
    def drawAlert(self,index):
        top1 = index[0]
        smoke = index[index[:,1]==1][0]
        fire = index[index[:,1]==2][0]
        outBoxCoord = [0,0,self.meta.width,self.meta.height]
        inBoxCoord = self.style.add(outBoxCoord,[self.style.alert_container.padding_left,self.style.alert_container.padding_top,-self.style.alert_container.padding_right,-self.style.alert_container.padding_bottom])
        ###
        top1Style = self.style.default_alert_container if top1[1] == 2 else self.style.default_warn_container
        self.alertCanvas.rectangle(outBoxCoord,self.style.alert_container.background_color,top1Style.border_color,self.style.default_alert_container.border_width)
        FireBoxCoord = [inBoxCoord[0],inBoxCoord[3]-self.style.alert_box.height,inBoxCoord[0]+self.style.alert_box.width,inBoxCoord[3]]
        SmokeBoxCoord = [inBoxCoord[2]-self.style.alert_box.width,inBoxCoord[3]-self.style.alert_box.height,inBoxCoord[2],inBoxCoord[3]]
        if fire[0] > 0.1: self.drawBBox(FireBoxCoord,self.style.default_alert_icon_style,self.style.default_alert_box,self.style.alert_text,'FIRE',fire[0])
        if smoke[0] > 0.1: self.drawBBox(SmokeBoxCoord,self.style.default_warn_icon_style,self.style.default_warn_box,self.style.warn_text,'SMOKE',smoke[0])