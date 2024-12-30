from dev.ai.xBase.annotator import XAnnotator
from dev.ai.xBase.label import YOLO_COCO,DEFAULT_COLOR
from dev.ai.xObject.style import XObjectStyle

class XObjectAnnotator(XAnnotator):
    def __init__(self) -> None:
        super().__init__(YOLO_COCO,DEFAULT_COLOR)
        self.style = XObjectStyle(self.meta)
    
    def drawBBox(self,box):
        boxCoord = box[0:4].astype(int)
        label = self.labelMap[int(box[-1])].upper()
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