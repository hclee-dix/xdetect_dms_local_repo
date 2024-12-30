import cv2
#
from dev.ai.xBase.capture import XCapture
from dev.schema.annotator import IAnnotatorMeta
#
from dev.util.file import loadImage

class XFallingDetectorCapture(XCapture):
    def __init__(self,source_type,path,meta:IAnnotatorMeta):
        self.meta = meta
        self.source_type = source_type
        self.init_source(source_type,path)
        self.init_video(path)
    
    def init_source(self,source_type,path):
        if source_type == 'image':
            raise TypeError('Image is not supported source type')
        else:
            self.init_video(path)
    
    def init_video(self,path):
        self.cv2 = cv2.VideoCapture(path)
        self.meta.width = int(self.cv2.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.meta.height = int(self.cv2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.meta.fps = int(self.cv2.get(cv2.CAP_PROP_FPS))
        self.meta.frame_count = int(self.cv2.get(cv2.CAP_PROP_FRAME_COUNT))

    def resize(self,source,dsize,fx,fy):
        return cv2.resize(source,dsize,None,fx,fy)
    
    def retrieve(self):
        ret,frame = self.cv2.read()
        if not ret: raise StopIteration
        return frame
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.frame = self.retrieve() if self.source_type != 'image' else None
        im = self.resize(self.frame,dsize=(0,0),fx=self.meta.scale[0],fy=self.meta.scale[1])
        return self.frame,im