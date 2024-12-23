
from dev.ai.xBase.perf import Performance

class XDetector():
    def __init__(self):
        self.performance = Performance()
    def init_model(self): raise NotImplementedError
    def init_callback(self): raise NotImplementedError
    def init_source(self,mode:bool):raise NotImplementedError
    def init_video(self):raise NotImplementedError
    def init_image(self):raise NotImplementedError
    def predict(self):raise NotImplementedError
    def predict_image(self):raise NotImplementedError
    def predict_video(self):raise NotImplementedError
    def predict_stream(self):raise NotImplementedError
    def visualize(self):raise NotImplementedError
    def getStream(self):raise NotImplementedError
    def getVideo(self):raise NotImplementedError
    def getImage(self):raise NotImplementedError