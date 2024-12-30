class XCapture:
    def __init__(self):
        ...
    
    def init_video(self):
        raise NotImplementedError
    
    def resize(self):
        raise NotImplementedError
    
    def retrieve(self):
        raise NotImplementedError
    
    def __iter__(self):
        return self
    
    def __next__(self):
        raise NotImplementedError