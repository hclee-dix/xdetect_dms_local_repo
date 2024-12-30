from fastapi import Request,HTTPException
from fastapi.responses import JSONResponse
from torch.cuda import OutOfMemoryError

class ServiceStatusError(Exception):
    def __init__(self,value):
        self.value = value
    
    def __str__(self):
        return self.value
    
class CudaOutOfMemoryError(OutOfMemoryError):
    def __init__(self,value):
        self.value = value
    
    def __str__(self):
        return self.value