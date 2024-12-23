#
from typing import Union,Literal,List
from timeit import default_timer as dt
##
import cv2

def t(name,show):
    def o(fn):
        def i(*args,**kwargs):
            s = dt()
            r = fn(*args,**kwargs)
            e = dt() - s
            if show: print(f"[{name}]#{e:.8f}")
            return r
        return i
    return o

def div(a,b):
    return a/b

def listDiv(data:List,value:int=1):
    assert value != 0, 'value is not equal to 0'
    return [div(v,value) for v in data]

def getUnit(value:Union[int,float],mode:Literal['GB','MB'],precision=2):
    if mode == 'GB':
        try:
            return round(value >> 30,precision) # type: ignore
        except TypeError:
            return round(value / 1024,precision)
    elif mode == 'MB':
        try:
            return round(value >> 20,precision) # type: ignore
        except TypeError:
            return round(value / 1024,precision)
    else:
        return value


def getEncodeFrame(frame):
    _,encoded = cv2.imencode('.jpg',frame)
    return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+bytearray(encoded) +b'\r\n'

