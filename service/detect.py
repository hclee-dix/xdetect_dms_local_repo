import asyncio
from typing import Literal
#
from dev.schema.model import DetectType
from dev.schema.file import IModeType
# detect
from dev.ai.xObject.detect import createXObjectDetector
from dev.ai.xAttribute.detect import createXAttrDetector
from dev.ai.xFalling.detect import createXFallingDetector
from dev.ai.xSmoking.detect import createXSmokingDetector
from dev.ai.xFight.detect import createXFightDetector
from dev.ai.xFire.detect import createXFireDetector
# test
from dev.ai.xObject.test import createXObjectDetectorTest
from dev.ai.xFalling.test import createXFallingDetectorTest
from dev.ai.xFight.test import createXFightDetectorTest
#
from dev.schema.model import IDetect
#
from dev.util.wrapper import commonResponse
#
def init_detect(request:IDetect,mode:IModeType):
    ##
    if request.d_model_type == 'animal detect':
        net = createXObjectDetector(request,mode,[14,15,16,17,18,19,20,21,22,23])
    elif request.d_model_type == 'vehicle detect':
        net = createXObjectDetector(request,mode,[1,2,3,4,5,6,7,8])
    elif request.d_model_type == 'food detect':
        net = createXObjectDetector(request,mode,[46,47,48,49,50,51,52,53,54,55])
    elif request.d_model_type == 'falling detect':
        net = createXFallingDetector(request)
    elif request.d_model_type == 'attribute detect':
        net = createXAttrDetector(request)
    elif request.d_model_type == 'smoking detect':
        net = createXSmokingDetector(request)
    elif request.d_model_type == 'fight detect':
        net = createXFightDetector(request)
    elif request.d_model_type == 'fire detect':
        net = createXFireDetector(request)
    else:
        raise ValueError("No Supported Type")
    ##
    if mode == 'image':
        return net.getImage()
    elif mode == 'video':
        return net.getVideo()
    elif mode == 'stream':
        return net.getStream()
    else:
        raise ValueError("No Supported Mode")

def init_test(type:DetectType,mode:IModeType):
    model_path,path,config = '','',''
    if type in ['animal detect', 'food detect', 'vehicle detect']:
        model_path = '/data2/xdetect_storage/dev/test/model/yolo11m.pt'
        if mode == 'video' or mode == 'stream':
            path = './dev/core/asset/video/zebra.mp4'
        elif mode == 'image':
            path = './dev/core/asset/image/bird.jpg'
    elif type == 'falling detect':
        model_path = '/data2/xdetect_storage/temp/model/FALLING'
        config = 'dev/ai/common/config/base_falling_config.yaml'
        if mode == 'video' or mode == 'stream':
            path = './dev/core/asset/video/faint1.mp4'
        else:
            raise ValueError("No Supported detect type")
    elif type == 'fight detect':
        model_path = '/data2/xdetect_storage/temp/model/FIGHT'
        config = 'dev/ai/common/config/base_fight_config.yaml'
        if mode == 'video' or mode == 'stream':
            path = './dev/core/asset/video/fight1.mp4'
        else:
            raise ValueError("No Supported detect type")
    if type == 'animal detect':
        net = createXObjectDetectorTest(model_path,path,mode,[14,15,16,17,18,19,20,21,22,23])
    elif type == 'food detect':
        net = createXObjectDetectorTest(model_path,path,mode,[46,47,48,49,50,51,52,53,54,55])
    elif type == 'vehicle detect':
        net = createXObjectDetectorTest(model_path,path,mode,[1,2,3,4,5,6,7,8])
    elif type == 'falling detect':
        net = createXFallingDetectorTest(model_path,path,mode,config)
    elif type == 'fight detect':
        net = createXFightDetectorTest(model_path,path,mode,config)
    else:
        raise ValueError("No Supported Type")
    
    if mode == 'image':
        return net.getImage()
    elif mode == 'video':
        return net.getVideo()
    elif mode == 'stream':
        return net.getStream()
    else:
        raise ValueError('No Supported Mode')