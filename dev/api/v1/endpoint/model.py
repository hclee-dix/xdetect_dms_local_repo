from fastapi import APIRouter,BackgroundTasks
from fastapi.responses import StreamingResponse
#
from typing import Literal
#
from dev.service.detect import init_detect,init_test
from dev.service.train import init_train
from dev.service.status import checkStatus
#
from dev.crud.d_train_history import createTrainHistory,ITrainHistory
#
from dev.schema.common import BaseResponse
from dev.schema.model import ITrainRequest,IDetectRequest,DetectType
from dev.schema.file import IModeType
from dev.schema.util import checkDetectRequest,checkTrainRequest
#
from dev.util.wrapper import commonJsonResponse
#
router = APIRouter()

@router.get('/detect/test',tags=['Model'])
def test_detect(type:DetectType,mode:IModeType,background_task:BackgroundTasks):
    if mode == 'stream':
        net = init_test(type,mode)
        return StreamingResponse(net,media_type="multipart/x-mixed-replace; boundary=frame") # type: ignore
    else:
        background_task.add_task(init_test,type,mode)
        return BaseResponse()

@router.post('/detect',tags=['Model'])
def run_detect(request:IDetectRequest,background_task:BackgroundTasks):
    checkStatus()
    data,mode = checkDetectRequest(request)
    if mode == 'stream':
        net = init_detect(data,mode)
        return StreamingResponse(net,media_type="multipart/x-mixed-replace; boundary=frame") # type: ignore
    elif mode in ['video','image']:
        background_task.add_task(init_detect,data,mode)
        # init_detect(data,isStream)
        return commonJsonResponse(BaseResponse(data={'historyId':request.d_detect_history_id}))
    else:
        return BaseResponse()

@router.post('/train',tags=["Model"])
def run_train(request:ITrainRequest,background_task:BackgroundTasks):
    checkStatus()
    train_info = checkTrainRequest(request)
    ## AddTrainHistory
    d_train_history_id =createTrainHistory(ITrainHistory(org_id=train_info.org_id,project_id=train_info.project_id,d_train_model_id=train_info.d_train_model_id,epoch=train_info.epoch))
    train_info.d_train_history_id = d_train_history_id
    ##
    background_task.add_task(init_train,train_info)
    return BaseResponse()
