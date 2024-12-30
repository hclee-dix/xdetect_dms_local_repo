from fastapi import APIRouter
#
from dev.service.status import getStatus
from dev.crud.s_server import readSystemServer,updateSystemServer
from dev.schema.common import BaseResponse
from dev.util.wrapper import commonJsonResponse
#
router = APIRouter()

@router.get('/status',tags=['Info'],response_model=BaseResponse)
def health_check():
    return commonJsonResponse(getStatus())
    
# @router.get('/serverStatus',tags=['Info'],response_model=BaseResponse)
# def getServerStatus():
#     s_server_id = 'UCqlZgXlNmSSRAK3ivBq'
#     response = readSystemServer(s_server_id)
#     return BaseResponse(data=response)

# @router.post('/updateServerStatus',tags=['Info'],response_model=BaseResponse)
# def updateServerStatus():
#     s_server_id = 'UCqlZgXlNmSSRAK3ivBq'
#     status = getStatus(merge_with_avg=True)
#     response = updateSystemServer(s_server_id,**{'cpu':1})
#     return BaseResponse(data=response)
