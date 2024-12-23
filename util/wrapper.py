import traceback
#
import asyncio
#
from typing import Callable,Union,Any
from typing import Coroutine
from types import SimpleNamespace
from typing import TypeVar
#
from dev.schema.common import BaseResponse
from dev.schema.exception import CudaOutOfMemoryError,OutOfMemoryError
#
from fastapi.responses import JSONResponse
#
T = TypeVar('T')

STATUS_CODE = SimpleNamespace(**{
    'OK':200,
    'BADREQUEST':400,
    'UNAUTHORIZED':401,
    'FORBIDDEN':403,
    "NOTFOUND":404,
    "INTERNALERROR":500,
})

def commonJsonResponse(response:BaseResponse):
    if isinstance(response.data,dict):
        return JSONResponse(content={**response.data,'is_ok':response.is_ok,'code':response.code,'message':response.message})
    
def commonResponse(with_result=False):
    def outer(f:Callable[...,T]):
        def inner(*args,**kwargs)->BaseResponse:
            try:
                result = f(*args,**kwargs)
                if with_result:
                    return BaseResponse(data=result)
                else:
                    return BaseResponse()
            except OutOfMemoryError as oome:
                raise CudaOutOfMemoryError("Out of Gpu Memory")
            except Exception as err:
                traceback.print_exc()
                raise Exception
        return inner
    return outer

def commonResponseAsync(with_result=False,early_return=False):
    def outer(f:Callable[...,Coroutine[Any,Any,T]])->Callable[...,Coroutine[Any,Any,BaseResponse]]:
        async def inner(*args,**kwargs)->BaseResponse:
            try:
                if early_return:
                    asyncio.create_task(f(*args,**kwargs)).add_done_callback(commonAsyncCallbackForEarlyReturn)
                    return BaseResponse()
                result = await f(*args,**kwargs)
                if with_result:
                    return BaseResponse(data=result)
                return BaseResponse(is_ok=True,code=STATUS_CODE.OK)
            except OutOfMemoryError as oome:
                raise CudaOutOfMemoryError("Out of Gpu Memory")
            except Exception as err:
                traceback.print_exc()
                raise Exception
        return inner
    return outer

def commonAsyncCallbackForEarlyReturn(task:asyncio.Task):
    try:
        result = task.result()
        print("Task Callback Result:",result)
    except Exception as err:
        traceback.print_exc()
