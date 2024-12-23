import sys
from argparse import ArgumentParser
#
import logging
#
from dev import createFastAPIserverInstance
from dev.core.auth import verify_key,verify_token
from dev.core.config import load_config
from dev.schema.common import BaseResponse
from dev.schema.exception import ServiceStatusError,CudaOutOfMemoryError,Request,JSONResponse
#
from fastapi import Depends
import uvicorn



def getParser():
    parser = ArgumentParser()
    parser.add_argument('mode',type=str,default='dev')
    return parser.parse_args()


def run():
    args = getParser()
    cfg = load_config(args)
    ##
    if not cfg:
        print(f'No Env File for Mode[{args.mode}]')
        sys.exit(0)
    ##
    logger = logging.getLogger('Main')
    handler = logging.StreamHandler()
    formatter = logging.Formatter(cfg.logger_formatter)
    handler.setFormatter(formatter)
    logger.addHandler(handler) 
    ##
    return createFastAPIserverInstance(cfg),cfg

app,cfg = run()

@app.get("/",tags=["Info"],)
def getInfo(dependencies=[Depends(verify_token),Depends(verify_key)]):
    return BaseResponse(data={'title':cfg.meta.title,'version':cfg.meta.version})

@app.exception_handler(ServiceStatusError)
async def service_status_handler(request:Request,exc:ServiceStatusError):
    return JSONResponse(status_code=500,content=BaseResponse(is_ok=False,code=500,message=exc.value).model_dump())

@app.exception_handler(CudaOutOfMemoryError)
async def cuda_error_handler(request:Request,exc:CudaOutOfMemoryError):
    return JSONResponse(status_code=500,content=BaseResponse(is_ok=False,code=500,message=exc.value).model_dump())

if __name__ == "__main__":
    uvicorn.run('main:app',host=cfg.host,port=cfg.port,reload=cfg.with_reload,workers=cfg.workers)
