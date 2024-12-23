import re,os
from typing import Literal
from urllib.parse import unquote
##
from dev.schema.file import IFile,IStreamable,IImageTypeList,IVideoTypeList,IDataType
from dev.schema.model import IDetectRequest,IDetect,ITrainRequest,ITrain
##
from dev.util.file import makeAsset

EX = r"(?:(https?|rtsp):\/\/)([^\/]+)\/([^?]+)([^\s#]*)"

def checkURL(url,data_type):
    url = unquote(url)
    protocol,host_url,domain,param = re.findall(EX,url)[0]
    sub_domain = "/".join(domain.split("/")[:-1]) if domain.find("/") != -1 else domain
    file = domain.split("/")[-1] if len(domain.split("/")) and domain.find("/") != -1 else None
    if file is not None:
        filename,fileExt = checkFile(file)
        iFile = IFile(src=url,name=filename,type=data_type,ext=fileExt)
        if iFile.ext in IImageTypeList:
            iFile.extType = 'image'
        elif iFile.ext in IVideoTypeList:
            iFile.extType = 'video'
        else:
            iFile.extType = 'zip'
        return iFile
            
    if isRTSP(protocol):
        iStreamable = IStreamable(src=url,protocol=protocol)
        return iStreamable
        
def checkFile(file):
    if file is None: raise ValueError("No file")
    filename,fileExt = os.path.splitext(file)
    return filename,fileExt

def isRTSP(protocol:str):
    return protocol.startswith('rtsp')

def checkData(data,mode:IDataType):
    if not isinstance(data,IFile) or (isinstance(data,IFile) and data.type != mode):
        return False
    return True

def checkDetectRequest(request:IDetectRequest):
    ## check model info
    target_model_path,target_src_path,mode = '','','image',
    data = checkURL(request.d_detect_model_url,'model')
    if not checkData(data,'model'):
        raise Exception
    else:
        src_path,target_model_path = makeAsset('model',request.d_detect_model_url,{'category':request.d_base_model_category,'model_id':request.d_base_model_id,'project_id':request.project_id},None,None)
    ## check target url
    data = checkURL(request.target_url,'input')
    if isinstance(data,IFile):
        assert data.extType != 'zip',"No Supported Detect Format."
        src_path,target_src_path = makeAsset('data_in',request.target_url,{'project_id':request.project_id,'detect_model_id':request.d_detect_model_id,'ext':data.ext},None,{'history_id':request.d_detect_history_id})
        if data.extType == 'video':
            mode = 'video'
        else:
            mode = 'image'
    elif isinstance(data,IStreamable):
        assert data.protocol == 'rtsp',"No Supported Detect Format."
        mode = 'stream'
    else:
        raise Exception
    ##
    detect_info = IDetect(**{**request.model_dump(),'model_path':target_model_path,'target_path':target_src_path,'source_type':data.extType})
    return detect_info,mode

def checkTrainRequest(request:ITrainRequest):
    ## check model info
    target_model_path,target_dataset_path= '',''
    model_data = checkURL(request.model_info.storage_url,'model')
    dataset_data = checkURL(request.dataset_info.storage_url,'dataset')
    if not checkData(model_data,'model') or not checkData(dataset_data,'dataset'):
        raise Exception
    else:
        src_model_path,target_model_path = makeAsset('model',request.model_info.storage_url,{'category':request.model_info.category,'model_id':request.model_info.d_model_id,'project_id':request.project_id},None,None)
        src_dataset_path,target_dataset_path = makeAsset('dataset',request.dataset_info.storage_url,{'project_id':request.project_id,'dataset_id':request.dataset_info.d_dataset_id},None,None)
    train_info = ITrain(**{**request.model_dump(),'model_path':target_model_path,'dataset_path':target_dataset_path})
    return train_info