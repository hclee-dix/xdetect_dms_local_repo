from typing import Literal,Optional,Any,Union,Tuple,get_args
#
from pydantic import BaseModel,Field

IVideoType = Literal['.mp4','.avi','.webm']
IImageType = Literal['.jpeg','.jpg','.png','webp']
IZipType = Literal['.zip']
IDataType = Literal['model','dataset','input'] 
IExtType = Literal['image','video','zip','url']
IModeType = Literal['image','video','stream']
IVideoTypeList:Tuple[IVideoType,...] = get_args(IVideoType)
IImageTypeList:Tuple[IImageType,...] = get_args(IImageType)
IZipTypeList:Tuple[IZipType,...] = get_args(IZipType)

class IFile(BaseModel):
    src:str
    name:str
    type:IDataType = Field(default=Literal['model'])
    extType:IExtType = Field(default=Literal['zip'])
    ext:Union[IImageType,IVideoType,IZipType]
    
class IStreamable(BaseModel):
    src:str
    protocol:Literal['rtsp','http']
    extType:IExtType = Field(default=Literal['url'])