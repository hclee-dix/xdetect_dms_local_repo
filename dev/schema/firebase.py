"""
Firebase Database Document
"""
from typing import List,Union,Literal,Optional
from pydantic import BaseModel,Field
from datetime import datetime,timezone
#region common
class IBaseField(BaseModel):
    is_visible:bool = Field(default=True)
    create_at:datetime = Field(default=datetime.now(timezone.utc))
    update_at:datetime = Field(default=datetime.now(timezone.utc))

#region system
class ISystemServerDisk(BaseModel):
    free:Union[int,float]
    percent:Union[int,float]
    total:Union[int,float]
    used:Union[int,float]

class ISystemServerMemory(BaseModel):
    available:Union[int,float]
    percent:Union[int,float]
    total:Union[int,float]
    used:Union[int,float]

class ISystemServer(IBaseField):
    id:str
    default_host_name:str
    port:int
    status:str
    msg:str
    project_list:List[str]
    cpu:Union[int,float]
    memory:ISystemServerMemory
    disk:ISystemServerDisk
#endregion
#region detect_history
class IDetectHistoryResolution(BaseModel):
    width:int
    height:int

class IDetectHistoryResult(BaseModel):
    index: int
    name: str
    conf: float

class IDetectHistory(IBaseField):
    org_id: str
    project_id:str
    d_detect_model_id:str
    source_from:Literal['WEB','API']
    source_type:Literal['image','video','zip','url']
    storage_url:Optional[str] = Field(default=None)
    accuracy:float
    inference:float
    resolution:IDetectHistoryResolution
    fps:float
    result_list:List[IDetectHistoryResult]
#endregion
#region train_history
class ITrainHistory(IBaseField):
    org_id: str
    project_id:str
    d_train_model_id:str
    epoch: int
    accuracy: float = Field(default=0.0)
    train_loss: float  = Field(default=1.0)
    validation_loss: float  = Field(default=1.0)

#endregion
#region train_model
#endregion