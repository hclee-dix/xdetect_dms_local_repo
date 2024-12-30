"""
Api Request Model
"""
from typing import Optional,Literal,List
from pydantic import BaseModel,ConfigDict,Field
from dev.schema.file import IExtType

#region Common
class ITime(BaseModel):
    _seconds: int
    _nanoseconds: int

#region Train
class ITrainRequestModelInfo(BaseModel):
    category: Literal['base','trained']
    class_list: List[str]
    d_model_id:str
    storage_url:str

class ITrainRequestDatasetInfo(BaseModel):
    d_dataset_id: str
    data_count: Optional[int] = Field(default=0)
    fps:Optional[int] = Field(default=0)
    storage_url: str
    test: int
    title: str
    total_duration: Optional[str] =  Field(default="")
    total_size: int
    train: int
    valid: int

class ITrainRequestTrainInfo(BaseModel):
    start_at: ITime
    type: str
    epoch: int
    status: Literal['READY','TRAIN','DONE']
    

class ITrainRequest(BaseModel):
    model_config  = ConfigDict(protected_namespaces=())
    create_at: ITime
    d_train_model_id:str
    dataset_info: ITrainRequestDatasetInfo
    description: str
    is_visible: bool
    model_info: ITrainRequestModelInfo
    org_id: str
    project_id: str
    tag_list: List[str]
    title: str
    train_info: ITrainRequestTrainInfo
    update_at: ITime
    user_id: str

class ITrain(ITrainRequest):
    model_config  = ConfigDict(protected_namespaces=())
    d_train_history_id: str = Field(default='')
    epoch:int = Field(default=100)
    model_path: str
    dataset_path: str
    
#region Detect
DetectType = Literal['attribute detect','fire detect','falling detect','smoke detect','fight detect','food detect','vehicle detect','animal detect']
CategoryType = Literal['base','trained']
class IDetectRequest(BaseModel):
    organization_id: str
    project_id: str
    d_detect_model_id: str
    d_detect_model_url: str
    d_base_model_category:CategoryType
    d_base_model_id: str
    d_detect_history_id: str
    d_model_type: DetectType
    target_url: str
    
class IDetect(IDetectRequest):
    model_config  = ConfigDict(protected_namespaces=())
    model_path: str
    target_path: str
    source_type:IExtType