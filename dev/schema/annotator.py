from typing import Tuple,Literal,Optional,Any,Union
#
from pydantic import BaseModel,Field, computed_field

class IAnnotatorMeta(BaseModel):
    width:int = Field(default=1920)
    height:int = Field(default=1080)
    # size:Tuple[int,int] = Field(default=(1920,1080))
    fps:int = Field(default=30)
    frame_count:int = Field(default=0)
    scale:Tuple[float,float] = Field(default=(0.5,0.5))
    # rescale:Tuple[float,float] = Field(default=(0.5,0.5))
    
    # def model_post_init(self, __context: Any) -> None:
    #     self.size = (self.width,self.height)
    #     self.rescale = (1/self.scale[0],1/self.scale[1])
    #     return super().model_post_init(__context)
    
    @computed_field
    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)
    
    @computed_field
    @property
    def rescale(self) -> Tuple[float, float]:
        return (1/self.scale[0], 1/self.scale[1])