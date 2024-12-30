from typing import Tuple,Dict,Any
from pydantic import BaseModel,RootModel,Field
from PIL import Image


class IStyle(BaseModel):
    width:int = Field(default=0)
    height:int = Field(default=0)
    top:int = Field(default=0)
    left:int = Field(default=0)
    bottom:int = Field(default=0)
    right:int = Field(default=0)
    margin:Tuple[int,int,int,int] = Field(default=(0,0,0,0))
    margin_top:int = Field(default=0)
    margin_left:int = Field(default=0)
    margin_bottom:int = Field(default=0)
    margin_right:int = Field(default=0)
    padding:Tuple[int,int,int,int] = Field(default=(0,0,0,0))
    padding_top:int = Field(default=0)
    padding_left:int = Field(default=0)
    padding_bottom:int = Field(default=0)
    padding_right:int = Field(default=0)
    background_color:str = Field(default="#FFFFFFFF")
    border_width:int = Field(default=0)
    border_color:str = Field(default="#FFFFFFFF")
    border_radius:int = Field(default=0)

class IFontStyle(BaseModel):
    font_family:str = Field(default="")
    font_size:int = Field(default=0)
    font_color:str = Field(default="#FFFFFFFF")

class IIconStyle(BaseModel):
    width:int = Field(default=0)
    height:int = Field(default=0)
    color:Tuple[int,int,int] = Field(default=(255,255,255))
    
class IIcon(BaseModel):
    src: Dict[str,Any]
    