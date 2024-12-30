from typing import Literal,Optional,Any,Union
#
from pydantic import BaseModel,Field

class BaseResponse(BaseModel):
    is_ok:bool= Field(default=True)
    code:Literal[200,400,401,403,404,500] = Field(default=200)
    message:Optional[str] = Field(default=None)
    data:Optional[Any] = Field(default=None)