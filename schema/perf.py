from typing import Dict,List,Optional,Union
from typing_extensions import TypedDict
#
from pydantic import BaseModel,Field,computed_field
#
from dev.util.util import listDiv

class IPerfBase(BaseModel):
    start:float = Field(0.0)
    end:float = Field(0.0)
    raw:float = Field(-.1)
    precision:int = Field(-1)

    @computed_field
    @property
    def value(self) -> float:
        v = self.end - self.start if not self.hasRaw else self.raw
        return v if not self.hasPrecision else round(v,self.precision)

    
    @computed_field
    @property
    def hasRaw(self)->bool:
        return self.raw >=0
    
    @computed_field
    @property
    def hasPrecision(self)->bool:
        return self.precision>=0

class IPerfStat(BaseModel):
    pre:IPerfBase = Field(default_factory=IPerfBase)
    inf:IPerfBase = Field(default_factory=IPerfBase)
    post:IPerfBase = Field(default_factory=IPerfBase)
    precision:int = Field(-1)

    def model_post_init(self, __context):
        self.pre.precision = self.precision
        self.inf.precision = self.precision
        self.post.precision = self.precision
        return super().model_post_init(__context)

    def update_raw(self,data:List[float],unit):
        data = listDiv(data,1000) if unit == 'ms' else data
        self.pre.raw,self.inf.raw,self.post.raw = data[:3]
    
    def update_precision(self,precision):
        self.precision = precision
        self.pre.precision = precision
        self.inf.precision = precision
        self.post.precision = precision
    
    def print(self):
        return f"pre:{self.pre.value:4f}|inf:{self.inf.value:4f}|post:{self.post.value:4f}"

    @computed_field
    @property
    def hasPrecision(self)->bool:
        return self.precision>=0

    @computed_field
    @property
    def sum(self) -> float:
        if self.precision >= 0:
            return round(self.pre.value + self.inf.value + self.post.value,self.precision)
        return self.pre.value + self.inf.value + self.post.value
    
    @computed_field
    @property
    def avg(self) -> float:
        if self.precision >= 0:
            return round(self.sum/3,self.precision)
        return self.sum/3
    
    @computed_field
    @property
    def toList(self) -> List[float]:
        return [self.pre.value , self.inf.value , self.post.value]


class IPerfTask(BaseModel):
    data:Dict[str,IPerfStat] = Field(default_factory=dict)

    def __str__(self):
        return '='.join(f'[{k}#{v.print()}]' for k,v in self.data.items())