from typing import Literal,Optional,Union,List
#
from timeit import default_timer as dt
import numpy as np
#
from dev.schema.perf import IPerfTask,IPerfStat,IPerfBase


class Performance:
    def __init__(self):
        self.task:IPerfTask = IPerfTask()

    def start(self,task_name:str,stat_name:Literal['pre','inf','post']):
        if task_name not in self.task.data: return
        task = self.task.data[task_name]
        getattr(task,stat_name).start = dt()

    def end(self,task_name:str,stat_name:Literal['pre','inf','post']):
        if task_name not in self.task.data: return
        task = self.task.data[task_name]
        getattr(task,stat_name).end = dt()

    def addTask(self,task_name:str,data:Optional[IPerfStat]=None,precision:int=-1):
        if task_name in self.task.data: return
        if data is None:
            self.task.data[task_name] = IPerfStat(precision=precision)
            return
        self.task.data[task_name] = data

    def setTask(self,task_name:str,data:Union[List,IPerfStat],unit:Literal['ms','s']):
        if isinstance(data,list):
            self.task.data[task_name].update_raw(data,unit)
            return
        self.task.data[task_name] = data

    def resetTask(self,task_name:str,precision:int=-1):
        if task_name not in self.task.data: return
        precision= self.task.data[task_name].precision if self.task.data[task_name].hasPrecision else precision
        self.task.data[task_name] = IPerfStat(precision=precision)

    def delTask(self,task_name:str):
        if task_name not in self.task.data: return
        del self.task.data[task_name]

    def getTask(self,task_name:str):
        if task_name in self.task.data:
            return self.task.data[task_name].print()
        return IPerfStat()

    def getTaskAll(self,dim:Literal['task','stat','all','none']):
        if dim=='none':
            return self.task
        if dim=='task':
            return self.mergeTask()
        if dim=='stat':
            return self.mergeStat()
        if dim=='all':
            return self.mergeAll()
        
    def mergeTask(self):
        return {k:v.sum for k,v in self.task.data.items()}
    
    def mergeStat(self):
        stat = []
        for k,v in self.task.data.items():
            stat.append(v.toList)
        a = np.sum(np.array(stat),axis=0,dtype=np.float_)
        return {'pre':a[0],'inf':a[1],'post':a[2]}
    
    def mergeAll(self):
        return sum(self.mergeStat().values())