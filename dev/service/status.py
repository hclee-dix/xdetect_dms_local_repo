import psutil
import GPUtil
from dev.util.wrapper import commonResponse
from dev.util.util import getUnit
from torch.cuda import memory_allocated,memory_usage,memory_reserved,get_device_properties
from dev.schema.exception import ServiceStatusError

@commonResponse(with_result=True)
def getStatus(merge_with_avg=False):
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    gpuList = GPUtil.getGPUs()
    gpu = []
    for g in gpuList:
        gpu.append({
            'total':getUnit(g.memoryTotal,'GB'),
            'used':getUnit(g.memoryUsed,'GB'),
            'percent':round(g.memoryUtil * 100,1)
        })
    if merge_with_avg:
        merge_gpu = {
            'total':sum(g['total'] for g in gpu)/len(gpu),
            'used':sum(g['used'] for g in gpu)/len(gpu),
            'percent':sum(g['percent'] for g in gpu)/len(gpu),
        }
        gpu = merge_gpu

    return {
        "cpu":cpu,
        "mem":{
            "total":getUnit(mem.total,'GB'),
            "available":getUnit(mem.available,'GB'),
            "used":getUnit(mem.used,'GB'),
            "percent":mem.percent
        },
        "disk":{
            "total":getUnit(disk.total,'GB'),
            "free":getUnit(disk.free,'GB'),
            "used":getUnit(disk.used,'GB'),
            "percent":disk.percent,
        },
        "gpu":gpu,
        "cuda":{
            "total":getUnit(get_device_properties(0).total_memory,'GB'),
            "allocated":getUnit(memory_allocated(),'GB'),
            "reserved":getUnit(memory_reserved(),'GB'),
            "used":getUnit(memory_usage(),'GB'),
            "percent":round(getUnit(memory_reserved(),'GB')/getUnit(get_device_properties(0).total_memory,'GB')*100,1)
        }
    }
    
def checkStatus():
    status = getStatus(merge_with_avg=True)
    if status.data is None or status.data['cpu'] >= 80 or status.data['mem']['percent'] >=80 or status.data['gpu']['percent'] >=50 or status.data['cuda']['percent'] >= 50:
        raise ServiceStatusError("Server is busy now")