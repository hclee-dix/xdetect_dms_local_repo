from dev.ai.xBase.callback import XCallback
from dev.ai.xBase.label import FALLING
#
from dev.crud.d_detect_history import updateDetectHistory
#
from dev.schema.model import IDetect,ITrain
from dev.schema.firebase import IDetectHistory,IDetectHistoryResolution,IDetectHistoryResult
#
from dev.util.file import makeAsset
from dev.util.firebase import uploadFile
#
from dev.util.ops import getFightResultFromSingle,getFightResultFromTotal
#

class XFightDetectorCallback(XCallback):
    def __init__(self,request,labelMap):
        super().__init__()
        self.request:IDetect = request
        self.labelMap = labelMap
        self.ext = '.mp4' if self.request.source_type == 'video' else '.jpg'
        self.fps = 30 if self.request.source_type == 'video' else 0
    
    def onPredictStart(self):
        print("predict start")
        
    def onPredictEnd(self,imageList,perfList,resultList):
        h,w,c = imageList[0].shape
        ##
        src_path,dst_path = makeAsset('data_out',imageList,{'project_id':self.request.project_id,'detect_model_id':self.request.d_detect_model_id,'ext':self.ext,'fps':self.fps},None,{'history_id':self.request.d_detect_history_id,"format":""})
        storage_url = uploadFile(src_path,dst_path)
        ##
        result,accuracy,perf = getFightResultFromTotal(perfList,resultList)
        history = IDetectHistory(
            org_id=self.request.organization_id,
            project_id=self.request.project_id,
            d_detect_model_id=self.request.d_detect_model_id,
            source_from="WEB",
            storage_url=storage_url,
            source_type='video',
            resolution=IDetectHistoryResolution(width=w,height=h),
            inference=perf,
            fps=1/perf,
            accuracy=float(accuracy),
            result_list=[IDetectHistoryResult(index=int(result[0]),conf=result[1],name=self.labelMap[int(result[0])])]
        )
        print(history)
        updateDetectHistory(history,self.request.d_detect_history_id)
        print("predict end")
        
    def onPredictEpochStart(self):
        ...
    
    def onPredictEpochEnd(self,image,perf,result,frame_num=None):
        h,w,c = image.shape
        ##
        frame_num = "" if not frame_num else frame_num
        hid = self.request.d_detect_history_id if not frame_num else f"{self.request.d_detect_history_id}_{frame_num}"
        src_path,dst_path = makeAsset('data_out',image,{'project_id':self.request.project_id,'detect_model_id':self.request.d_detect_model_id,'ext':'.jpg','fps':0},None,{'history_id':hid,"format":""})
        storage_url = uploadFile(src_path,dst_path)
        totalResult,accuracy = getFightResultFromSingle(result)
        history = IDetectHistory(
            org_id=self.request.organization_id,
            project_id=self.request.project_id,
            d_detect_model_id=self.request.d_detect_model_id,
            source_from="WEB",
            storage_url=storage_url,
            source_type='image',
            resolution=IDetectHistoryResolution(width=w,height=h),
            inference=perf,
            fps=1/perf,
            accuracy=float(accuracy),
            result_list=[IDetectHistoryResult(index=int(totalResult[0]),conf=totalResult[1],name=self.labelMap[int(totalResult[0])])]
        )
        print(history)
        updateDetectHistory(history,hid)
        return totalResult


class XTestCallback(XCallback):
    def __init__(self,labelMap):
        self.labelMap = labelMap
        super().__init__()
    
    def onPredictStart(self):
        print("predict start")
        
    def onPredictEnd(self,imageList,perfList,resultList):
        h,w,c = imageList[0].shape
        ##
        src_path,dst_path = makeAsset('data_out',imageList,{'project_id':'PID','detect_model_id':'MID','ext':'.mp4','fps':30},None,{'history_id':'HID',"format":""},True)
        ##
        result,accuracy,perf = getFightResultFromTotal(perfList,resultList)
        history = IDetectHistory(
            org_id='OID',
            project_id='PID',
            d_detect_model_id='MID',
            source_from="WEB",
            storage_url='',
            source_type='video',
            resolution=IDetectHistoryResolution(width=w,height=h),
            inference=perf,
            fps=1/perf,
            accuracy=float(accuracy),
            result_list=[IDetectHistoryResult(index=int(result[0]),conf=result[1],name=self.labelMap[int(result[0])])]
        )
        print(history)
        # print("predict end")
        
    def onPredictEpochStart(self):
        ...
    
    def onPredictEpochEnd(self,image,perf,result,frame_num=None):
        h,w,c = image.shape
        ##
        frame_num = "" if not frame_num else frame_num
        src_path,dst_path = makeAsset('data_out',image,{'project_id':'PID','detect_model_id':'MID','ext':'.jpg','fps':0},None,{'history_id':'HID'+f'_{frame_num}' if frame_num else 'HID',"format":""},True)
        ##
        totalResult,accuracy = getFightResultFromSingle(result)
        history = IDetectHistory(
            org_id='OID',
            project_id='PID',
            d_detect_model_id='MID',
            source_from="WEB",
            storage_url='',
            source_type='image',
            resolution=IDetectHistoryResolution(width=w,height=h),
            inference=perf,
            fps=1/perf,
            accuracy=float(accuracy),
            result_list=[IDetectHistoryResult(index=int(totalResult[0]),conf=totalResult[1],name=self.labelMap[int(totalResult[0])])]
        )
        return history