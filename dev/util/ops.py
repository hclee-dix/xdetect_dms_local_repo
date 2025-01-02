import numpy as np


## falling result
def getFallingResultFromSingle(result):
    # [N,22,3]
    # :,3,: falling result [track_id, falling_class, conf] [0: None, 1: Falling]
    # :,4,: detect result  [track_id, detect_class, conf] [0: Human]
    result = result[:,3,:].reshape(-1,3) ## [N,1,3] -> [N,3]
    accuracy = np.max(result[:,2],axis=0)
    return result,accuracy

def getFallingResultFromTotal(perfList,resultList):
    perf = np.average(perfList)
    # [F,N,22,3]
    # :,3,: falling result [track_id, falling_class, conf] [0: None, 1: Falling]
    # :,4,: detect result  [track_id, detect_class, conf] [0: Human]
    max_m = max(len(group) for group in resultList)
    padded_data = np.zeros((len(resultList),max_m,22,3))
    for i,result in enumerate(resultList):
        padded_data[i,:len(result),:] = result
    unique_labels = np.unique(padded_data[:,:,4,0][padded_data[:,:,4,-1]>0]) ## unique detect_class for conf > 0
    if len(unique_labels) == 0: ## No detect among Total frame
        return [],0,perf
    avgResult = np.zeros((len(unique_labels),22,3))
    for i, value in enumerate(unique_labels):
        mask = padded_data[:,:,4,0] == value ## mask detect_class
        selected_data = padded_data[mask].reshape(-1,22,3)
        avgResult[i] = selected_data.max(axis=0)
    # accuracy = np.average(avgResult[:,3,2],axis=0) ## average for falling_class
    accuracy = np.max(avgResult[:,3,2],axis=0) ## top for falling_class
    return avgResult,accuracy,perf

## fight result
def getFightResultFromSingleFrame(result):
    # [2,] [classes, scores]
    return result,result[1]

def getFightResultFromTotalFrame(perfList,resultList):
    perf = np.average(perfList)
    result = np.array(resultList)
    # [F,2] [classes, scores]
    classes = np.max(result[:,0],axis=0) ## has fight
    result = result[result[:,0]>0] ## filter fight
    accuracy = np.average(result[:,1],axis=0) ## average accuracy
    return [classes,accuracy],accuracy,perf

## object result
def getTopResultFromSingleFrame(result):
    # [N,6] [x,y,x,y,conf,cls]
    unique_labels = np.unique(result[:,-1]) # cls기준
    topResult = np.array([
        result[result[:,-1]==key][np.argmax(result[result[:,-1]==key][:,4])] for key in unique_labels
    ]) # cls기준 topResult [M,6] M: classes
    accuracy = np.average(topResult[:,4],axis=0)
    return topResult,accuracy

def getAvgResultFromTotalFrame(perfList,resultList):
    perf = np.average(perfList)
    max_m = max(len(group) for group in resultList)
    padded_data = np.zeros((len(resultList),max_m,6))
    for i,result in enumerate(resultList):
        padded_data[i,:len(result),:] = result
    unique_labels = np.unique(padded_data[:,:,-1][padded_data[:,:,-2]>0]) ## unique cls for conf > 0
    if len(unique_labels) == 0: ## No detect among Total frame
        return [],0,perf
    avgResult = np.zeros((len(unique_labels),6))
    for i, value in enumerate(unique_labels):
        mask = padded_data[:,:,-1] == value
        selected_data = padded_data[mask].reshape(-1,6)
        avgResult[i] = selected_data.mean(axis=0)
    accuracy = np.average(avgResult[:,4],axis=0)
    return avgResult,accuracy,perf