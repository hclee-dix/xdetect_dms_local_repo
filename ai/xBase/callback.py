

class XCallback:
    def __init__(self):...
    ## detect
    def onPredictStart(self): raise NotImplementedError
    def onPredictEpochStart(self): raise NotImplementedError
    def onPredictEpochEnd(self): raise NotImplementedError
    def onPredictEnd(self): raise NotImplementedError
    ## train
    def onPrepareStart(self):raise NotImplementedError
    def onPrepareEnd(self):raise NotImplementedError
    def onTrainStart(self):raise NotImplementedError
    def onTrainEpochStart(self):raise NotImplementedError
    def onTrainBatchStart(self):raise NotImplementedError
    def onTrainBatchEnd(self):raise NotImplementedError
    def onTrainEpochEnd(self):raise NotImplementedError
    def onTrainModelSave(self):raise NotImplementedError
    def onFitEpochEnd(self):raise NotImplementedError
    def onTrainEnd(self):raise NotImplementedError
    ## validation
    def onValidateStart(self):raise NotImplementedError
    def onValidateBatchStart(self):raise NotImplementedError
    def onValidateBatchEnd(self):raise NotImplementedError
    def onValidateEnd(self):raise NotImplementedError
    