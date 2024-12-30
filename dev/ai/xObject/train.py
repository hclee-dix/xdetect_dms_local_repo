import os
import glob
#
from dev.ai.xBase.train import XTrainer
from dev.ai.xObject.callback import XObjectTrainerCallback

#
from dev.schema.model import ITrain
#
from ultralytics import YOLO

class XObjectTrainer(XTrainer):
    def __init__(self,request:ITrain,override):
        self.request = request
        self.override = override        
        self.init_model()
        self.init_dataset()
        self.init_callback()
        
    def init_model(self):
        self.model_file = glob.glob(os.path.join(self.request.model_path,"**","*.pt"))[0]
        self.initObjectTrainer()
    
    def init_callback(self):
        self.callback = XObjectTrainerCallback(self.request)
        self.net.add_callback('on_train_start',self.callback.onTrainStart)
        self.net.add_callback('on_train_epoch_start',self.callback.onTrainEpochStart)
        self.net.add_callback('on_train_end',self.callback.onTrainEnd)

    def initObjectTrainer(self):
        self.net = YOLO(model=self.model_file,task='train')

    def init_dataset(self):
        self.dataset_file = glob.glob(os.path.join(self.request.dataset_path,"**","*.yaml"))[0]

    def train(self):
        self.net.train(data=self.dataset_file,**self.override)
    
        
def createXObjectTrainer(request:ITrain):
    override = dict(device='0',project=request.model_path,verbose=False,epoch=request.epoch)
    #
    trainer = XObjectTrainer(request,override)
    #
    return trainer
    
    # ## get model
    # base_path = '/data2/xdetect_storage/temp'
    # model_path = os.path.join(base_path, 'model/yolov8m.pt')
    # dataset_path = os.path.join(base_path,'dataset/Y9w0kkdwPkEn1F/data.yaml')
    # ## init args
    # overrides = dict(model=model_path,data=dataset_path,device='0',epochs=10,project=base_path)
    # ## instance
    # trainer = XYoloTrainer(overrides=overrides)
    # ## add callback
    
    # trainer.add_callback('on_pretrain_routine_start',cb.on_prepare_start)
    # trainer.add_callback('on_pretrain_routine_end',cb.on_prepare_end)
    # trainer.add_callback('on_train_start',cb.on_train_start)
    # trainer.add_callback('on_train_epoch_start',cb.on_train_epoch_start)
    # trainer.add_callback('on_train_batch_start',cb.on_train_batch_start)
    # trainer.add_callback('on_train_batch_end',cb.on_train_batch_end)
    # trainer.add_callback('on_train_epoch_end',cb.on_train_epoch_end)
    # trainer.add_callback('on_train_model_save',cb.on_train_model_save)
    # trainer.add_callback('on_fit_epoch_end',cb.on_fit_epoch_end)
    # trainer.add_callback('on_train_end',cb.on_train_end)
    # ## train
    # trainer.train()

    
    
    
    