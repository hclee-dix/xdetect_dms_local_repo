
from dev.ai.xObject.train import createXObjectTrainer 
#
from dev.schema.model import ITrain
#
from dev.util.wrapper import commonResponse
#






@commonResponse(with_result=True)
def init_train(request:ITrain):
    if request.tag_list[-1] == 'object_detect':
        net = createXObjectTrainer(request)
    return net.train()
    # import ultralytics
    # base="/data2/xdetect_storage/temp"
    # model = ultralytics.YOLO('yolov8m.pt',task="detect")
    # model.train(data=base+"/dataset/Y9w0kkdwPkEn1F/data.yaml",epochs=2,imgsz=640,project=base+"/output")