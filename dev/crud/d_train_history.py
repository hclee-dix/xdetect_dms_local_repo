from dev.core.firebase import firebase_app
#
from dev.schema.firebase import ITrainHistory
from dev.schema.validate import validate_data
#
COLLECTION_NAME = 'd_train_history'

@validate_data(ITrainHistory)
def createTrainHistory(data:ITrainHistory):
    _,snapshot = firebase_app.db.collection(COLLECTION_NAME).add(data.model_dump())
    return snapshot.id
    
@validate_data(ITrainHistory)
def updateTrainHistory(data:ITrainHistory,d_train_history_id):
    firebase_app.db.collection(COLLECTION_NAME).document(d_train_history_id).update(data.model_dump())
    