from dev.core.firebase import firebase_app
from dev.crud import datetime,timezone
#
from dev.schema.firebase import IDetectHistory
from dev.schema.validate import validate_data
#
COLLECTION_NAME = 'd_detect_history'

@validate_data(IDetectHistory)
def createDetectHistory(data:IDetectHistory):
    firebase_app.db.collection(COLLECTION_NAME).add(data.model_dump())
    
@validate_data(IDetectHistory)
def updateDetectHistory(data:IDetectHistory,d_detect_history_id):
    data.update_at = datetime.now(timezone.utc)
    firebase_app.db.collection(COLLECTION_NAME).document(d_detect_history_id).set(data.model_dump(),merge=True)
    