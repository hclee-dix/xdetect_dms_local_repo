from dev.core.firebase import firebase_app
#
from dev.schema.firebase import ISystemServer
from dev.schema.validate import validate_data,validate_document

COLLECTION_NAME = 's_server'

@validate_data(ISystemServer)
def createSystemServer(data:ISystemServer):
    firebase_app.db.collection(COLLECTION_NAME).add(data.model_dump())

@validate_document(ISystemServer)
def readSystemServer(s_server_id):
    return firebase_app.db.collection(COLLECTION_NAME).document(s_server_id).get()

@validate_data(ISystemServer)
def updateSystemServer(data:ISystemServer,s_server_id):
    firebase_app.db.collection(COLLECTION_NAME).document(s_server_id).update(data.model_dump())

@validate_document(ISystemServer)
def deleteSystemServer(s_server_id,permanent=False):
    doc = firebase_app.db.collection(COLLECTION_NAME).document(s_server_id)
    if permanent: doc.delete()
    else: doc.update({'is_visible':False})
