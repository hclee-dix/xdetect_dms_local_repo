from firebase_admin import firestore
from datetime import datetime,timezone

def commonField():
    return {'is_visible':True,'create_at':firestore.firestore.SERVER_TIMESTAMP,'update_at':firestore.firestore.SERVER_TIMESTAMP}

def timestamp():
    return firestore.firestore.SERVER_TIMESTAMP