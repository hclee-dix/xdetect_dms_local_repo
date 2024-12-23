import os
from dotenv import load_dotenv
from firebase_admin import initialize_app,App,credentials, firestore, storage
#
from dataclasses import dataclass
from google.cloud.firestore import Client
from google.cloud.storage import Bucket

@dataclass
class Firebase:
    app:App
    db:Client
    bucket:Bucket

def getFirebaseInstance():
    load_dotenv()
    env = os.getenv("PYTHON_FIREBASE_ENV","dev")
    isDev = env == 'dev'
    config = 'firebase-adminsdk-dev.json' if isDev else 'firebase-adminsdk-prd.json'
    bucketName = 'xdetect-dev.appspot.com' if isDev else 'xdetect-prd.appspot.com'
    
    # Firebase 프로젝트의 서비스 계정 키 파일 경로
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cred_path = os.path.join(current_dir,config)
    cred = credentials.Certificate(cred_path)

    # Firebase 앱 초기화
    app = initialize_app(cred)
    # Firestore,Storage 클라이언트 초기화
    db = firestore.client(app)
    bucket = storage.bucket(bucketName,app)
    instance = Firebase(app=app,db=db,bucket=bucket)
    return instance

firebase_app = getFirebaseInstance()