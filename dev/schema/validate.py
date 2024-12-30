from typing import Callable,Union,List,Optional,Any,TypeVar,Type
from typing_extensions import Never
from google.cloud.firestore_v1.base_document import DocumentSnapshot
from firebase_admin.exceptions import FirebaseError
from pydantic import BaseModel,ValidationError

T = TypeVar('T',bound=BaseModel)

def validate_data(schema:Type[T]):
    def outer(f:Callable[...,Any]):
        def inner(*args,**kwargs):
            try:
                d = args[0]
                schema(**d.model_dump())
            except ValidationError:
                raise ValidationError
            except FirebaseError as firebaseError:
                raise firebaseError
            else:
                result = f(*args,**kwargs)
                return result
        return inner
    return outer      


def validate_document(schema:Type[T]):
    def outer(f:Callable[...,Optional[DocumentSnapshot]]):
        def inner(*args,**kwargs)->Optional[schema]:
            try:
                document = f(*args,**kwargs)
                if document is None: return None
                if document.exists:
                    data = document.to_dict()
                    if not data or schema is None: return None
                    data = {'id':document.id,**data}
                    return schema(**data)
            except FirebaseError as firebaseError:
                raise firebaseError
            return
        return inner
    return outer

def validate_query(schema:Type[T]):
    def outer(f:Callable[...,List[DocumentSnapshot]]):
        def inner(*args,**kwargs)->Union[List[schema],List[Never]]:
            try:
                documentList = f(*args,**kwargs)
                if len(documentList) == 0: return []
                else:
                    resultList = []
                    for document in documentList:
                        if document.exists:
                            data = document.to_dict()
                            if not data: continue
                            data = {'id':document.id,**data}
                            resultList.append(data)
                    return [schema(**data) for data in resultList]
            except FirebaseError as firebaseError:
                raise firebaseError
        return inner
    return outer
    