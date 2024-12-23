#
# Description: create FastApi Instance with arguments
#

from fastapi import FastAPI
from .api.v1.endpoint import model
from .api.v1.endpoint import status


## markdown format
## NOTE: load from file
DESCRIPTION = """
# A Integration for X.Detect Service🪄

## Train

You can **read train**.

## Detect

You can **read detect**.

---
"""

def getAppInstance(args):
    """configure fastapi app instance with arguments

    Args:
        args (dict): environment argument
            meta (dict): configure meta information
                title (str): set title
                version (str): set version
                tos (str): set term of services url
                contact (json): set contact information
                license (json): set license information
            with_cors (bool): set CORS(cross origin resource sharing)
                origin_list (list): allow origin list
                allow_credentials (bool): allow authentication
                allow_methods (list): allow REST API method (e.g. get,post,...)
                allow_headers (list): allow Header information (e.g. *)

    Returns:
        FastAPI: fastapi instance
    """
    instance = FastAPI(
        title=args.meta.title,
        description=DESCRIPTION,
        version=args.meta.version,
        terms_of_service=args.meta.tos,
        contact=args.meta.contact,
        license_info=args.meta.license,
    )
    
    instance.include_router(model.router)
    instance.include_router(status.router)
    if args.with_cors:
        from fastapi.middleware.cors import CORSMiddleware
        instance.add_middleware(
            CORSMiddleware,
            allow_origins=args.origin_list,
            allow_credentials=args.allow_credentials,
            allow_methods=args.allow_methods,
            allow_headers=args.allow_headers,
        )
    return instance

def createFastAPIserverInstance(cfg):
    return getAppInstance(cfg)
