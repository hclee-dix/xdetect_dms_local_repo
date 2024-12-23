import os
from dotenv import load_dotenv
#
from types import SimpleNamespace
from ast import literal_eval
#

TYPE_TRUE_STRING = ['true','1','t']
ENV_FIREBASE_PREFIX = 'PYTHON_FIREBASE_'
ENV_PREFIX = 'PYTHON_FASTAPI_'
ENV_META_PREFIX = ENV_PREFIX+'META_'

def load_config(args):
    if not load_dotenv(f'.env.{args.mode}'):
        return None
    default_config = {}
    ## common
    port = int(os.getenv(ENV_PREFIX+'PORT',3000))
    host = os.getenv(ENV_PREFIX+'HOST','0.0.0.0')
    with_reload = os.getenv(ENV_PREFIX+'WITH_RELOAD','True').lower() in TYPE_TRUE_STRING
    ## worker
    workers = None
    if not with_reload:
        workers = int(os.getenv(ENV_PREFIX+'WORKERS',0))
    default_config.update({'workers':workers})
    ## cors
    with_cors = os.getenv(ENV_PREFIX+'WITH_CORS','False').lower() in TYPE_TRUE_STRING
    if with_cors:
        allow_credentials = os.getenv(ENV_PREFIX+'ALLOW_CREDENTIALS','False').lower() in TYPE_TRUE_STRING
        allow_methods = literal_eval(os.getenv(ENV_PREFIX+'ALLOW_METHODS','["*"]'))
        allow_headers = literal_eval(os.getenv(ENV_PREFIX+'ALLOW_HEADERS','["*"]'))
        origin_list = literal_eval(os.getenv(ENV_PREFIX+'ORIGIN_LIST','["*"]'))
        default_config.update({'allow_credentials':allow_credentials,'allow_methods':allow_methods,'allow_headers':allow_headers,'origin_list':origin_list})
    ## logging
    logger_formatter = os.getenv('PYTHON_FAST_API_LOGGER_FORMATTER','%(message)s')
    ## meta
    meta = SimpleNamespace(**{
        'title':os.getenv(ENV_META_PREFIX+'TITLE',''),
        'version':os.getenv(ENV_META_PREFIX+'VERSION','0.0.1'),
        'tos':os.getenv(ENV_META_PREFIX+'TERMS',''),
        'contact':literal_eval(os.getenv(ENV_META_PREFIX+'CONTACT','{"name":"","url":"","email":""}')),
        'license':literal_eval(os.getenv(ENV_META_PREFIX+'LICENSE_INFO','{"name":'',"url":""}'))
    })
    
    default_config.update({'mode':args.mode,'host':host,'port':port,'with_reload':with_reload,'with_cors':with_cors,'logger_formatter':logger_formatter,'meta':meta})
    cfg = SimpleNamespace(**default_config)
    return cfg