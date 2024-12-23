from fastapi import HTTPException,Header
from typing_extensions import Annotated

async def verify_token(x_token: Annotated[str, Header()]):
    if x_token != "gHWdiqSVuUa5EG7IhYxRyMjIwbfzwscdO7XewY4NN":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def verify_key(x_key: Annotated[str, Header()]):
    if x_key != "CfDlzVa44WQKVefQbgwMsw9PQoqyL458q3UaWDO2":
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    return x_key