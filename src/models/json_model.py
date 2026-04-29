from abc import ABC
from pydantic import BaseModel


class jsonModel(ABC, BaseModel):
    ...
