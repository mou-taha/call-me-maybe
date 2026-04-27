from pydantic import BaseModel
from .json_model import jsonModel

class FuncDefinition(BaseModel, jsonModel):
    prompt: str