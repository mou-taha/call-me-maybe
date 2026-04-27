from pydantic import BaseModel
from .json_model import jsonModel


class Parameter(BaseModel):
    type: str

class FuncDefinition(BaseModel, jsonModel):
    name: str
    description: str  
    parameters: dict[str, Parameter]
    returns: Parameter
