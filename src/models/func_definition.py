from pydantic import BaseModel, Field
from .json_model import jsonModel


class Parameter(BaseModel):
    type: str


class FuncDefinition(jsonModel):
    name: str = Field(..., max_length=100)
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter
