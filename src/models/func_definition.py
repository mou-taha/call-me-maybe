from pydantic import BaseModel, Field
from .json_model import jsonModel
from typing import Literal


class Parameter(BaseModel):
    type: Literal["string", "number"]


class FuncDefinition(jsonModel):
    name: str = Field(..., max_length=50)
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter
