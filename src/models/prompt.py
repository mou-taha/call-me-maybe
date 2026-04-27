from pydantic import BaseModel
from .json_model import jsonModel


class Prompt(BaseModel, jsonModel):
    prompt: str
