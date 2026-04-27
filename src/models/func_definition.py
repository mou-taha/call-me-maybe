from pydantic import BaseModel, field_validator
from .json_model import jsonModel
import builtins


class Parameter(BaseModel):
    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        # Common Python types mapping
        valid_types = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "None": type(None),
            "number": float,  # Common in JSON schemas
            "string": str,
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        if v not in valid_types:
            # Try to get from builtins
            if hasattr(builtins, v):
                return v
            raise ValueError(f"'{v}' is not a valid Python type")
        return v


class FuncDefinition(BaseModel, jsonModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter
