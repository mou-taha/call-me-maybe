from getopt import getopt
from .models.options import Options
from pathlib import Path
from .models.prompt import Prompt
from .models.func_definition import FuncDefinition
from .models.json_model import jsonModel
from pydantic import TypeAdapter, ValidationError
from json import loads


def readArgs(args: list[str]) -> Options:
    options = Options()
    long_options = ["functions_definition=", "input=", "output="]
    arguments, _ = getopt(args, "", long_options)

    for arg, val in arguments:
        if arg == "--functions_definition":
            options.functions_definition = val
        elif arg == "--input":
            options.input = val
        elif arg == "--output":
            options.output = val
    return options


def verifyOptions(options: Options) -> tuple[bool, str]:
    if not Path(options.functions_definition).exists():
        return (
            False,
            "Functions definition file is required."
            f"{options.functions_definition} does not exist.",
        )
    if not Path(options.input).exists():
        return False, f"Input file is required.{options.input} does not exist."
    return True, ""


def parseJsonData(filePath: str, cls: type) -> list[jsonModel]:
    data: list[cls] = []
    with open(filePath, "r") as f:
        jsonData = loads(f.read())
        for p in jsonData:
            try:
                prompt = cls.model_validate(p)
                data.append(prompt)
            except ValidationError as e:
                print(e)
                continue
    return data
