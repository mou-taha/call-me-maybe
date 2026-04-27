from getopt import getopt
from .models.options import Options
from pathlib import Path
from .models.prompt import Prompt
from .models.func_definition import FuncDefinition
from .models.json_model import jsonModel
from pydantic import TypeAdapter


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


def parseJsonData(options: Options) -> dict[str, list[jsonModel]]:
    promptAdapter = TypeAdapter(list[Prompt])
    jsonPrompt = Path(options.input).read_text()
    prompts = promptAdapter.validate_json(jsonPrompt)
    funcDefAdapter = TypeAdapter(list[FuncDefinition])
    jsonFuncDef = Path(options.functions_definition).read_text()
    func_calling_tests = funcDefAdapter.validate_json(jsonFuncDef)
    return {"prompts": prompts, "func_definitions": func_calling_tests}
