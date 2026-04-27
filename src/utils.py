from getopt import getopt
from .models.options import Options
from pathlib import Path
from json import load

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
        return  False, f"Functions definition file is required. {options.functions_definition} does not exist."
    if  not Path(options.input).exists():
        return False, f"Input file is required. {options.input} does not exist."
    return True, ""


def parseJsonFile(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return load(file)