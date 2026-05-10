from getopt import getopt
from .models.options import Options
from pathlib import Path
from .models.json_model import jsonModel
from pydantic import ValidationError  # type: ignore
from json import loads
from .models.func_definition import FuncDefinition
from typing import Type


def readArgs(args: list[str]) -> Options:
    """this function will read the options from terminal and return an Options
    object, it will use getopt to parse the options and it will return the
    options as an Options object."""
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
    """this function will verify the options provided by the user,
    it will return a tuple of a boolean and a string,
    the boolean will indicate if the options are valid or not,
    and the string will contain the error message if any."""
    if not Path(options.functions_definition).exists():
        return (
            False,
            "Functions definition file is required."
            f"{options.functions_definition} does not exist.",
        )
    if not Path(options.input).exists():
        return False, f"Input file is required.{options.input} does not exist."
    return True, ""


def parseJsonData(filePath: str,
                  model_class: Type[jsonModel]) -> list[jsonModel]:
    data: list[jsonModel] = []
    with open(filePath, "r") as f:
        jsonData = loads(f.read())
        for p in jsonData:
            try:
                prompt = model_class.model_validate(p)
                data.append(prompt)
            except ValidationError:
                continue
    return data


def generatePrompt(userPrompt: str, funcDef: list[FuncDefinition]) -> str:
    """this function will generate the prompt for the LLM model based on the
    user question and the function definitions, it will return the generated
    prompt as a string."""
    userPrompt = userPrompt.replace('"', '\\"')
    prompt = """You ar a function calling system, Your task is to extract the
correct function name and its parameters from the user's
question based on the definitions below.
this is the list of functions that you must pick from it:

### AVAILABLE FUNCTIONS:"""
    for func in funcDef:
        prompt += f"""\n
    - name: {func.name}
      description: {func.description}
      parameter: -"""
        prompt += " -".join(
            [f"{key}: type {value.type}" for key,
                value in func.parameters.items()]
        )
        prompt += f"""
      return: type {func.returns.type}"""
    prompt += """
### INSTRUCTIONS:
1. analyze the user question
2. select the most relevant function name
3. extract the required parameters with respecting their types
4. respond only with a JSON object in this exact format :
{"prompt": "user prompt","name": "function name","parameters": {"a": value
with exact type, "b": value with exact type}}"""
    prompt += f'''

### USER QUESTION:
{userPrompt}

### RESPONSE:
{{"prompt": "{" ".join(userPrompt.split())}","name": "'''
    return prompt


def write_output(filePath: str, result: list[str]) -> None:
    if len(result) > 0:
        output_file = Path(filePath)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(output_file, "w") as file:
            file.write("[")
            for index, r in enumerate(result):
                file.write(r)
                if len(result) - index != 1:
                    file.write(",")
            file.write("]")
