from getopt import getopt
from .models.options import Options
from pathlib import Path
from .models.json_model import jsonModel
from pydantic import ValidationError
from json import loads
from llm_sdk import Small_LLM_Model
from .models.func_definition import FuncDefinition, Parameter


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
                continue
    return data


def generatePrompt(userPrompt: str, funcDef: list[FuncDefinition]) -> str:
    prompt = """You ar a function calling system, Your task is to extract the correct function name and its parameters from the user's question based on the definitions below.
this is the list of functions that you must pick from it:

### AVAILABLE FUNCTIONS:"""
    for func in funcDef:
        prompt += f"""\n
    - name: {func.name}
      description: {func.description}
      parameter: -"""
        prompt += " -".join(
            [f"{key}: type {value.type}" for key, value in func.parameters.items()]
        )
        prompt += f"""
      return: type {func.returns.type}"""
    prompt += """
### INSTRUCTIONS:
1. analyze the user question
2. select the most relevant function name
3. extract the required parameters with respecting their types
4. respond only with a JSON object in this exact format :
{"prompt": "user prompt","name": "function name","parameters": {"a": value with exact type, "b": value with exact type}}"""
    prompt += f'''

### USER QUESTION:
{userPrompt}

### RESPONSE:
{{"prompt": "{userPrompt}","name":" '''
    return prompt
