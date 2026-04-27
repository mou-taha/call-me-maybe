from sys import argv
from .utils import *
from .models.options import Options
from .models.func_definition import FuncDefinition   
import pathlib
from pydantic import TypeAdapter


def main():
    try:
        # validate options from terminal and input files if exist
        options: Options = readArgs(argv[1:])
        is_valid_options, msg = verifyOptions(options)
        if is_valid_options:
            funcDefAdapter = TypeAdapter(list[FuncDefinition])
            jsonFuncDef = pathlib.Path(options.functions_definition).read_text()
            func_definitions = funcDefAdapter.validate_json(jsonFuncDef)
            for func_def in func_definitions:
                print("Function Name:", func_def.name)
                print("Description:", func_def.description)
                print("Parameters:", func_def.parameters)
                print("Returns:", func_def.returns)
                print()
        else:
            print(f"Invalid options provided: {msg}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()