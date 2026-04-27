from sys import argv
from .utils import readArgs, verifyOptions, parseJsonData
from .models.options import Options
from pydantic import ValidationError


def main():
    try:
        # validate options from terminal and input files if exist
        options: Options = readArgs(argv[1:])
        is_valid_options, msg = verifyOptions(options)
        if is_valid_options:
            try:
                # parse json data from input files and validate it with
                # pydantic models
                data = parseJsonData(options)
            except ValidationError:
                print("Error parsing JSON data")
                return
            for func_def in data["func_definitions"]:
                print("Function Name:", func_def.name)
                print("Description:", func_def.description)
                print("Parameters:", func_def.parameters)
                print("Returns:", func_def.returns)
                print()
            for func_def in data["prompts"]:
                print("Prompt:", func_def.prompt)
                print()
        else:
            print(f"Invalid options provided: {msg}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
