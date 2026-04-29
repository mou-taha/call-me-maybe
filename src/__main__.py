from sys import argv
from .utils import readArgs, verifyOptions, parseJsonData
from .models.options import Options
from pydantic import ValidationError
from .models.prompt import Prompt
from .models.func_definition import FuncDefinition
from typing import cast
from llm_sdk import Small_LLM_Model
from json import loads


def main():
    try:
        # validate options from terminal and input files if exist
        options: Options = readArgs(argv[1:])
        is_valid_options, msg = verifyOptions(options)
        if is_valid_options:
            try:
                # parse json data from input files and validate it with
                # pydantic models
                prompts: list[Prompt] = cast(
                    list[Prompt], parseJsonData(options.input, Prompt)
                )
                func_defs: list[FuncDefinition] = cast(
                    list[FuncDefinition],
                    parseJsonData(options.functions_definition, FuncDefinition),
                )
                if len(func_defs) == 0 or len(prompts) == 0:
                    print(
                        f"{options.input} or {options.functions_definition} is empty or contain invalid object."
                    )
                    return
                else:
                    print(f"treating prompts... [{prompts[2].prompt}]")
                    model = Small_LLM_Model()
                    vocab: list[tuple[str, str]] = []
                    with open(model.get_path_to_vocab_file(), "r") as vocabfile:
                        # read json file and convert the dictionary data to a list of tuples that contain vocab and its token
                        vocab = list(loads(vocabfile.read()).items())

                    tokens: list[int] = model.encode(prompts[2].prompt)[0].tolist()
                    print(tokens)
                    while True:
                        logits = model.get_logits_from_input_ids(tokens)
                        max_logit_index = logits.index(max(logits))
                        _, max_logit = vocab[max_logit_index]
                        tokens.append(max_logit)
                        print(model.decode(tokens))
            except ValidationError:
                print("Error parsing JSON data")
                return
        else:
            print(f"Invalid options provided: {msg}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
