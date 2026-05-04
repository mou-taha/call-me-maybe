from sys import argv
from .utils import readArgs, verifyOptions, parseJsonData, generatePrompt
from .models.options import Options
from pydantic import ValidationError
from .models.prompt import Prompt
from .models.func_definition import FuncDefinition
from typing import cast
from llm_sdk import Small_LLM_Model
from json import loads
import numpy as np


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
                    print("treating prompts...")
                    model = Small_LLM_Model()
                    vocab: dict[str, int] = {}
                    with open(model.get_path_to_vocab_file(), "r") as vocabFile:
                        # read json file and convert the dictionary data to a list of tuples that contain vocab and its token
                        vocab = loads(vocabFile.read())

                    for userQuestion in prompts:
                        prompt: str = generatePrompt(userQuestion.prompt, func_defs)
                        generateResponse(prompt, func_defs, model, vocab)
            except ValidationError:
                print("Error parsing JSON data")
                return
        else:
            print(f"Invalid options provided: {msg}")
    except Exception as e:
        print(f"An error occurred: {e}")


def generateResponse(
    prompt: str,
    funcDefs: list[FuncDefinition],
    model: Small_LLM_Model,
    vocab: dict[str, int],
) -> str:
    response: str = ""
    tokens: list[int] = model.encode(prompt)[0].tolist()

    func_def_logits: list[list[int]] = []
    for func in funcDefs:
        func_def_logits.append(model.encode(func.name)[0].tolist())

    while True:
        logits = model.get_logits_from_input_ids(tokens)
        # if len(func_def_logits) != 1 and len([func[0] for func in func_def_logits]) != 0:
        # if len(func_def_logits) == 1:
        #      break
        # TODO: find the condition to break at the end of func name
        for _, id in vocab.items():
            if id not in [tokens[0] for tokens in func_def_logits]:
                logits[id] = -np.inf

        best_token = logits.index(max(logits))
        tokens.append(best_token)
        func_def_logits = [
            tokens for tokens in func_def_logits if tokens[0] == best_token
        ]
        func_def_logits = [tokens[1:] for tokens in func_def_logits]
        print(len(func_def_logits))
        print(len([func[0] for func in func_def_logits]))
        # print(model.decode(tokens))

    # example masking
    # [10, 20]
    # [40, 80]

    # allowed = [10, 40]

    # for token, id in model_dict.items():
    #     if id not in allowed:
    #         logits[id] = -inf

    return response


if __name__ == "__main__":
    main()
