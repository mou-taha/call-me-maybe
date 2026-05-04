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
                    model = Small_LLM_Model()
                    vocab: dict[str, int] = {}
                    with open(model.get_path_to_vocab_file(), "r") as vocabFile:
                        # read json file and convert the dictionary data to a list of tuples that contain vocab and its token
                        vocab = loads(vocabFile.read())

                    # for userQuestion in prompts:
                    #     prompt: str = generatePrompt(userQuestion.prompt, func_defs)
                    #     generateResponse(prompt, func_defs, model, vocab)
                    generateResponse(
                        generatePrompt(prompts[10].prompt, func_defs),
                        func_defs,
                        model,
                        vocab,
                    )
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
    if len(func_def_logits) == 0:
        raise ValueError("No matching function names remaining.")

    picked_function_tokens: list[int] = []
    # generating function name
    while any([f for f in func_def_logits if len(f) > 0]):
        logits = np.array(model.get_logits_from_input_ids(tokens))

        mask = np.full_like(logits, -np.inf)
        allowed_tokens = [tokens[0] for tokens in func_def_logits]
        for tokenId in allowed_tokens:
            mask[tokenId] = logits[tokenId]

        best_token = np.argmax(mask)
        tokens.append(best_token)
        picked_function_tokens.append(best_token)
        # keep only function logits that start
        func_def_logits = [
            tokens for tokens in func_def_logits if tokens[0] == best_token
        ]
        func_def_logits = [tokens[1:] for tokens in func_def_logits]

    # start generating parameters based on picked function
    picked_function_name: str = model.decode(picked_function_tokens)
    picked_function: FuncDefinition = [
        func for func in funcDefs if func.name == picked_function_name
    ][0]

    # encoding function parameters
    func_param_logits: list[list[int]] = []
    for paramName, paramType in picked_function.parameters.items():
        func_param_logits.append(model.encode(paramName)[0].tolist())

    if len(func_param_logits) == 0:
        raise ValueError("No matching function parameters names remaining.")

    tokens.extend(model.encode('","parameters": {"')[0].tolist())

    picked_param_logits: list[int] = []
    param_handled: int = 0
    # generating param
    for param_name, param_type in picked_function.parameters.items():
        param_handled = param_handled + 1
        # work on a copy for function parameters logits
        func_param_logits_clone = func_param_logits[:]

        # remove the parameters that already picked
        if len(picked_param_logits) > 0:
            # TODO: fix the error when iterating over string args
            func_param_logits.remove(picked_param_logits)
        # generating param key
        while any([param for param in func_param_logits_clone if len(param) > 0]):
            logits = model.get_logits_from_input_ids(tokens)
            mask = np.full_like(logits, -np.inf)
            allowed_tokens = [token[0] for token in func_param_logits_clone]
            for tokenId in allowed_tokens:
                mask[tokenId] = logits[tokenId]
            best_token = np.argmax(mask)
            tokens.append(best_token)
            picked_param_logits.append(int(best_token))
            func_param_logits_clone = [
                logits for logits in func_param_logits_clone if logits[0] == best_token
            ]
            func_param_logits_clone = [logits[1:] for logits in func_param_logits_clone]

        # start generating param value
        tokens.extend(model.encode('": ')[0].tolist())
        if param_type.type.lower() == "number":
            while model.decode(tokens)[-1] != '"' and model.decode(tokens)[-1] != "}":
                logits = model.get_logits_from_input_ids(tokens)
                mask = np.full_like(logits, -np.inf)
                tokenId = model.encode(' "')[0].tolist()[0]
                mask[tokenId] = logits[tokenId]
                tokenId = model.encode('"')[0].tolist()[0]
                mask[tokenId] = logits[tokenId]
                tokenId = model.encode(".")[0].tolist()[0]
                mask[tokenId] = logits[tokenId]
                tokenId = model.encode("-")[0].tolist()[0]
                mask[tokenId] = logits[tokenId]
                # include the char "," only if still parameters to add for the function if not its put on -inf to not be included in next tokens
                if len(picked_function.parameters.items()) - param_handled > 0:
                    tokenId = model.encode(",")[0].tolist()[0]
                    mask[tokenId] = logits[tokenId]
                tokenId = model.encode("}")[0].tolist()[0]
                mask[tokenId] = logits[tokenId]
                for n in range(0, 9):
                    tokenId = vocab[str(n)]
                    mask[tokenId] = logits[tokenId]
                tokens.append(np.argmax(mask))

        if param_type.type.lower() == "string":
            while True:
                logits = model.get_logits_from_input_ids(tokens)
                # put on the char "," to -inf if we are in the last parameter to force the model to close the object
                # if len(picked_function.parameters.items()) - param_handled == 0:
                #     tokenId = model.encode(",")[0].tolist()[0]
                #     logits[tokenId] = -np.inf
                best_token = np.argmax(logits)
                tokens.append(best_token)
                print(model.decode(tokens))
                if model.decode(best_token) == '",':
                    break

    # print(model.decode(tokens))

    return response


if __name__ == "__main__":
    main()
