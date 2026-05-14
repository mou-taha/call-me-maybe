from sys import argv
from .utils import (readArgs, verifyOptions, parseJsonData,
                    generatePrompt, write_output)
from .models.options import Options
from pydantic import ValidationError
from .models.prompt import Prompt
from .models.func_definition import FuncDefinition
from typing import cast
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from json import loads
import numpy as np
from json.decoder import JSONDecodeError
from .terminal_utils import (print_status, clear_terminal,
                             GREEN, CYAN, YELLOW, BLUE)


def main() -> None:
    """the entry point of the program, it will read the options from terminal,
    validate them, then parse the input files and validate them with pydantic
    models, after that it will generate the prompt for each user question
    and call the LLM model to generate the response, finally it will write
    the output to the output file."""
    try:
        # validate options from terminal and input files if exist
        options: Options = readArgs(argv[1:])
        is_valid_options, msg = verifyOptions(options)
        if is_valid_options:
            try:
                clear_terminal()
                print_status("Validating and loading input files...", GREEN)
                # parse json data from input files and validate it with
                # pydantic models
                prompts: list[Prompt] = cast(
                    list[Prompt], parseJsonData(options.input, Prompt)
                )
                func_defs: list[FuncDefinition] = cast(
                    list[FuncDefinition],
                    parseJsonData(options.functions_definition,
                                  FuncDefinition, True),
                )
                if len(func_defs) == 0 or len(prompts) == 0:
                    print(
                        f"{options.input} or {options.functions_definition}",
                        "is empty or contain invalid object.",
                    )
                    return
                else:
                    print_status("Preparing the model...", GREEN)
                    model = Small_LLM_Model()
                    vocab: dict[str, int] = {}
                    with (open(model.get_path_to_vocab_file(), "r")
                          as vocabFile):
                        # read json file and convert the dictionary data
                        vocab = loads(vocabFile.read())

                    prompts_result: list[str] = []
                    print_status("Generating responses...", GREEN)
                    for index, userQuestion in enumerate(prompts, start=1):
                        clear_terminal()
                        print_status(
                            f"User question: {userQuestion.prompt}",
                            CYAN,
                        )
                        print_status(
                            (f"[{index}/{len(prompts)}] Starting"
                             + "prompt generation..."),
                            CYAN,
                        )
                        prompt: str = generatePrompt(userQuestion.prompt,
                                                     func_defs)
                        prompts_result.append(
                            generateResponse(prompt, func_defs, model, vocab)
                        )

                    print_status("Writing output file...", GREEN)
                    write_output(options.output, prompts_result)
                    print("Done.")
            except (ValidationError, JSONDecodeError):
                print(
                    "Error while parsing JSON data, please check files are",
                    "contains valid JSON",
                )
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
    """this function will generate the response from the LLM model
    based on the generated prompt, it will first generate the function name
    then it will generate the parameters for that function,
    it will use the logits from the model to generate the response
    and it will use a mask to only allow the tokens that are in
    the function definitions and parameters,
    it will return the generated response as a string."""
    response: str = ""
    tokens: list[int] = model.encode(prompt)[0].tolist()

    func_def_logits: list[list[int]] = []
    for func in funcDefs:
        func_def_logits.append(model.encode(func.name)[0].tolist())
    if len(func_def_logits) == 0:
        raise ValueError("No matching function names remaining.")

    picked_function_tokens: list[int] = []
    print_status("Determining best function name...", CYAN)
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
        # pop the first value
        func_def_logits = [tokens[1:] for tokens in func_def_logits]

    # start generating parameters based on picked function
    picked_function_name: str = model.decode(picked_function_tokens)
    picked_function: FuncDefinition = [
        func for func in funcDefs if func.name == picked_function_name
    ][0]
    print_status(
        f"Picked function: {picked_function_name}",
        YELLOW,
    )
    print_status(
        f"Starting parameter generation for {picked_function_name}...",
        CYAN,
    )

    # encoding function parameters
    func_param_logits: list[list[int]] = []
    for paramName, paramType in picked_function.parameters.items():
        func_param_logits.append(model.encode(paramName)[0].tolist())

    if len(func_param_logits) == 0:
        raise ValueError("No matching function parameters names remaining.")

    tokens.extend(model.encode('","parameters": {')[0].tolist())

    picked_param_logits: list[int] = []
    # generating param
    params = list(picked_function.parameters.items())
    for i, (param_name, param_type) in enumerate(params):

        print_status(
            f"Generating value for parameter: {param_name}"
            + f" ({param_type.type})",
            BLUE,
        )

        # remove the parameters that already picked
        if picked_param_logits in func_param_logits:
            func_param_logits.remove(picked_param_logits)

        tokens.extend(model.encode(f'"{param_name}":')[0].tolist())
        # start generating param value
        if param_type.type.lower() in ["number", "integer", "float"]:
            counter: int = 0
            generated_tokens = []
            while True:
                counter += 1
                logits = model.get_logits_from_input_ids(tokens)
                mask = np.full_like(logits, -np.inf)
                tokenId = model.encode(",")[0].tolist()[0]
                mask[tokenId] = logits[tokenId]
                tokenId = model.encode("}")[0].tolist()[0]
                mask[tokenId] = logits[tokenId]
                if param_type.type.lower() != "integer":
                    tokenId = model.encode(".")[0].tolist()[0]
                    mask[tokenId] = logits[tokenId]
                tokenId = model.encode("-")[0].tolist()[0]
                mask[tokenId] = logits[tokenId]
                for n in range(0, 10):
                    tokenId = vocab[str(n)]
                    mask[tokenId] = logits[tokenId]
                best_token = np.argmax(mask)
                generated_tokens.append(model.decode(best_token))
                if any(x in ["}", ","] for
                       x in generated_tokens) or counter == 50:
                    break
                tokens.append(best_token)
            if (param_type.type.lower() in ["number", "float"]
                    and "." not in generated_tokens):
                tokens.extend(model.encode(".0")[0].tolist())

        if param_type.type.lower() == "string":
            tokens.extend(model.encode('"')[0].tolist())
            while True:
                logits = model.get_logits_from_input_ids(tokens)
                best_token = int(np.argmax(logits))
                decoded = model.decode(best_token)
                if '"' in decoded and "\\" not in decoded:
                    before_quotes = decoded.split('"')
                    tokens.extend(model.encode(before_quotes[0])[0].tolist())
                    break
                tokens.append(best_token)

            tokens.extend(model.encode('"')[0].tolist())
        if i < len(params) - 1:
            tokens.extend(model.encode(",")[0].tolist())
    tokens.extend(model.encode("}}")[0].tolist())
    response = model.decode(tokens).split("### RESPONSE:")[1]
    return response


if __name__ == "__main__":
    main()
