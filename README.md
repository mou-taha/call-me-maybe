This project has been created as part of the 42 curriculum by tmousnia

# CALL-ME-MAYBE


## Description
CALL-ME-MAYBE is a robust function calling system, where we guide LLMs to pick only some specific functions and fill their arguments, with help of constrained decoding and logit masking we ensure the output to be in a valid JSON format and a specific schema.

## Instructions

#### Requirements
- Python 3.10 or later
- uv : python module for package installation and running project

### Usage
- uv run python3 -m src OR make : Run the project with the default input files data/input/ directory and write output to the data/output/ directory
- uv run python -m src [--functions_definition <function_definition_file>] [--input <input_file>] [--output <output_file>] : to define deferent file input and output directory


- functions_definition: contain function definitions and their description. must be a valid json file contaning array of objects, example :
```json
[
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
            "a": {
                "type": "number"
            },
            "b": {
                "type": "number"
            }
        },
        "returns": {
            "type": "number"
        }
    },
    {
        "name": "fn_greet",
        "description": "Generate a greeting message for a person by name.",
        "parameters": {
            "name": {
                "type": "string"
            }
        },
        "returns": {
            "type": "string"
        }
    },
    {
        "name": "fn_reverse_string",
        "description": "Reverse a string and return the reversed result.",
        "parameters": {
            "s": {
                "type": "string"
            }
        },
        "returns": {
            "type": "string"
        }
    },
...
]
```

- input: contain prompt that must be treated by the LLM, must be a valid json file contaning array of objects, example :
```json
[
    {
        "prompt": "What is the sum of 2 and 3?"
    },
    {
        "prompt": "What is the sum of 265 and 345?"
    },
    {
        "prompt": "Greet shrek"
    },
    {
        "prompt": "Greet john"
    },
    {
        "prompt": "Reverse the string 'hello'"
    },
...
]
```

##  Algorithm explanation
### first step prepare the prompt

    to guid the LLM to generate a valid json, i start explain the mission inside a prompt that i'll load to the LLM later, where i tell him that he is a function calling system and he must pick one of the functions that i listed in the prompt, and he must generate a valid json object.
    to ensure a valid JSON entry point, i do a prefix forcing ```{{"prompt":"user prompt","name":"... ```.

### constrained decoding: 
    After encoding the initial prompt, the system enters a controlled generation loop:

###### function name:

The system ensures a valid function name is picked by using a **Trie-based filtering** approach.

1. The available function names are converted into token sequences ($list[list[int]]$).
2. At each generation step, the system calculates logits but applies a **Logit Mask**.
3. All tokens in the vocabulary are set to $-\infty$ except for the specific tokens that could legally continue one of the remaining function names.
4. After the model selects the best (highest logit) valid token, the list of candidate functions is narrowed down.
5. This iterates until a full function name is completed, guaranteeing the model never "invents" a tool.

##### parameters:
   
Once the function is identified, the system transitions to the "Parameter generating." It force this structural JSON switch `", "parameters": {`.
For each parameter defined in the **JSON schema**:

* **Type-Aware Masking:** The system checks the parameter type.
* **Numbers:** The mask is restricted to digits, decimal points, comma, and signs. The loop breaks when a exit (comma) is detected in the model's generated token.
* **Strings:** The system forces an opening quote (`"`), then switches to a "Content". the loop break when closing double quotes '"'.
* **Structural Forcing:** The system manually manages commas (`,`) between arguments and ensures the final JSON object is closed with the correct number of braces (`}}`).

## Challenges faced:

The primary hurdle was the **"Greedy Token"** problem.

Subword tokenizers often combine data and syntax into a single token (the model might want to generate `]"` as one token ID). My initial algorithm would break the loop upon seeing the quote, but because I didn't append that specific token, I would lose the vital character (the `]`) preceding it.

To solve this, I implemented this strategy: when a "contaminated" token is detected, the system extracts the valid content before the structural character, appends that part to the response, and then breaks to let the manual state machine handle the clean JSON punctuation.

---
##  Testing strategy:

The system was validated using the provided `function_calling_tests.json`. Furthermore, testing was conducted with irrelevant prompts to ensure the model maintains structural integrity even when semantic intent is low.

---


## Design decisions

* **State Machine Approach:** Instead of letting the LLM "guess" the JSON structure, the system uses a deterministic state machine. This ensures the model only generates values (strings/numbers) while the code handles the structural syntax (brackets, colons, commas).
* **Prefix Forcing:** By manually pre-filling the response with `{"prompt": "...", "name": "`, the model is "trapped" in a valid JSON state from the very first token, preventing it from starting with conversational filler like "Sure, here is the function...".
* **Intent-Based Breaking:** The system monitors the model's "natural intent" (what it *wants* to pick) to decide when to close a value. This allows the model to finish its thought (like closing a regex bracket) before the system forces the structural closing quote.

## Performance analysis:

* **Optimization:** To avoid slow iteration over the 151k+ vocabulary items during generation, i iterate over "allowed token" lists.


## Example usage:

To run the system with the default configuration, use the following command:

```bash
# Using the makefile
make

# Or using uv directly
uv run python3 -m src

```

To process a custom function list and input file:

```bash
uv run python -m src --functions_definition my_tools.json --input my_prompts.json --output results.json

```

---