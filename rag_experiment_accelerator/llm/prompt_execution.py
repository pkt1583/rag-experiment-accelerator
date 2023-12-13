import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=90), stop=stop_after_attempt(6))
def generate_response(sys_message, prompt, engine_model, temperature):
    """
    Generates a response to a given prompt using the OpenAI Chat API.

    Args:
        sys_message (str): The system message to include in the prompt.
        prompt (str): The user's prompt to generate a response to.
        engine_model (str): The name of the OpenAI engine model to use for generating the response.
        temperature (float): Controls the "creativity" of the response. Higher values result in more creative responses.

    Returns:
        str: The generated response to the user's prompt.
    """
    prompt_structure = [
        {
            "role": "system",
            "content": sys_message,
        }
    ]

    prompt_structure.append({"role": "user", "content": prompt})

    params = {
        "messages": prompt_structure,
        "temperature": temperature,
    }
    if openai.api_type == "azure":
        params["engine"] = engine_model
    else:
        params["model"] = engine_model

    response = openai.ChatCompletion.create(**params)
    answer = response.choices[0]["message"]["content"]
    
    # Cleanse the response to remove any non-JSON characters, as different models return different formats
    answer = answer.replace('```json\n', '')
    answer = answer.replace('\n```', '')
    answer = answer.replace('\n', '')
    answer = answer.replace('\t', '')
    
    return answer
