from transformers import AutoTokenizer
from openai import OpenAI

def replace_final_eot(text, replacement):
    parts = text.rsplit("<|im_end|>", 1)
    if len(parts) == 2:
        return f"{parts[0]}{replacement}{parts[1]}"
    return text
    
def interactive_math_solver(
    g_model_name,
    rm_model_name,
    g_base_url,
    rm_base_url,
    g_api_key,
    rm_api_key,
    max_n,
    prompt,
):
    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(g_model_name)

    # Initialize Clients
    g_client = OpenAI(base_url=g_base_url, api_key=g_api_key)
    rm_client = OpenAI(base_url=rm_base_url, api_key=rm_api_key)

    # System Messages
    g_system_message = "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. Return your final answer in \\boxed{N} format."

    rm_system_message = (
        "**You are an expert evaluator tasked with assessing solutions. Given a question and its proposed solution, decide the appropriate action based on its quality:**\n"
        "1. **\"Continue\"**: The solution is correct or sufficiently on track and can proceed without changes.\n"
        "2. **\"Retry\"**: The solution has minor errors or issues that require a small, specific correction before continuing.\n"
        "3. **\"Startover\"**: The solution is fundamentally flawed or off-track, requiring a complete restart."
    )

    # Reinforcement Map
    rmap = {
        "continue": "Hmm.. For now the solution looks fine. Let's continue.",
        "retry": "Hmm.. I just looked over solution. Something is off with the previous line. Let me retry it.",
        "start": "Wait... I'm pretty sure this response is totally off. Let me try re-solving the question from the beginning.",
    }

    
    # Initialize loop variables
    messages = [
        {"role": "system", "content": g_system_message},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    completion = g_client.completions.create(
            model=g_model_name, prompt=text, max_tokens=512, temperature=0.0, stop=["\n\n"]
        )
    initial_node = completion.choices[0].text
    response = [initial_node]
    rm_judgement = ""
    n = 0

    while True:
        # Generate solution step
        messages = [
          {"role": "system", "content": g_system_message},
          {"role": "user", "content": prompt},
          {"role": "assistant", "content": '\n'.join(response)}
      ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        text = replace_final_eot(text, "")

        completion = g_client.completions.create(
            model=g_model_name, prompt=text, max_tokens=512, temperature=0.0, stop=["\n\n"]
        )

        generated_text = completion.choices[0].text
        response.append(generated_text)

        # Check stopping conditions
        if n > 1:
            rm_messages = [
                {"role": "system", "content": rm_system_message},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "\n\n".join(response)},
            ]
            rm_text = tokenizer.apply_chat_template(rm_messages, tokenize=False)

            rm_completion = rm_client.completions.create(
                model=rm_model_name, prompt=rm_text, max_tokens=1
            )

            rm_judgement = rmap[rm_completion.choices[0].text]
            response.append(rm_judgement)

        if n > max_n or "\\boxed" in response[-2]:
            break

        n += 1

    return "\n".join(response[:-1])

# Example usage:
# response = interactive_math_solver(
#     g_model_name="Qwen/Qwen2.5-7B-Instruct",
#     rm_model_name="amphora/dprm-ckpt1",
#     g_base_url="http://80.188.223.202:13912/v1",
#     rm_base_url="http://80.188.223.202:11533/v1",
#     g_api_key="token-abc123",
#     rm_api_key="token-abc123",
#     max_n=10,
#     prompt="Your math problem here",
# )
# print(response)
