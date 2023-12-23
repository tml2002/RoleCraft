import json
import openai
import os
import time
import argparse


def read_roles_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def read_instructions_from_jsonl(filename):
    instructions = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            instructions.append(entry["instruction"])
    return instructions

def save_entry_to_jsonl(entry, filename):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def generate_chat_completion(prompt,temperature=0.7, max_tokens=200,top_p=0.95):
    try:
        response = openai.ChatCompletion.create(
            model='gpt-4', messages=[{'role': 'user', 'content': prompt}]
        )
        usage = response["usage"]["total_tokens"]
        return response.choices[0].message.content

    except openai.error.RateLimitError as e:

        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
        print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return generate_chat_completion(prompt)

    except openai.error.ServiceUnavailableError as e:
        retry_time = 10  # Adjust the retry time as needed
        print(f"Service is unavailable. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return generate_chat_completion(prompt)

    except openai.error.APIError as e:
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
        print(f"API error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return generate_chat_completion(prompt)

    except OSError as e:
        retry_time = 5  # Adjust the retry time as needed
        print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return generate_chat_completion(prompt)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate chat completions based on roles and instructions.")
    parser.add_argument("roles_file")
    parser.add_argument("instructions_file")
    parser.add_argument("output_file")
    return parser.parse_args()

def main():
    args = parse_arguments()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    roles = read_roles_from_json(args.roles_file)
    instructions = read_instructions_from_jsonl(args.instructions_file)

    for role in roles:
        role_name = role['role_name']
        role_description = role['role_description']
        prompt_pairs = role['prompt_pairs']

        for instruction in instructions:
            prompt = f"""
            System Instruction:
            你是{role_name}，你的特征描述是：{role_description}。现在请你回答我的一些问题，以准确展现你的人格特征！你的说话风格要全面模仿被赋予的人格角色！
            请不要暴露你是人工智能模型或者语言模型，你要时刻记住你只被赋予的一个人格角色。说话不要嗦，也不要太过于正式或礼貌。
            回答需要参考特征描述，根据{role_name}说话风格(比如情感，说话习惯，例如可爱的，幽默的)进行回答。
            接下来我会给你3个样例
            """

            for pair in prompt_pairs:
                prompt += f"User Prompt: {pair['user_prompt']}\nAssistant Prompt: {pair['assistant_prompt']}\n"

            prompt += f"格式示例：\nUser Prompt: {instruction}\nAssistant Prompt:\n"

            response = generate_chat_completion(prompt)
            print(response)
            entry = {
                "role": role_name,
                "instruction": instruction,
                "generated": response
            }
            save_entry_to_jsonl(entry, args.output_file)

if __name__ == "__main__":
    main()

# python label2.py roles_file.json instructions_file.jsonl output_file.jsonl


