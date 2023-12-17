import json
import jsonlines
import argparse

def count_tokens(text):
    return len(text)

def read_examples_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

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
    parser = argparse.ArgumentParser(description="Generate questions based on scripts and roles.")
    parser.add_argument("scripts_file")
    parser.add_argument("output_file")
    parser.add_argument("role_name")
    parser.add_argument("role_description")
    parser.add_argument("examples_file")
    return parser.parse_args()

def main():
    args = parse_arguments()
    max_tokens = 1000
    total_tokens = 0
    combined_list = []
    examples = read_examples_from_json(args.examples_file)

    with jsonlines.open(args.scripts_file) as scripts:
        for obj in scripts:
            current_text = f"{obj['role']}：{obj['content']}"
            current_tokens = count_tokens(current_text)

            if total_tokens + current_tokens > max_tokens:
                script = '\n'.join(combined_list)
                prompt = construct_prompt(script, args.role_name, args.role_description, examples)
                response = generate_chat_completion(prompt)
                write_to_file(args.output_file, response)

                combined_list = []
                total_tokens = 0

            combined_list.append(current_text)
            total_tokens += current_tokens

        if combined_list:
            script = '\n'.join(combined_list)
            prompt = construct_prompt(script, args.role_name, args.role_description, examples)
            response = generate_chat_completion(prompt)
            write_to_file(args.output_file, response)

def construct_prompt(script, role_name, role_description, examples):
    example_text = "\n".join([f"问题{i+1}: {ex['question']}\n回复: {ex['answer']}" for i, ex in enumerate(examples)])
    return f"""
    System Instruction:
        你的任务是设计{question_num}个向角色提问的问题，为了帮助你更好地
        设计问题，我会给你角色的简要描述、角色的部分剧本内容，剧本中包含角色情感的分类，在情感条件下更好的理解角色说话风格。这段剧本内容可能不连
        续，你需要根据上下文判断对话是否连续，如果不连续，不能构建上下句的逻辑关系。设计问题的
        规则如下：
        1. 记住，你所有的问题都需要向角色进行提问。
        2.问题和回答需要参考角色描述，但不要所有问题都来自于角色的描述，问题尽可能多样化。根据角色说话风格(比如情感，说话习惯，例如可爱的，幽默的)进行对话。
        3. 问题需要有完整性,完整性的高低取决于问题是否指明具体的人物，地点，事件。
        4. 问题需要围绕剧本的主要情节以及情节对应的剧本内容进行设计。
        5. 记住，你一共需要设计{question_num}个问题。
        6. 剧本只是辅助你设计问题，你应该更多地基于角色的常识进行设计。
        [样例]
        {example_text}
        User Prompt:
        [角色名及描述]
        剧本角色为{role_name}，角色描述为{role_description}
        [剧本内容]
        {script}
        """

def write_to_file(output_file, response):
    with open(output_file, "a", encoding="utf-8") as file:
        file.write("\n" + response)

if __name__ == "__main__":
    main()

# python label3.py scripts.jsonl output.txt "Role Name" "Role Description" examples.json
