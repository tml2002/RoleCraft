import openai
import os
import time
import argparse
import json

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
    parser = argparse.ArgumentParser(description="Perform sentiment analysis on dialogues from a JSON file.")
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    return parser.parse_args()

def main():
    args = parse_arguments()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    with open(args.input_file, "r", encoding="utf-8") as file:
        json_data_list = json.load(file)

    sentiments = []
    emotion_labels = ["愤怒", "厌恶", "恐惧", "快乐", "悲伤", "惊讶", "中立", "沮丧", "兴奋", "其他"]

    for data in json_data_list:
        emotions = []
        for sentence in data["dialog"]:
            print(sentence)
            prompt = f"""
            你的任务是对提供的文本进行情感分析。每段文本应被分类为以下情感之一：愤怒, 厌恶, 恐惧, 快乐,悲伤,惊讶,中立,沮丧,兴奋,其他。请仔细阅读每个文本，根据其内容和语调判断出最显著的情感，并给出相应的分类。
            1. 对每段文本只做一个情感分类。
            2. 如果文本表达了多种情感，选择最为突出或主要的情感类型。
            3. 如果文本中的情感不明显或难以界定，请将其归类为中性。
            4. 确保你的分类是基于文本内容而非你个人的感受或偏见。
            User Prompt:
            [文本]
            {sentence}
            [请根据以上指示对每个文本进行情感分类，并说明你的分类理由]
            """
            response = generate_chat_completion(prompt)
            print(response)

            emotion_index = emotion_labels.index(response) if response in emotion_labels else emotion_labels.index("其他")
            emotions.append(emotion_index)

        data["emotion"] = emotions

    with open(args.output_file, "w", encoding="utf-8") as output_file:
        json.dump(json_data_list, output_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()

# python lebel.py input.json output.json
