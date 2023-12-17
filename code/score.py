import json
import jieba
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from tqdm import tqdm
import argparse

rouge = Rouge()

# Function to get QAs from a file
def get_qas(p, key):
    with open(p, 'r', encoding="utf-8") as f:
        source_data = json.load(f)
    return [item[key] for item in source_data]

# Function to calculate ROUGE score
def caculate_rouge_score(generated, ground_truths):
    try:
        rouge_l = 0
        for ground_truth in ground_truths:
            score = rouge.get_scores(' '.join(list(generated)), ' '.join(list(ground_truth)))
            rouge_l = max(score[0]['rouge-l']['r'], rouge_l)
        return rouge_l
    except:
        return 0

# Function to calculate BLEU score
def caculate_bleu_score(generated, ground_truths):
    ground_truths = [list(text) for text in ground_truths]
    generated = list(generated)
    return sentence_bleu(ground_truths, generated)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate BLEU and ROUGE scores for generated texts.")
    parser.add_argument("target_file")
    parser.add_argument("generated_file")
    parser.add_argument("result_file")
    return parser.parse_args()

def main():
    args = parse_arguments()

    result = []
    t_qas = get_qas(args.target_file, "generated")
    g_qas = get_qas(args.generated_file, "predict")

    average_score_dict = {
        "specific": [],
    }

    for g_qa, t_qa in zip(g_qas, t_qas):
        average_score_dict["specific"].append(caculate_rouge_score(g_qa, t_qa))

    # Calculate average score
    for item in average_score_dict.keys():
        average_score_dict[item] = sum(average_score_dict[item]) / len(average_score_dict[item])
    result.append("rouge_l @ raw:    " + str(average_score_dict))

    with open(args.result_file, "w", encoding="utf-8") as f:
        for line in result:
            print(line)
            f.write(line + "\n")

if __name__ == "__main__":
    main()

# python score.py target.json generated.json result.txt
