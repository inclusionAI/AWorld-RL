import transformers
import torch
import requests
import json
import re
import string
import pandas as pd
import time

test_data_name = 'HotpotQA'
model_path = "../model/hotpotqa_train_filter_mq-RAG-R1-qwen2.5-7b-sft-ppo"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

curr_eos = [151645, 151643]  # for Qwen2.5 series models
curr_search_template = '{output_text}\n<information>\n{search_results}\n</information>\n'

# Prepare the message
prompt_template = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query_1,query_2 </search> and it will return the top searched results for each query between <information> and </information>. \
You can search as many times as you want, using up to three queries each time. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")


# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in
                           target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False


def load_test_data(test_data_name):
    test_data = []

    if test_data_name == 'HotpotQA':
        data_path = '../data/HotpotQA/hotpot_dev_fullwiki_v1.json'
        with open(data_path, "r") as f:
            raw_test_data = json.load(f)
            for data in raw_test_data:
                question = data['question']
                answer = data['answer']
                test_data.append({
                    'question': question,
                    'answer': answer
                })
    elif test_data_name == '2WikiMultihopQA':
        data_path = '../data/2WikiMultihopQA/dev.json'
        with open(data_path, "r") as f:
            raw_test_data = json.load(f)
            for data in raw_test_data:
                question = data['question']
                answer = data['answer']
                test_data.append({
                    'question': question,
                    'answer': answer
                })
    elif test_data_name == 'Musique':
        data_path = '../data/Musique/musique_ans_v1.0_dev.jsonl'
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                question = data['question']
                answer = data['answer']
                test_data.append({
                    'question': question,
                    'answer': answer
                })
    elif test_data_name == 'Bamboogle':
        data_path = '../data/Bamboogle/Bamboogle.csv'
        df = pd.read_csv(data_path, encoding="utf-8")
        for row in df.itertuples(index=False):
            question = row.Question
            answer = row.Answer
            test_data.append({
                'question': question,
                'answer': answer
            })
    else:
        pass

    return test_data


def get_query(text):
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None


def get_answer(text):
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def search(query_list: list):
    payload = {
        "queries": query_list,
        "topk": 3,
        "return_scores": True
    }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']

    def _passages2string(query_list, results):
        retrieval_result = {}
        document_list = []
        for documents in results:
            format_reference = ''
            for idx, doc_item in enumerate(documents):
                content = doc_item['document']['contents']
                format_reference += f"Doc {idx + 1}: {content}\n"
            document_list.append(format_reference.strip())
        retrieval_result['query'] = query_list
        retrieval_result['documents'] = document_list

        return json.dumps(retrieval_result, indent=2)

    return _passages2string(query_list, results)


# Initialize the stopping criteria
target_sequences = ["</search>", " </search>", "?</search>", '"</search>', "'</search>", ")</search>", "</search>\n",
                    " </search>\n", "?</search>\n", '"</search>\n', "'</search>\n", ")</search>\n", "</search>\n\n",
                    " </search>\n\n", "?</search>\n\n", '"</search>\n\n', "'</search>\n\n", ")</search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

print('\n\n################# [Start Reasoning + Searching] ##################\n\n')

test_data = load_test_data(test_data_name)
result = []
acc = 0
retrieval_count_sum = 0
start_time = time.time()
retrieval_max_count = 5

for index,data in enumerate(test_data):
    retrieval_count = 0
    question = data['question']
    gold_answer = data['answer']
    prompt = prompt_template.format(question=question)
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                               tokenize=False)

    # Encode the chat-formatted prompt and move it to the correct device
    while retrieval_count < retrieval_max_count:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)

        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            prompt += output_text
            break

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            search_query_list = [query.strip() for query in tmp_query.split(',')]
            search_query_list = [query for query in search_query_list if query != '']
            search_results = search(search_query_list)
        else:
            search_results = ''

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        retrieval_count += 1
    retrieval_count_sum += retrieval_count
    predict_answer = get_answer(prompt)
    reward = float(normalize_answer(predict_answer) == normalize_answer(gold_answer))

    print(f'process: {index}/{len(test_data)}')
    print(f'[all prompt]: \n{prompt}')
    print(f'[gold_answer]: {gold_answer}')
    print(f'[predict_answer]: {predict_answer}')
    print(f'[reward]: {reward}')
    print('-' * 50)

    acc += reward
    result.append({
        'question': question,
        'gold_answer': gold_answer,
        'predict_answer': predict_answer,
        'prompt': prompt,
        'reward': reward,
        'rc': retrieval_count
    })

accuracy = acc / len(test_data)
print(f'[Accuarcy]: {accuracy}')

end_time = time.time()
time_cost = end_time - start_time
time_per_sample = time_cost / len(test_data)
print(f'[Time cost]: {time.time() - start_time} s')
print(f'[Time cost per sample]: {time_per_sample} s')

retrieval_count_per_sample = retrieval_count_sum / len(test_data)
print(f'[Retrieval count per sample]: {retrieval_count_per_sample}')

result_path = test_data_name + '.jsonl'
with open(result_path, 'w') as f:
    for item in result:
        f.write(json.dumps(item) + '\n')
