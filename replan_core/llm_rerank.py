"""
LLM-based reranking for RePlan/DualR path predictions.
Supports both OpenAI API (GPT-4o-mini) and vLLM local models.

Usage:
    # Use GPT-4o-mini (OpenAI API)
    python llm_rerank.py --dataset webqsp --backend openai --model gpt-4o-mini

    # Use local vLLM server
    python llm_rerank.py --dataset CWQ --backend vllm --model meta-llama/Llama-3.1-8B-Instruct --vllm_url http://localhost:8001/v1
"""

import json
import os
import time
import argparse
from tqdm import tqdm

# Support both old (0.28.1) and new (>=1.0) openai API
try:
    from openai import OpenAI
    OPENAI_NEW_API = True
except ImportError:
    import openai
    OPENAI_NEW_API = False


def load_questions(dataset):
    """Load questions from dataset."""
    all_q = []

    if dataset == 'webqsp':
        filepath = '../data/webqsp/test_simple.json'
        with open(filepath, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                question = entry.get('question', '')
                if not question.endswith('?'):
                    question += '?'
                all_q.append(question)

    elif dataset == 'CWQ':
        filepath = '../data/CWQ/test_simple.json'
        with open(filepath, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                question = entry.get('question', '')
                all_q.append(question)

    return all_q


def load_ground_truth(dataset):
    """Load ground truth answers."""
    all_ta = []

    if dataset == 'webqsp':
        ta_file = '../data/webqsp/Webqsp.txt'
    elif dataset == 'CWQ':
        ta_file = '../data/CWQ/CWQ.txt'

    with open(ta_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            try:
                _, ta = line[0], line[1]
                ta = ta.strip()
            except:
                ta = 'null'
            all_ta.append(ta)

    return all_ta


def parse_path_file(path_file):
    """Parse path file to extract candidates, scores, and paths."""
    all_entities = []
    all_scores = []
    all_paths = []
    all_ids = {}

    with open(path_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        parts = line.split('\t')

        entities = []
        scores = []
        paths = []

        qid = int(parts[0])
        parts = parts[1:]

        for part in parts:
            if part == '':
                continue
            try:
                entity, score, path = part.split('|')
            except:
                print(f"Warning: failed to parse part: {part}")
                continue

            entities.append(entity)
            scores.append(float(score))

            # Clean path
            path = path.split(';')
            split_path = []
            for p in path:
                if 'self_loop' not in p and p != '' and p not in split_path:
                    split_path.append(p)
            split_path = ', '.join(split_path)
            paths.append(split_path)

        all_entities.append(entities)
        all_scores.append(scores)
        all_paths.append(paths)
        all_ids[qid] = i

    return all_entities, all_scores, all_paths, all_ids


def create_prompt(question, candidates, scores, paths, top_k=4):
    """Create prompt for LLM reranking."""
    # System prompt
    system_prompt = (
        "Given a question, and the reference answers with their correct probabilities "
        "and associated retrieved knowledge graph triples (entity, relation, entity) as related facts, "
        "you are asked to answer the question with this information and your own knowledge. "
        "If the reference answers contain the correct answer, please output the label and content of the answer; "
        "If not, please answer the question based on your own knowledge. "
        "Please end your reply with 'The answer is ***'."
    )

    # Build reference answers
    labels = ['A', 'B', 'C', 'D', 'E']
    ref_parts = []
    for idx in range(min(top_k, len(candidates))):
        label = labels[idx]
        entity = candidates[idx]
        score = scores[idx]
        path = paths[idx]
        ref_parts.append(
            f"{label}. {entity} (correct probability: {score:.3f}) {{relevant facts: {path}}}"
        )

    reference = ' '.join(ref_parts)
    user_prompt = f"Question: {question}\nReference answer: {reference} Answer:"

    return system_prompt, user_prompt


def call_llm(client, model, system_prompt, user_prompt, temperature=0.0):
    """Call LLM API. Supports both old (0.28.1) and new (>=1.0) openai."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        if OPENAI_NEW_API:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature
            )
            output = response.choices[0].message.content
        else:
            response = openai.ChatCompletion.create(
                model=model, messages=messages, temperature=temperature
            )
            output = response['choices'][0]['message']['content']
        return output
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return 'NULL'


def extract_answer_from_response(response, candidates, all_ids, qid):
    """Extract answer from LLM response."""
    if response == 'NULL':
        return response

    # Clean response
    response = response.replace("\n", "  ")

    # Try to extract label (A/B/C/D/E)
    s = response
    index_a = s.find('A. ')
    index_b = s.find('B. ')
    index_c = s.find('C. ')
    index_d = s.find('D. ')
    index_e = s.find('E. ')

    if qid not in all_ids:
        return response

    i = all_ids[qid]

    # Determine which label was selected
    if 0 <= index_a and (index_b == -1 or index_b > index_a) and (index_c == -1 or index_a < index_c) and (index_d == -1 or index_a < index_d):
        return 'A. ' + candidates[i][0].lower()
    elif 0 <= index_b and (index_a == -1 or index_a > index_b) and (index_c == -1 or index_b < index_c) and (index_d == -1 or index_b < index_d):
        return 'B. ' + candidates[i][1].lower()
    elif 0 <= index_c and (index_a == -1 or index_a > index_c) and (index_b == -1 or index_b > index_c) and (index_d == -1 or index_c < index_d):
        return 'C. ' + candidates[i][2].lower()
    elif 0 <= index_d and (index_a == -1 or index_a > index_d) and (index_b == -1 or index_b > index_d) and (index_c == -1 or index_d < index_c):
        return 'D. ' + candidates[i][3].lower()
    else:
        return response.lower()


def evaluate(all_answers, all_answer_ids, all_ta, all_entities, all_ids):
    """Evaluate Hit@1."""
    check = []
    n_null = 0

    for i in range(len(all_answers)):
        answer = all_answers[i]
        qid = all_answer_ids[i]
        ta = all_ta[qid]

        if ta == 'null':
            n_null += 1
            check.append(0)
            continue

        ta_list = ta.split('|')
        flag = 0

        for oneta in ta_list:
            if oneta.lower() in answer.lower():
                check.append(1)
                flag = 1
                break

        if flag == 0:
            check.append(0)

    hit1 = sum(check) / (len(check) - n_null)
    return hit1, check


def main():
    parser = argparse.ArgumentParser(description="LLM reranking for RePlan/DualR")
    parser.add_argument('--dataset', type=str, required=True, choices=['webqsp', 'CWQ'])
    parser.add_argument('--backend', type=str, required=True, choices=['openai', 'vllm'])
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--vllm_url', type=str, default='http://localhost:8001/v1')
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--path_file', type=str, default=None, help='Path to path file (default: results/{dataset}-test-path.txt)')
    parser.add_argument('--output', type=str, default=None, help='Output file (default: results/{dataset}_llm_rerank.jsonl)')
    parser.add_argument('--top_k', type=int, default=4, help='Number of candidates to show to LLM')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--sleep', type=float, default=0.1, help='Sleep time between API calls')

    args = parser.parse_args()

    # Set default paths
    if args.path_file is None:
        args.path_file = f'results/{args.dataset}-test-path.txt'
    if args.output is None:
        args.output = f'results/{args.dataset}_llm_rerank.jsonl'

    # Initialize OpenAI client
    client = None  # only used for new API
    if OPENAI_NEW_API:
        if args.backend == 'openai':
            api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set --api_key or OPENAI_API_KEY env var")
            client = OpenAI(api_key=api_key)
        else:  # vllm
            client = OpenAI(base_url=args.vllm_url, api_key="EMPTY")
    else:
        # Old openai 0.28.x
        if args.backend == 'openai':
            api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set --api_key or OPENAI_API_KEY env var")
            openai.api_key = api_key
            openai.api_base = "https://api.openai.com/v1"
        else:  # vllm
            openai.api_key = "EMPTY"
            openai.api_base = args.vllm_url

    print(f"Using {'OpenAI API' if args.backend == 'openai' else 'vLLM at ' + args.vllm_url} with model: {args.model}")

    # Load data
    print("Loading data...")
    all_q = load_questions(args.dataset)
    all_ta = load_ground_truth(args.dataset)
    all_entities, all_scores, all_paths, all_ids = parse_path_file(args.path_file)

    print(f"Loaded {len(all_q)} questions, {len(all_ids)} with paths")

    # Run LLM reranking
    all_answers = []
    all_answer_ids = []

    # Remove existing output file if exists
    if os.path.exists(args.output):
        os.remove(args.output)

    for qid in tqdm(range(len(all_q)), desc="LLM reranking"):
        question = all_q[qid]

        if qid in all_ids:
            i = all_ids[qid]
            system_prompt, user_prompt = create_prompt(
                question,
                all_entities[i],
                all_scores[i],
                all_paths[i],
                top_k=args.top_k
            )
        else:
            system_prompt = "You are a helpful assistant."
            user_prompt = f"Question: {question} Answer:"

        # Call LLM
        response = call_llm(client, args.model, system_prompt, user_prompt, args.temperature)

        # Extract answer
        answer = extract_answer_from_response(response, all_entities, all_ids, qid)

        all_answers.append(answer)
        all_answer_ids.append(qid)

        # Save to file
        with open(args.output, 'a', encoding='utf-8') as f:
            data = {
                'id': qid,
                'question': question,
                'answer': answer,
                'raw_response': response
            }
            f.write(json.dumps(data) + '\n')

        time.sleep(args.sleep)

    # Evaluate
    print("\nEvaluating...")
    hit1, check = evaluate(all_answers, all_answer_ids, all_ta, all_entities, all_ids)

    print(f"\nResults:")
    print(f"Hit@1: {hit1:.4f}")

    # Save evaluation results
    eval_output = args.output.replace('.jsonl', '_eval.json')
    with open(eval_output, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'backend': args.backend,
            'model': args.model,
            'hit1': hit1,
            'total': len(check),
            'correct': sum(check)
        }, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"Evaluation saved to {eval_output}")


if __name__ == "__main__":
    main()
