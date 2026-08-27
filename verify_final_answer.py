#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import torch
import re
import json
from math_verify import LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig, parse
from math_verify import verify
from lcb_runner.evaluation.testing_util import run_test
from lcb_runner.utils.extraction_utils import extract_test_output_code
import multiprocessing as mp
import tempfile
from hendryck_cleanup import *
import os
import shutil
import base64
import zlib
import pickle

class CrossEncoderClient():
    def __init__(self, model_path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    def run(self,answer:str, gold_standard:str, threshold:float):
        print(f"Answer: {answer}; type: {type(answer)}")
        print(f"Gold: {gold_standard}; type: {type(gold_standard)}")
        test_features = self.tokenizer(answer, gold_standard, return_tensors='pt')
        print(f"Length without truncation: {len(test_features)}")
        features = self.tokenizer(answer, gold_standard,return_tensors='pt', truncation=True)
        print(f"Length with truncation: {len(features)}")
        with torch.no_grad():
            scores = self.model(**features).logits
        probs = torch.softmax(scores, dim=1)[0]
        label_mapping = ['contradiction', 'entailment', 'neutral']
        entailment_prob = probs[1].item()
        predicted_idx = probs.argmax().item()

        if predicted_idx == 1 and entailment_prob >= threshold:
            return True
        return False
        """for i, score_max in enumerate(scores.argmax(dim=1)):
            if label_mapping[score_max] == 'entailment' and scores[i][score_max]>=threshold:
                return True
        return False"""
    
def string_matching(answer:str, gold_standard:str, letter:str):
    print("--------------------------String matching-------------------------------")
    print(f"Answer: {answer}")
    print(f"Gold: {gold_standard}")
    print(f"Letter: {letter}")
    if "boxed{" in answer:
        extracted_answer = answer.split("boxed{")[1].split("}")[0].lower().replace('\n','').replace(' ','')
        lower_gold = gold_standard.lower().replace('\n','').replace(' ','')
        print(f"Extracted answer: {extracted_answer}")
        if lower_gold in extracted_answer:
            return True
        if letter is not None and letter.lower() in extracted_answer:
            return True
    else:
        extracted_answer = answer
        print(f"Extracted answer: {extracted_answer}")
        if gold_standard in extracted_answer:
            return True
        return False
    
def string_matching2(answer:str, gold_standard:str, letter:str):
    if "boxed{" in answer:
        parts = answer.split("boxed{")[-1]
        extracted_answer = parts.rsplit("}", 1)[0]
        lb = extracted_answer.count("{")
        rb = extracted_answer.count("}")
        print(f"Number of left brackets: {lb}")
        print(f"Number of right brackets: {rb}")
    else:
        extracted_answer = answer
    
    def clean(s):
        if not s:
            return ""
        s = s.lower().replace('\n','').replace(' ','')
        return s.strip('.')
    
    clean_extracted = clean(extracted_answer)
    clean_gold = clean(gold_standard)
    clean_letter = letter.lower() if letter else None
    print(f"Extracted: {clean_extracted}")
    print(f"Gold: {clean_gold}")
    print(f"Letter: {clean_letter}")

    if clean_letter:
        if clean_extracted==clean_letter:
            return True
        elif len(answer)>10 and (answer.strip().lower().endswith(f"is{clean_letter}") or answer.strip().lower().endswith(f"answer:{clean_letter}")):
            return True
    
    if clean_extracted==clean_gold:
        return True
    elif clean_gold in clean_extracted and len(clean_gold)>1:
        print("Exception used!")
        return True
    return False

def string_matching3(answer:str, gold_standard:str, letter:str):
    print(f"Letter: {letter}")
    extracted_answer = None
    patterns = [
        r"\\boxed\{([A-Z])\}",
        r"(?:answer)\s*[:\s]*\(?([A-Z])\)?",
        r"\b([A-Z])\b(?:\s*)$"
    ]
    for i,p in enumerate(patterns):
        match = re.search(p, answer, re.IGNORECASE)
        if match:
            print(f"Pattern used: {p}, n°{i}")
            extracted_answer = match.group(1).upper()
            break
    if extracted_answer and extracted_answer==letter:
        return True
    print(f"Extracted answer: {extracted_answer}")
    return False

def clean_answer(answer:str)->str:
    answer = re.sub(f"<think>.*?</think>", "", answer, flags=re.DOTALL)
    boxed_match = re.findall(r"\\boxed{(.*?)\}", answer)
    if boxed_match:
        return boxed_match[-1].strip()
    return answer.strip()

def grade_math(answer:str, gold_standard:str):
    config = [
        LatexExtractionConfig(),
        ExprExtractionConfig()
    ]
    cleaned_answer = last_boxed_only_string(answer)
    result = parse(cleaned_answer, config)
    parsed_gold = parse(gold_standard, config)
    is_correct = verify(parsed_gold, result)
    rm_ans = remove_boxed(cleaned_answer)
    equiv = is_equiv(rm_ans, gold_standard)
    print(f"Answer: {rm_ans}")
    print("\n")
    #print(f"Result: {type(result)}")
    print(f"Gold: {gold_standard}")
    return equiv

def worker(queue, samp, test_code, temp_dir):
    try:
        os.chdir(temp_dir)
        results, metadata = run_test(samp, test=test_code)
        queue.put((results, metadata))
    except Exception as e:
        queue.put(([False], str(e)))

def run_test_isolated(sample, code, timeout=6):
    context = mp.get_context("spawn")

    q = context.Queue()
    temp_dir = tempfile.mkdtemp()
    p = context.Process(target=worker, args=(q, sample, code, temp_dir))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        shutil.rmtree(temp_dir, ignore_errors=True)
        return [False], {"error":"Timeout or Infinite Loop"}
    
    if not q.empty():
        return q.get()
    else:
        return [False], {"error":"Process crashed silently"}

def grade_lcb(answer:str, sample):
    code = extract_test_output_code(answer)
    # Now we get the private test cases from the encrypted string
    inputs = []
    outputs = []
    decoded_bytes = base64.b64decode(sample["ptc"])
    decompressed_bytes = zlib.decompress(decoded_bytes)
    readable_cases = pickle.loads(decompressed_bytes)
    ptc = json.loads(readable_cases)
    for x in ptc:
        if len(x)>0:
            inputs.append(x["input"])
            outputs.append(x["output"])
        else:
            print(f"Empty x: {x}")
            continue
    s = {'input_output':json.dumps({'inputs':inputs, 'outputs':outputs, 'fn_name':sample["fn_name"]})}
    print(f"Code: {code}\n")
    if not code:
        print(f"No code found.")
        return False
    #str_sample = json.dumps(sample)
    results, metadata = run_test_isolated(s, code)
    print(f"Results: {results}, Metadata: {metadata}")
    # Let's assume that results is a list of bools
    if not results:
        return False
    for result in results:
        if result is not True:
            return False
    return True
    



def grade_answers(answers:List[str], gold_standard:List[str|Dict], letters:List[str|None], model_path:str, threshold:float, verbose:bool, dataset_n:int):
    if dataset_n<2:
        labels = [string_matching3(answer, gold,letter) for answer, gold, letter in zip(answers, gold_standard, letters)]
    elif dataset_n==2:  #LiveCodeBench
        labels = [grade_lcb(answer, gold) for answer, gold in zip(answers, gold_standard)]
    else:  # MATH
        labels = [grade_math(answer, gold) for answer, gold in zip(answers, gold_standard)]
    return labels
    """trimmed_answers = [
        answer[:-int(min(max(len(answer)//10, 500), len(answer)-1))] 
        if len(answer) > 1000 else answer 
        for answer in answers
    ]
    cross_client = CrossEncoderClient(model_path=model_path)
    labels = [cross_client.run(answer, g, threshold) for answer, g in zip(trimmed_answers, gold_standard)]"""
        