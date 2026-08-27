#!/usr/bin/env python3
from datasets import load_dataset
from typing import List, Tuple, Dict, Any
from random import shuffle
import string
import re
import numpy as np
import os
from pathlib import Path
import json
from argparse import ArgumentParser
import numpy as np
from sklearn.model_selection import train_test_split
from verify_final_answer import string_matching3, grade_lcb, grade_math
from vllm import LLM, SamplingParams

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def split_samples(samples:Dict, n:int, directory_path:str, split:str, dataset:str, model_name:str, label:int)->None:
    """
    Split the samples into different files, with a specified maximum number of samples per file.

    Args:
        samples (Dict): The samples.
        n (int): The maximum number of samples per file.
        directory_path (str): The path of the directory inside which the files are created.
        split (str): The name of the dataset split (train, eval, test).

    Returns:
        None
    """
    lcots_dir = os.path.join(directory_path,"lcots/")
    if not os.path.isdir(lcots_dir):
        os.mkdir(lcots_dir)
    string_label = "false" if label == 0 else "true"
    length = len(samples)
    nb_files = length // n + 1
    for i in range(nb_files):
        s = samples[i * n:min((i + 1) * n, length - 1)]
        with open(os.path.join(lcots_dir, f"{split}_{dataset}_{model_name}_{string_label}_{i}.jsonl"), "w+") as f:
            for sample in s:
                f.write(json.dumps(sample) + '\n')


def load_MMLU_pro(dataset_path:str)->List[Dict[str,str]]:
    """
        Load the samples of MMLU-Pro from the HuggingFace dataset into a jsonl file, with split steps and corresponding embeddings.
    
        Args:
            dataset_path (str): The HF repo name or the local path to the dataset.
        
        Returns:
            List[Dict[str,str]]
        """
    dataset = load_dataset(dataset_path)
    test_split = dataset["test"]
    samples = [{"id":sample["question_id"],"query":sample["question"], "prompt":sample["question"]+"\nPossible answers: "+"\n".join([f"{letters[i]}:{option}" for i, option in enumerate(sample["options"])])+"Put the final answer in the following format: \\boxed\{answer\}", "answer": sample["options"][int(sample["answer_index"])], "letter":letters[int(sample["answer_index"])], "subject":sample["category"]}for sample in test_split]
    return samples

def load_live_code_bench(dataset_path:str)->List[Dict[str,str]]:
    """
    Load the samples of LiveCodeBench-v5 from the HuggingFace dataset into a jsonl file, with split steps and corresponding embeddings.

    Args:
        dataset_path (str): The HF repo name or the local path to the dataset.
    
    Returns:
        List[Dict[str,str]]
    """
    dataset = load_dataset(dataset_path)
    train_split = dataset["train"]
    samples = []
    for sample in train_split:
        fn_name = None
        inputs = []
        outputs = []
        verification_info = json.loads(json.loads(sample["verification_info"])["ground_truth"])
        for x in verification_info:
            inputs.append(x["input"])
            outputs.append(x["output"])
        function = verification_info[0]["metadata"]["func_name"]
        if function!="null":
            fn_name=function
        samples.append({"id":sample["problem_id"], "query": sample["prompt"], "input_output":json.dumps({"inputs":inputs, "outputs":outputs, "fn_name":fn_name})})

    return samples

def load_MATH(dataset_path:str)->List[Dict[str,str]]:
    """
        Load the samples of MATH from the HuggingFace dataset into a jsonl file, with split steps and corresponding embeddings.
    
        Args:
            dataset_path (str): The HF repo name or the local path to the dataset.
        
        Returns:
            List[Dict[str,str]]
    """
    dataset = load_dataset(dataset_path)
    main = dataset["train"]
    samples = [{"id":sample["unique_id"], "query":sample['problem'], "answer":sample["answer"], "difficulty":sample["level"]} for sample in main]
    return samples

def load_GPQA(dataset_path:str)->List[Dict[str,str]]:
    """
        Load the samples of GPQA from the HuggingFace dataset into a jsonl file, with split steps and corresponding embeddings.
    
        Args:
            dataset_path (str): The HF repo name or the local path to the dataset.
        
        Returns:
            List[Dict[str,str]]
    """
    # For this dataset, we need the path to the csv file. Here: ".cache/huggingface/hub/datasets--Idavidrein--gpqa/snapshots/633f5ee89ab8ad4522a9f850766b73f62147ffdd/gpqa_main.csv"
    #main = load_dataset("csv", data_files=os.path.join(parent_dir, ".cache/huggingface/hub/datasets--Idavidrein--gpqa/snapshots/633f5ee89ab8ad4522a9f850766b73f62147ffdd/gpqa_main.csv"))["train"]
    if len(dataset_path.split("/")) == 2:
        main = load_dataset(dataset_path)
    else:
        main = load_dataset("csv", data_files=dataset_path)["train"]
    samples = []
    for sample in main:
        correct_answer = sample["Correct Answer"]
        options = [correct_answer, sample["Incorrect Answer 1"], sample["Incorrect Answer 2"], sample["Incorrect Answer 3"]]
        np.random.shuffle(options)
        correct_idx = options.index(correct_answer)
        correct_letter = letters[correct_idx]
        formatted_options = "\n".join([f"{letters[i]}: {options[i]}" for i in range(4)])
        prompt = sample["Question"]+"\nPossible answers:\n"+formatted_options+"\nPut the final answer in the following format: \\boxed\{answer\}"
        samples.append({"id":sample["Record ID"],"query":sample["Question"],"prompt":prompt, "answer":correct_answer, "letter":correct_letter, "subject":sample["High-level domain"], "subdomain":sample["Subdomain"]})
    return samples

def find_gpqa_csv_file_path(hf_cache_dir:str)->str:
    """
    Find the csv file containing the GPQA dataset inside the HuggingFace directory.

    Args:
        hf_cache_dir (str): Path to the HuggingFace cache directory.

    Returns:
        str
    """
    dataset_dir = os.path.join(hf_cache_dir,"datasets--Idavidrein--gpqa/")

    print(f"dataset_dir: {dataset_dir}")
    for root, dirs, files in os.walk(dataset_dir):
        print(f"root: {root}, dirs: {dirs}, files: {files}")
        for file in files:
            if ".csv" in file:
                return os.path.join(root,file)

def find_model_path(model_dir:str):
    """
    Finds the path containing the model files.

    Args:
        model_dir (str): Path to the model directory.

    Returns:
        str
    """
    snap_path = os.path.join((model_dir),"snapshots/")
    name = list(os.listdir(snap_path))[0]
    return os.path.join(snap_path, name)

def initialize_model_with_vLLM(model_id:str)->Tuple[LLM, SamplingParams, Any, int]:
    """
    Initializes the vLLM model that will produce the LCoTs.

    Args:
        model_id (str): Path to the model.

    Returns:
        Tuple[LLM, SamplingParams, Any, int]
    """
    llm = LLM(
        
    model=model_id,
        
    trust_remote_code=True,
        
    quantization="fp8",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    kv_cache_dtype="fp8",
    enable_chunked_prefill=True
    )
    max_model_len = llm.llm_engine.model_config.max_model_len
    if "Llama" in model_id:
        stop = ["<|end_of_text|>", "<|eot_id|>"]
    elif "Qwen" in model_id:
        stop = ["<|im_end|>", "<|endoftext|>"]
    else:
        stop = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
    params = SamplingParams(max_tokens=max_model_len, temperature=0.6, top_p=0.95, stop=stop)
    tokenizer = llm.get_tokenizer()
    return llm, params, tokenizer, max_model_len


def run_with_vLLM(llm:LLM, params:SamplingParams, queries:List[str], tokenizer:Any)->List[str]:
    """
    Produces a batch of LCoTs from a batch of queries.

    Args:
        llm (LLM): The LLM that generates the LCoTs.
        params (SamplingParams): The model's generation parameters.
        queries (List[str]): The batch of prompts passed to the model.
        tokenizer (Any): The tokenizer of the model.

    Returns:
        List[str]
    """
    formatted_queries = []
    for query in queries:
        message = [{"role":"user","content":query}]
        prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        formatted_queries.append(prompt)
    outputs = llm.generate(formatted_queries, params)
    answers = [output.outputs[0].text for output in outputs]
    return answers

def generate_lcots(samples:List[Dict], path_to_model:str, iterations:int, dataset:str)->List[str]:
    """
    Generates LCoTs on the samples of one dataset using one LRM.

    Args:
        samples (List[Dict]): The samples containing the prompt.
        path_to_model (str): The path to the LRM to generate the LCoTs.
        iterations (int): The number of time each prompt must be used.

    Returns:
        List[str]
    """
    prompts = []
    multiplied_samples = []
    golds = []
    letters = []
    for sample in samples:
        for i in range(iterations):
            prompts.append(sample["prompt"] if "prompt" in sample else sample["query"])
            multiplied_samples.append(sample)
            golds.append(sample["answer"])
            letters.append(sample["letter"] if "letter" in sample else None)
    model, params, tokenizer, max_model_len = initialize_model_with_vLLM(path_to_model)
    lcots = run_with_vLLM(llm=model, params=params, queries=prompts, tokenizer=tokenizer)
    print(f"Right after run_with_vLLM: {len(lcots)}")
    if dataset == "mmlu" or dataset == "gpqa":
        labels = [string_matching3(answer=answer, gold_standard=gold, letter=letter) for answer, gold, letter in zip(lcots, golds, letters)]
    elif dataset == "math":
        labels = [grade_math(answer=answer, gold_standard=gold) for answer, gold in zip(lcots, golds)]
    elif dataset == "lcb":
        labels = [grade_lcb(answer=lcot, sample=sample) for lcot, sample in zip(lcots,samples)]
    print(f"Number of labels: {len(labels)}")
    for lcot, label, sample in zip(lcots, labels, multiplied_samples):
        sample["lcot"] = lcot
        sample["label"] = 1 if label is True else 0
    
    return multiplied_samples


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--data-dir", type=str, default="data/", help="The directory where all the data of the project is stored.")
    parser.add_argument("-n", "--nb-samples-per-file",type=int, default=15, help="The number of samples stored in each file.")
    parser.add_argument("-f", "--hf-dir", type=str, default=None, help="The directory in which the HugginFace models and datasets are stored. By default, $HOME/.cache/huggingface/hub/ is used.")
    parser.add_argument("-r", "--from-remote", action="store_true", help="Use if you want to download the datasets directly from the HuggingFace API instead of having them stored on your machine.")
    parser.add_argument("-d", "--dataset", type=str, choices=["math","mmlu","gpqa","lcb"], help="The name of the dataset.")
    parser.add_argument("-m", "--model", type=str, choices=["qwen","llama","qwq"],help="The name of the model.")
    parser.add_argument("-l", "--limit-nb-samples", type=int, help="The maximum number of samples needed to produce the LCoTs.")
    parser.add_argument("-i", "--iterations", type=int, help="The number of samples produced for each prompt.")
    args = parser.parse_args()
    print(f"Args: {args}")
    if args.hf_dir is None:
        home = Path.home()
        hf_dir = os.path.join(home,".cache/huggingface/hub/")
    else:
        hf_dir = args.hf_dir

    print(f"hf_dir: {hf_dir}")
    # Load the dataset samples

    if args.dataset == "gpqa":
        dataset_path = "Idavidrein/gpqa" if args.from_remote else find_gpqa_csv_file_path(hf_dir)
        samples = load_GPQA(dataset_path)
    elif args.dataset == "math":
        dataset_path = "simplescaling/openaimath" if args.from_remote else os.path.join(hf_dir,"datasets--simplescaling--openaimath/")
        samples = load_MATH(dataset_path)
    elif args.dataset == "mmlu":
        dataset_path = "TIGER-Lab/MMLU-Pro" if args.from_remote else os.path.join(hf_dir,"datasets--TIGER-Lab--MMLU-Pro/")
        samples = load_MMLU_pro(dataset_path)
    elif args.dataset == "lcb":
        dataset_path = "PrimeIntellect/LiveCodeBench-v5" if args.from_remote else os.path.join(hf_dir, "datasets--PrimeIntellect--LiveCodeBench-v5/")
        samples = load_live_code_bench(dataset_path)
    else:
        raise ValueError(f"Not a valid dataset option: {args.dataset}")

    shuffle(samples)

    samples = samples[:args.limit_nb_samples]

    if args.model == "qwen":
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" if args.from_remote else find_model_path(os.path.join(hf_dir, "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/"))
    elif args.model == "llama":
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" if args.from_remote else find_model_path(os.path.join(hf_dir, "models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/"))
    elif args.model == "qwq":
        model_path = "Qwen/QwQ-32B" if args.from_remote else find_model_path(os.path.join(hf_dir, "models--Qwen--QwQ-32B/"))

    # Generate the LCoTs and place them inside the samples
    samples = generate_lcots(samples=samples, path_to_model=model_path, iterations=args.iterations, dataset=args.dataset)
    print(f"Number of samples right after generate_lcots: {len(samples)}")
    # Split the samples between those labeled as True and those labeled as False
    true_samples = [sample for sample in samples if sample["label"] == 1]
    false_samples = [sample for sample in samples if sample["label"] == 0]

    # Shuffle and cap the number at 334 for each
    shuffle(true_samples)
    shuffle(false_samples)
    if len(true_samples) > 334:
        true_samples = true_samples[:334]
    if len(false_samples) > 334:
        false_samples = false_samples[:334]

    # Split into train, test, and eval
    true_train_eval_set, true_test_set = train_test_split(np.array(true_samples), test_size=0.2, random_state=42)
    false_train_eval_set, false_test_set = train_test_split(np.array(false_samples), test_size=0.2, random_state=42)
    true_train_set, true_eval_set = train_test_split(np.array(true_train_eval_set), test_size=0.1, random_state=42)
    false_train_set, false_eval_set = train_test_split(np.array(false_train_eval_set), test_size=0.1, random_state=42)


    # Split each of the splits into several files to facilitate the graph construction function in the next step.
    split_samples(samples=true_train_set, n=args.nb_samples_per_file, directory_path=args.data_dir, split="train", dataset=args.dataset, model_name=args.model, label=1)
    split_samples(samples=true_test_set, n=args.nb_samples_per_file, directory_path=args.data_dir, split="test", dataset=args.dataset, model_name=args.model, label=1)
    split_samples(samples=true_eval_set, n=args.nb_samples_per_file, directory_path=args.data_dir, split="eval", dataset=args.dataset, model_name=args.model, label=1)
    split_samples(samples=false_train_set, n=args.nb_samples_per_file, directory_path=args.data_dir, split="train", dataset=args.dataset, model_name=args.model, label=0)
    split_samples(samples=false_test_set, n=args.nb_samples_per_file, directory_path=args.data_dir, split="test", dataset=args.dataset, model_name=args.model, label=0)
    split_samples(samples=false_eval_set, n=args.nb_samples_per_file, directory_path=args.data_dir, split="eval", dataset=args.dataset, model_name=args.model, label=0)

    
    