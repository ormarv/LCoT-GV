#!/usr/bin/env python3
from typing import List, Dict, Tuple
import networkx as nx
import itertools
from tqdm import tqdm
import random
from language_models import  NLI_client, EmbeddingModel
import re
from argparse import ArgumentParser
import os
from pathlib import Path
import json
import string
from io import TextIOWrapper
import torch

def length_regularity(steps:List[str]):
    """
    Counts the number of steps in each length bucket.

    Args:
        steps (List[str]): The list of reasoning steps.

    Returns:
        Dict[int,int]
    """
    lengths = {i*10:0 for i in range(101)}
    for step in steps:
        q = len(step.split(' '))//10
        if q>=100:
            lengths[1000]+=1
        else:
            lengths[q*10]+=1
    print(f"lengths: {lengths}")
    return lengths

def contains_letters(chain:str)->bool:
    """
    Checks whether a string contains letters.

    Args:
        chain (str): The chain of characters to check.

    Returns:
        bool 
    """
    intersection = set(string.ascii_letters).intersection(set(chain))
    if len(intersection)>0:
        return True
    return False

def intelligent_split(lcot:str, n_first:int)->List[str]:
    """
    Splits a Long Chain-of-Thought using the specified number of keywords.

    Args:
        lcot (str): The Long Chain-of-Thought
        n_first: The number of keywords.

    Returns:
        List[str]
    """
    first_words = {}
    raw_steps = lcot.split("\n\n")
    for raw_step in raw_steps:
        words = raw_step.split(' ')
        if contains_letters(words[0]):
            fw = words[0].replace(',','')
            if fw.strip()=="I":
                fw = r'[\.;:\?\!\n]\s+I'
            if fw not in first_words:
                first_words[fw] = 0
            first_words[fw] += 1
    sorted_words = [k for k,_ in sorted(first_words.items(), key=lambda item: item[1], reverse=True)]
    keywords = sorted_words[:n_first]
    augmented_keywords = []
    for keyword in keywords:
        augmented_keywords.append(re.escape(keyword+" "))
        augmented_keywords.append(re.escape(keyword+","))
    string = '|'.join(augmented_keywords)
    steps = re.finditer(string,lcot)  # , flags=re.IGNORECASE
    split_indices = []
    for match in steps:
        start = match.start()
        if " I," in match.group() or " I " in match.group() or "\nI " in match.group() or "\nI," in match.group() and match.start()<len(string)-1:
            start+=1
        split_indices.append(start)
    start_indices = [0]+split_indices[:-1]
    end_indices = split_indices+[len(split_indices)]
    all_indices = zip(start_indices,end_indices)
    full_steps = []
    for (i,j) in all_indices:
        step = lcot[i:j]
        if len(step)>0:
            full_steps.append(step)
    return full_steps

def get_reduced_attachment_pool(last_node:int,leaves:Dict, main_branch:List[int], repeated_steps:List[int],div_factor:float, graph:nx.DiGraph)->List[int]:
    """
    Builds the list of potential parents.

    Args:
        last_node (int): the index of the last node inseted in the graph.
        leaves (Dict): the set of all nodes in the graph without children.
        main_branch (List[int]): the branch on which the last node was inserted.
        repeated_steps (List[int]): the list of indices of the steps that are repetitions of previous steps.
        div_factor (float): the percentage of one-child nodes from the main branch to keep as potential parents.
        graph (nx.DiGraph): the graph.

    Returns:
        List[int]
    """
    # leaves are a set of integers
    attachment_pool = set()
    for i,node in enumerate(main_branch):
        # the main branch doesn't contain a leaf
        positive_children = [edge for edge in graph.out_edges(node) if graph.get_edge_data(edge[0], edge[1])['relation']==1]
        if i >= len(main_branch)-30 or len(positive_children) >= 2:
            print(f"Adding node {node}, n°{i} in the main branch, to the pool.")
            attachment_pool.add(node)
    # we find and remove the former leaves
    intersection = attachment_pool.intersection(leaves)
    for node in intersection:
        if node in leaves:
            leaves.remove(node)
    for node in leaves:
        if repeated_steps is None or node not in repeated_steps:
            attachment_pool.add(node)
    # we add the current node to the leaves, for next time
    leaves.add(last_node)
    print('\n')
    l_attachment_pool = list(attachment_pool)
    l_attachment_pool.sort()
    print(f"The attachment pool for {last_node} is {l_attachment_pool}")
    print('\n')
    attachment_pool = list(attachment_pool)
    attachment_pool.sort(reverse=True)
    return attachment_pool

def get_path_content(path:List[int],steps:Dict[int,str])->str:
    """
    Takes a list of step indices and returns the corresponding text.

    Args:
        path (List[int]): The path in the graph.
        steps (Dict[int,str]): The complete list of steps.

    Returns:
        str
    """
    path_content = ""
    for node in path:
        path_content = path_content + steps[node]
    return path_content

def build_step_dictionaries_and_lists(steps:Dict[int,str])->Tuple[Dict[int,List[int]],Dict[int,int]]:
    """"
    Builds dictionaries that link repeated steps and the corresponding original step.

    Args:
        steps (Dict[int,str]): A dictionary containing each step index and the corresponding step text.
    
    Returns:
        Tuple[Dict[int,List[int]],Dict[int,int]]
    """
    or_to_cop = {}
    cop_to_or = {}
    for i, steptext in list(steps.items()):
        if i not in cop_to_or:
            print(f"i: {i}")
            copies = [j for j,s in list(steps.items()) if s == steptext and i!=j]
            print(f"copies: {copies}")
            if len(copies) > 70 or len(steptext)>50:
                or_to_cop[i] = copies
                for copy in copies:
                    cop_to_or[copy] = i
    return or_to_cop, cop_to_or

def groupc(steplist:List[int])->List[Tuple[int]]:
    """
    Groups a list of step indices by groups of continuous indices.

    Args:
        steplist (List[int]): List of step indices.

    Returns:
        List[Tuple[int]]
    """
    consecutive = []
    i = 0
    print(f"Length before sorting: {len(steplist)}")
    steplist.sort()
    print(f"Length after sorting: {len(steplist)}")
    while i < len(steplist):
        j = i
        while j < len(steplist) - 1 and steplist[j+1] == steplist[j]+1:
            j += 1
        consecutive.append(tuple(range(steplist[i],steplist[j]+1)))
        i = j+1
    return consecutive


def match_sequences_dicts(or_to_cop:Dict[int,List[int]], cop_to_or:Dict[int,int])->Tuple[Dict[Tuple[int],Tuple[int]],Dict[int,Tuple]]:
    """
    Matches sequences of repeated steps together, and individual steps to the sequence to which they belong.

    Args:
    or_to_cop (Dict[int,List[int]]): Dictionary linking each original step to the list of its copies.
    cop_to_or (Dict[int,int]): Dictionary linking each copy to its original step.

    Returns:
        Tuple[Dict[Tuple[int],Tuple[int]],Dict[int,Tuple]]
    """
    
    copseq_to_orseq = {}
    step_to_seq = {}
    orseq = groupc(list(or_to_cop.keys()))
    copseq = groupc(list(cop_to_or.keys()))
    print(f"orseq: {orseq}")
    print(f"copseq: {copseq}")
    for seq in copseq:
        for step in seq:
            step_to_seq[step] = seq
    print(f"Here is step_to_seq: {step_to_seq}")
    for o in orseq:
        print(f"o: {o}")
        start = o[0]
        end = o[-1]
        diff = end-start
        start_copies = or_to_cop[start]
        for sc in start_copies:
            c = tuple(range(sc,sc+diff+1))
            if c in copseq:
                copseq_to_orseq[c] = o
    print(f"copseq_to_orseq: {copseq_to_orseq}")
    return copseq_to_orseq, step_to_seq

def construct_graph(steps:Dict[int,str], nli_client:NLI_client, embedding_client:EmbeddingModel, threshold:float = 0.7, neg_threshold:float=-0.7, max_path_length_for_nli=5, k1:float=0.01, k2:float=0.02, wanted_features:Dict[str,int]=[], prune_coeff:float=0.33)->Tuple[nx.DiGraph,List[List[float]],torch.Tensor]:
    """
    Construct a reasoning graph from the list of the steps.
    Args:
    steps (Dict[int,str]): A dictionary that associates the name of the step (i.e. s_i) to its content
    nli_client (NLI_client): The Natural Language Inference model used to build the graph.
    embedding_client (EmbeddingModel): The sentence-transformers model used to create step embeddings.
    threshold (float): The classification score above which a relation is an entailment.
    neg_threshold (float): The classification score below which a relation is a contradiction.
    k1 (float): The percentage of first nodes that cannot have more than one parent.
    k2 (float): The percentage of first nodes that cannot have more than two parents.
    wanted_features (Dict[str,int]): The list (with index) of features selected by the user.
    prune_coeff (float): The proportion of single-child nodes in the main branch that is kept in the parent pool.

    Return:
    A graph, an incomplete features list, and steps embeddings (Tuple[nx.DiGraph,List[List[float]],torch.Tensor]).
    """
    q1 = int(k1*len(steps))
    q2 = int(k2*len(steps))
    graph = nx.DiGraph()
    paths = {}  # key: int (node index); value: List of paths
    new_paths = []
    main_branch = []
    added_steps = []
    leaves = set()
    original_parents = {}
    or_to_cop = None
    cop_to_or = None
    copseq_to_orseq = None
    step_to_seq = None
    if len(steps) >= 1000:
        or_to_cop, cop_to_or = build_step_dictionaries_and_lists(steps=steps)
        for o in or_to_cop:
            print(f"Example of an original step: {steps[o]}, n°{o}, its copies are: {or_to_cop[o]}")
        copseq_to_orseq, step_to_seq = match_sequences_dicts(or_to_cop, cop_to_or)
        print(f"Number of copies: {len(cop_to_or.keys())}")
        if len(cop_to_or.keys()) <= 50:
            cop_to_or = None
        else:
            print(f"There are {len(cop_to_or)} repetitive steps.")
            print(f"The copseq_to_orseq dict has {len(copseq_to_orseq)} keys.")
            print(f"There are {len(or_to_cop)} original steps being repeated.")
            print(f"There are {len(step_to_seq)} steps in the step_to_seq dictionary.")
    print(f"There are {len(steps)} steps.")
    graph_features = []
    total_nb_words = sum([len(steps[step].split(' ')) for step in steps])
    cumulative_tokens = 0  # nb of words, actually: we separate on whitespace
    for step in tqdm(steps):
        # create the empty features list
        features = [None]*len(wanted_features)
        if 'node_index' in wanted_features:
            features[wanted_features['node_index']] = step
        if 'distance_to_end' in wanted_features:
            distance = (total_nb_words - cumulative_tokens)/total_nb_words
            features[wanted_features['distance_to_end']] = distance
        if 'nb_words_before' in wanted_features:
                features[wanted_features['nb_words_before']] = cumulative_tokens
                cumulative_tokens+=len(steps[step].split(' '))
        print('\n')
        print(f"---------------------------------Inserting step {step}---------------------------------")
        print(f"The step's content is {steps[step]}")
        print('\n')
        graph.add_node(step)
        if cop_to_or and step in cop_to_or:
            original = cop_to_or[step]
            sequence = step_to_seq[step]
            or_parents = original_parents[original]
            if not or_parents:
                print(f"Node {original} has no parents. Using root as replacement.")
                or_parents = [0]
            features[wanted_features['nb_parents']] = len(or_parents)
            if sequence in copseq_to_orseq:
                orseq = copseq_to_orseq[sequence]
                diff = sequence[0] - orseq[0]
                for p in or_parents:
                    relation = graph.get_edge_data(p, original, default={'relation':0})['relation']
                    if relation == 0:
                        print(f"Problem: no relation found for original edge ({p},{original}).")
                    if p in orseq:
                        graph.add_edge(p+diff,step, relation=relation)
                    else:
                        graph.add_edge(p,step, relation=relation)
            else:  # no corresponding original sequence, we just add the copy node to the parent of the original node
                for p in or_parents:
                    relation = graph.get_edge_data(p, original, default={'relation':0})['relation']
                    if relation == 0:
                        print(f"Problem: no relation found for original edge ({p},{original}).")
                    graph.add_edge(p,step, relation=relation)
        else:
            branch_scores = {}
            negative_branch_scores = {}
            if cop_to_or:
                rep_steps = list(cop_to_or.keys())
            else:
                rep_steps = None
            attachment_pool = get_reduced_attachment_pool(step, leaves, main_branch, rep_steps, prune_coeff, graph)
            additional_negative_attachment_pool = set()
            non_candidates_for_negative_attachment = set()
            while len(attachment_pool)>0:
                node = attachment_pool[0]
                relevant_paths = paths[node]
                random.shuffle(relevant_paths)
                print(f"Relevant paths: {len(relevant_paths)}")
                is_parent = False
                is_neg_parent = False
                #print(f"Relevant paths: {relevant_paths}")
                very_long_step = len(steps[step].split(" ")) >= 150
                used_paths = set()
                path_count = 0
                for path in relevant_paths:
                    if max_path_length_for_nli is not None and len(path)>max_path_length_for_nli:
                        path_content = get_path_content(path[:max_path_length_for_nli],steps)
                    else:
                        path_content = get_path_content(path, steps)
                    if path_content not in used_paths:
                        used_paths.add(path_content)
                        path_count += 1
                        prediction, neg_prediction = nli_client.run(premise=path_content, hypothesis=steps[step])
                        # get entailment probability
                        # add to branch_scores
                        if not is_parent:
                            branch_scores[tuple(path)] = prediction
                            if prediction>=threshold:
                                is_parent = True
                        if not is_neg_parent and node not in non_candidates_for_negative_attachment:
                            negative_branch_scores[tuple(path)] = -neg_prediction
                            if -neg_prediction <= neg_threshold:
                                is_neg_parent = True
                        if is_parent:
                            break
                print(f"Nb of unique paths: {len(used_paths)}")
                ascendants = set(itertools.chain.from_iterable(relevant_paths))
                if is_parent:
                    for ascendant in list(ascendants):
                        if ascendant in attachment_pool:
                            attachment_pool.remove(ascendant)
                if node in attachment_pool:
                    attachment_pool.remove(node)
            print(f"The additional negative attachment pool contains{len(additional_negative_attachment_pool)} nodes.")
            #print(f"Branch scores: {branch_scores}")
            # get three highest scored paths (if there are at least three paths)
            sorted_scores = [(key,value) for key,value in sorted(branch_scores.items(), key=lambda item: item[1], reverse=True)]
            neg_sorted_scores = [(key,value) for key,value in sorted(negative_branch_scores.items(), key=lambda item: item[1], reverse=False)]
            if step<q1:
                nb_parents = 1
            elif step<q2:
                nb_parents = 2
            else:
                nb_parents = 3
            if len(sorted_scores) > nb_parents:
                sorted_scores = sorted_scores[:nb_parents]
            if len(neg_sorted_scores) > 2:
                neg_sorted_scores = neg_sorted_scores[:2]
            # compare their scores to a threshold
            has_parent = False
            new_paths = []
            for k,v in sorted_scores:
                if v>=threshold:
                    has_parent = True
                    parent = list(k)[len(k)-1]
                    print('\n')
                    print(f"Adding edge between {parent} and {step}.")
                    print(f"Content of parent ({parent}): {steps[parent]}")
                    print('\n')
                    graph.add_edge(parent, step, relation=1)
                    new_paths.append(list(k)+[step])
                    if or_to_cop and step in or_to_cop:
                        if step not in original_parents:
                            original_parents[step] = []
                        original_parents[step].append(parent)
                    # add the number of parents
                    if 'nb_parents' in wanted_features:
                        if features[wanted_features['nb_parents']]==None:
                            features[wanted_features['nb_parents']]=0
                        features[wanted_features['nb_parents']] += 1
            for k,v in neg_sorted_scores:
                if v <= neg_threshold:
                    parent = list(k)[len(k)-1]
                    print('\n')
                    print(f"Adding negative edge between {parent} and {step}.")
                    print(f"Content of negative parent ({parent}): {steps[parent]}")
                    print('\n')
                    graph.add_edge(parent,step, relation=2)
                    if or_to_cop and step in or_to_cop:
                        if step not in original_parents:
                            original_parents[step] = []
                        original_parents[step].append(parent)
                    # add the number of parents
                    if 'nb_parents' in wanted_features:
                        if features[wanted_features['nb_parents']]==None:
                            features[wanted_features['nb_parents']]=0
                        features[wanted_features['nb_parents']] += 1
            if not has_parent and step!=0:
                if 'nb_parents' in wanted_features:
                    features[wanted_features['nb_parents']]=1
                print(f"Sorted_scores: {sorted_scores}")
                
                graph.add_edge(sorted_scores[0][0][0], step, relation=1)
                new_paths.append([0,step])
                print('\n')
                print(f"No satisfactory entailment. Adding {sorted_scores[0][0][0]} as parent of {step}")
                print(f"This is the content of the default parent: {steps[sorted_scores[0][0][0]]}")
                print('\n')
                if or_to_cop and step in or_to_cop:
                    if step not in original_parents:
                        original_parents[step] = []
                    original_parents[step].append(sorted_scores[0][0][0])
            elif not has_parent:
                new_paths.append([0])
                if 'nb_parents' in wanted_features:
                    features[wanted_features['nb_parents']]=0
            dict_graph = nx.to_dict_of_dicts(graph)
            paths[step] = new_paths
            if len(sorted_scores)>0:
                main_branch, _ = sorted_scores[0]
            else:
                main_branch = []
            main_branch = list(main_branch)
        graph_features.append(features)
    print(f"The new graph is: {dict_graph}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Number of nodes attached to the root: {len(graph.out_edges(0))}")
    list_steps = [v for _,v in steps.items()]
    embeddings = embedding_client.run(list_steps)
        
    return graph, graph_features, embeddings

def build_graph_from_chain(lcot:str,nli_client:NLI_client, embedding_client:EmbeddingModel,nb_keywords:int=8,max_path_length_for_nli:int=5, t2:float=None, logfile:TextIOWrapper=None, wanted_features=[])->Tuple[nx.DiGraph,List[List[float]],torch.Tensor]:
    """
    Takes an LCoT, splits it into steps, builds the corresponding graph, partial features, and step embeddings.

    Args:
        lcot (str): The LCoT to turn into a graph.
        nli_client (NLI_client): The Natural Language Inference Model to build the graph.
        embedding_client (EmbeddingModel): The sentece-transformers model used to make the step embeddings.
        nb_keywords (int): The number of keywords used to split the LCoT.
        max_path_length_for_nli (int): The maximum number of previous steps given to the NLI model as context.
    """
    steps = intelligent_split(lcot,nb_keywords)
    lengths = length_regularity(steps)
    new_keyword_nb = None
    for length, cnt in lengths.items():
        if length >= 1500 and length < 3000 and cnt > 0:
            new_keyword_nb = 12
        elif length >= 3000 and cnt > 0:
            new_keyword_nb = 15
    if new_keyword_nb:
        sorted_steps = sorted(steps, key=len)
        print(f"The longest step: {sorted_steps[-1]}")
        print(f"The second longest step: {sorted_steps[-2]}")
        steps = intelligent_split(lcot, new_keyword_nb)
    steps = {i:step for i,step in enumerate(steps)}
    graph, features, embeddings = construct_graph(steps=steps, nli_client=nli_client, embedding_client=embedding_client, max_path_length_for_nli=max_path_length_for_nli, wanted_features=wanted_features)
    return graph, features, embeddings

def build_features(graph:nx.DiGraph, all_features:List[List[float]], wanted_features:Dict[str, int])->List[List[float]]:
    """
    Builds the metadata features of each node.

    Args
        graph (nx.DiGraph): The graph.
        all_features (List[List[float]]): The already computed features.
        wanted_features (Dict[str,str]): The features to compute.

    Returns:
        List[List[float]]
    """
    # compute the number of children per node
    # compute the distance to the last node, computed as the number of words and given
    if len(all_features) ==0 or len(graph.nodes) == 0:
        print("Empty graph detected, skipping feature building.")
        return torch.tensor([])
    dict_graph = nx.to_dict_of_dicts(graph)
    print(f"All features before finishing building: {all_features}")
    if 'nb_children' in wanted_features:
        for i in range(len(all_features)):
            all_features[i][wanted_features['nb_children']] = float(len(dict_graph[i]))
    if 'nb_nodes_per_depth' in wanted_features:
        print("nb_nodes_per_depth is present.")
        node_to_depth = {0:0}
        depth_to_node = {0:[0]}
        parents = [0]
        while(len(parents))>0:
            parent = parents.pop(0)
            children = dict_graph[parent]
            print(f"children: {children}")
            for child in children:
                if child not in node_to_depth:
                    node_to_depth[child] = node_to_depth[parent] + 1
                    if node_to_depth[parent] + 1 not in depth_to_node:
                        depth_to_node[node_to_depth[parent] + 1] = []
                    depth_to_node[node_to_depth[parent] + 1].append(child)
                    # if child not in parents:
                    parents.append(child)
        nb_nodes_per_depth = {key:len(value) for key, value in depth_to_node.items()}
        for i in range(len(all_features)):
            node_depth = node_to_depth[i]
            all_features[i][wanted_features['nb_nodes_per_depth']] = float(nb_nodes_per_depth[node_depth])
    print(f"All features after: {all_features}")
    return torch.tensor(all_features)

def read_file_and_make_graphs(graphs_dir:str, lcots_dir:str, filename:str, max_path_length_for_nli:str, hf_dir:str)->None:
    """
    Takes a file containing LCoTs and produces a graph for each LCoT.

    Args:
        graphs_dir (str): The directory where the graphs will be stored.
        lcots_dir (str): The directory where the graphs are stored.
        filename (str): The name of the LCoTs file to read. The file containing the graphs will have the same name.
        max_path_length_for_nli (str): The maximum number of steps to use as context for the NLI model.
        hf_dir (str): The HuggingFace directory.

    Returns:
        None
    """
    lcots = []
    samples = []
    feature_names = ['nb_parents', 'nb_children', 'node_index', 'distance_to_end', 'nb_words_before', 'nb_nodes_per_depth', 'is_critique', 'is_conclusion', 'is_other']
    wanted_features = {feature:i for i, feature in enumerate(feature_names[:6])}
    readfile = os.path.join(lcots_dir,filename)
    writefile = os.path.join(graphs_dir,filename)
    with open(readfile,"r") as r:
        lines = r.readlines()
        for line in lines:
            loaded_line = json.loads(line)
            lcots.append(loaded_line["lcot"])
            samples.append(loaded_line)
    nli_client_path = os.path.join(hf_dir,"models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-docnli-ling-2c/snapshots/eff31bcd5e3d26a4246264878a14e937cc5d7fc0")
    nli_client = NLI_client(nli_client_path)
    embedding_client_path = os.path.join(hf_dir,"models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    embedding_client = EmbeddingModel(embedding_client_path)
    graphs = [build_graph_from_chain(lcot=lcot, nli_client=nli_client, embedding_client=embedding_client, nb_keywords=8, max_path_length_for_nli=max_path_length_for_nli, wanted_features=wanted_features) for lcot in lcots]
    with open(writefile,"w+") as w:
        for (graph, features, embeddings), sample in zip(graphs, samples):
            sample["graph"] = nx.to_dict_of_dicts(graph)
            sample["features"] = build_features(graph, features, wanted_features).tolist()
            sample["embeddings"] = embeddings.tolist()
            for k,v in sample.items():
                print(f"{k}: {type(v)}")
            w.write(json.dumps(sample)+'\n')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-x", "--hf-dir", type=str, default=None, help="The directory in which the HugginFace models and datasets are stored. By default, $HOME/.cache/huggingface/hub/ is used.")
    parser.add_argument("-d", "--data-dir", type=str, default="data/", help="The directory where all the data of the project is stored.")
    parser.add_argument("-f", type=str, help="The basename of the file containing the LCoTs.")
    parser.add_argument("-n", type=int, default=5, help="The max context window size for the NLI model.")

    args = parser.parse_args()

    # Get or create HuggingFace directory

    hf_dir = args.hf_dir if args.hf_dir is not None else os.path.join(Path.home(),".cache/huggingface/hub/")

    # Make sure the graphs directory exists.

    graphs_dir = os.path.join(args.data_dir,"graphs/")
    lcots_dir = os.path.join(args.data_dir,"lcots/")

    if not os.path.isdir(graphs_dir):
        os.mkdir(graphs_dir)

    read_file_and_make_graphs(graphs_dir, lcots_dir, args.f, args.n, hf_dir)
    
    
