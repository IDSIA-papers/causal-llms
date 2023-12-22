#!/usr/bin/env python
# coding: utf-8


import warnings

# Ignore all runtime warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)


import subprocess
import pkgutil

packages_to_install = [
    'openai', 'pyvis', 'plotly', 'cdt', 'python-dotenv',
    'pandas', 'matplotlib', 'requests', 'bs4', 'lxml', 'tqdm', 'torch', 'scikit-learn'
]

for package in packages_to_install:
    if not pkgutil.find_loader(package):
        subprocess.run(['pip', 'install', package, '--quiet'])

import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import re
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
from itertools import permutations, combinations
from datetime import datetime
import json
import sklearn
from sklearn.metrics import classification_report

import networkx as nx
from pyvis.network import Network
import graph_tool.all as gt

from cdt.metrics import SHD, precision_recall

import os
from dotenv import load_dotenv, find_dotenv
import random


_ = load_dotenv(find_dotenv())

api_key  = os.getenv('IDSIA_OPENAI_API_KEY')
organization = os.getenv('IDSIA_ORGANIZATION')

openai.api_key = api_key
openai.organization = organization

models = openai.Model.list()
model_ids = [model['id'] for model in models['data']]

gpt_4 = 'gpt-4'
gpt_4_prev = 'gpt-4-1106-preview'
default_model = 'gpt-3.5-turbo'
use_gpt_4=True
use_gpt_4_prev=True
if use_gpt_4_prev and gpt_4_prev in model_ids:
    default_model = gpt_4_prev
elif use_gpt_4 and gpt_4 in model_ids:
    default_model = gpt_4


forward_arrow = '->'
forward_arrow_answer = 'A'
backward_arrow = '<-'
backward_arrow_answer = 'B'
no_arrow = ' '
no_arrow_answer = 'C'
bidirectional_arrow = '<->'
bidirectional_arrow_answer = 'D'
arrows = {forward_arrow_answer:forward_arrow, backward_arrow_answer:backward_arrow, no_arrow_answer:no_arrow}
coherent_answers = [(forward_arrow, backward_arrow), (backward_arrow, forward_arrow), (no_arrow, no_arrow)]

answer_pattern = re.compile(r'^([A-Z])[.:]')

def init(query_for_bidirected_edges=True):
    if query_for_bidirected_edges:
        arrows[bidirectional_arrow_answer] = bidirectional_arrow
        coherent_answers.append((bidirectional_arrow, bidirectional_arrow))


def gpt_request(system_msg, user_msg, model=default_model, temperature=0.2):
    if not system_msg or not user_msg:
        return None
    try:
        response = openai.ChatCompletion.create(model=model,
                            messages=[
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": user_msg}], 
                            temperature=temperature)
        return response.choices[0].message.content
    except:
        return None

def pick_random_causal_verb():
    verbs = ['provokes', 'triggers', 'causes', 'leads to', 'induces', 'results in', 'generates', 'produces', 'stimulates', 'instigates', 'engenders', 'promotes', 'gives rise to', 'sparks']
    return random.choice(verbs)


def gpt_causal_discovery(entities, text=None, use_pretrained_knowledge=False, reverse_variable_check=False, query_for_bidirected_edges=True):

    graph_edges = []    

    system_msg = 'You are a helpful assistant for causal reasoning and cause-and-effect relationship discovery.'
 
    intro_msg = f'''
You will be provided with {"a text delimited by the <Text></Text> xml tags, and " if text else ""}\
a pair of entities delimited by the <Entity></Entity> xml tags representing entities {"extracted from the given text" if text else ""}.
            {f"""
            Text:  
            <Text>{text}</Text>""" if text else ""}'''
    instructions_msg = f'''
{"Please read the provided text carefully to comprehend the context and content." if text else ""}
Examine the roles, interactions, and details surrounding the entities {"within the text" if text else ""}.
Based {"only " if text and not use_pretrained_knowledge else ""}on {"the information in the text " if text else ""}{"and " if text and use_pretrained_knowledge else ""}\
{"your pretrained knowledge" if use_pretrained_knowledge or not text else ""}, determine the most likely cause-and-effect \
relationship between the entities from the following listed options (A, B, C{", D" if query_for_bidirected_edges else ""}):\
    '''
    option_choice_msg = f'''
Your response should analyze the situation in a step-by-step manner, ensuring the correctness of the ultimate conclusion, which should 
accurately reflect the likely causal connection between the two entities based on the 
information {"presented in the text" if text else ""} {"and any additional knowledge" if text and use_pretrained_knowledge else ""} {"you are aware of" if use_pretrained_knowledge or not text else ""}.
If no clear causal relationship is apparent, select the appropriate option accordingly.

Then provide your final answer within the tags <Answer>[answer]</Answer>, (e.g. <Answer>C</Answer>).
'''
    total_iterations = len(list(permutations(entities, 2))) if reverse_variable_check else len(list(combinations(entities, 2)))
    progress_bar = tqdm(total=total_iterations, desc="Progress")

    for i1, e1 in enumerate(entities):
        for i2, e2 in enumerate(entities):
            if i1 == i2 or (not reverse_variable_check and i1 >= i2):
                continue

            options_with_random_verb = f'''\
            Options:
            A: "{e1}" {pick_random_causal_verb()} "{e2}"; 
            B: "{e2}" {pick_random_causal_verb()} "{e1}"; 
            C: "{e1}" and "{e2}" are not directly causally related;
            {f"""D: there is a common factor that is the cause for both "{e1}" and "{e2}";""" if query_for_bidirected_edges else ""}
            '''

            user_msg = f'''\
            {intro_msg}

            Entities:
            <Entity>{e1}</Entity>
            <Entity>{e2}</Entity>
            \
            {instructions_msg}
            {options_with_random_verb}
            \
            {option_choice_msg}
            '''

            response = gpt_request(system_msg, user_msg)
            if response:
                graph_edges.append(((e1, e2), response))
            
            progress_bar.update(1)

    progress_bar.close()
    
    return graph_edges


def get_edge_answer(text):
    soup = BeautifulSoup(text, 'html.parser')
    answer = soup.find('answer').text

    if answer in arrows:
        return arrows[answer]

    match = answer_pattern.match(answer)
    if match:
        answer = match.group(1)

    if answer in arrows:
        return arrows[answer]
    
    return None


def print_edges(graph_edges):
    for (e1, e2), answer in graph_edges:
        try:
            print(f'{e1} {get_edge_answer(answer)} {e2}')
        except:
            print(f'{e1} ? {e2}')

def extract_edge_answers(edges):
    edges_with_answers = []

    for (e1, e2), text in edges:
        try:
            soup = BeautifulSoup(text, 'html.parser')
            answer = soup.find('answer').text
            
            if answer in arrows:
                edges_with_answers.append(((e1, e2), answer))
                continue

            match = answer_pattern.match(answer)
            if match:
                if match.group(1) in arrows:
                    edges_with_answers.append(((e1, e2), match.group(1)))
                    continue

        except:
            continue

    return edges_with_answers


def check_edge_compatibility(answer1, answer2):
    return (arrows[answer1], arrows[answer2]) in coherent_answers


def check_invalid_answers(directed_edges):
    invalid_edges = []
    valid_edges = []
    temp_edges = []
    answers = {}
    for (n1, n2), answer in directed_edges:

        if (n1, n2) not in temp_edges and (n2, n1) not in temp_edges:
            temp_edges.append((n1, n2))
            answers[(n1, n2)] = answer
        elif (n1, n2) in temp_edges:
            if answers[(n1, n2)] != answer:
                invalid_edges.append((((n1, n2), answer), ((n2, n1), answers[(n2, n1)])))
            else:
                valid_edges.append(((n1, n2), answer))
            
            temp_edges.remove((n1, n2))
        elif (n2, n1) in temp_edges:
            if check_edge_compatibility(answers[(n2, n1)], answer):
                valid_edges.append(((n1, n2), answer))
            else:
                invalid_edges.append((((n1, n2), answer), ((n2, n1), answers[(n2, n1)])))
            
            temp_edges.remove((n2, n1))

    for n1, n2 in temp_edges:
        if (n1, n2) not in invalid_edges:
            
            if (n2, n1) not in answers:
                invalid_edges.append((((n1, n2), answer), ((n2, n1), no_arrow_answer)))
            else:
                invalid_edges.append((((n1, n2), answer), ((n2, n1), answers[(n2, n1)])))
    
    return valid_edges, invalid_edges


def get_textual_answers(e1, e2, ans):
    if ans == forward_arrow_answer:
        return f'"{e1}" causes "{e2}"'
    elif ans == backward_arrow_answer:
        return f'"{e2}" causes "{e1}"'
    elif ans == no_arrow_answer:
        return f'"{e1}" and "{e2}" are not causally related'
    elif ans == bidirectional_arrow_answer:
        return f'there is a common factor that is the cause for both "{e1}" and "{e2}"'
    else:
        return None


def correct_invalid_edges(invalid_edges, text=None, use_pretrained_knowledge=False, query_for_bidirected_edges=True):
    graph_edges = []

    if not invalid_edges:
        return []
    
    system_msg = 'You are a helpful assistant for causal reasoning and cause-and-effect relationship discovery.'
 
    intro_msg = f'''
You will be provided with {"an text delimited by the <Text></Text> xml tags, and " if text else ""}\
a pair of entities delimited by the <Entity></Entity> xml tags {"representing entities extracted from the given text" if text else ""},\
and two answers you previously gave to this same request that are incoherent with each other, delimited by the <Answer></Answer> xml tags.
            {f"""
Text:  
<Text>{text}</Text>""" if text else ""}'''
    instructions_msg = f'''
{"Please read the provided text carefully to comprehend the context and content." if text else ""}
Consider the previous answers you gave to this same request that are incoherent with each other, and the entities they refer to in order to give a correct answer.
Examine the roles, interactions, and details surrounding the entities {"within the text" if text else ""}.
Based {"only " if text and not use_pretrained_knowledge else ""}on {"the information in the text " if text else ""}{"and " if text and use_pretrained_knowledge else ""}\
{"your pretrained knowledge" if use_pretrained_knowledge or not text else ""}, determine the most likely cause-and-effect \
relationship between the entities from the following listed options (A, B, C{", D" if query_for_bidirected_edges else ""}):\
    '''
    option_choice_msg = f'''
Your response should analyze the situation in a step-by-step manner, ensuring the correctness of the ultimate conclusion, which should 
accurately reflect the likely causal connection between the two entities based on the 
information {"presented in the text" if text else ""} {"and any additional knowledge" if text and use_pretrained_knowledge else ""} {"you are aware of" if use_pretrained_knowledge or not text else ""}.
If no clear causal relationship is apparent, select the appropriate option accordingly.

Then provide your final answer within the tags <Answer>[answer]</Answer>, (e.g. <Answer>C</Answer>).
'''

    for ((e1, e2), answer1), ((e3, e4), answer2) in invalid_edges:       

        previous_answers_msg = f'''
        Previous incoherent answers:
        <Answer>{get_textual_answers(e1, e2, answer1)}</Answer>
        <Answer>{get_textual_answers(e3, e4, answer2)}</Answer>'''

        options_with_random_verb = f'''
        Options:
        A: "{e1}" {pick_random_causal_verb()} "{e2}"; 
        B: "{e2}" {pick_random_causal_verb()} "{e1}"; 
        C: "{e1}" and "{e2}" are not directly causally related; 
        {f"""D: there is a common factor that is the cause for both "{e1}" and "{e2}";""" if query_for_bidirected_edges else ""}
        '''

        user_msg = f'''\
        {intro_msg}

        Entities:
        <Entity>{e1}</Entity>
        <Entity>{e2}</Entity>

        {previous_answers_msg}
        \
        {instructions_msg}
        
        {options_with_random_verb}
        \
        {option_choice_msg}
        '''

        response = gpt_request(system_msg, user_msg)
        if response:
            graph_edges.append(((e1, e2), response))
            
    return graph_edges



def normalize_edge_direction(e1, e2, answer):
    if answer in arrows:
        if arrows[answer] == forward_arrow:
            return [(e1, e2)]
        elif arrows[answer] == backward_arrow:
            return [(e2, e1)]
        elif arrows[answer] == bidirectional_arrow:
            return [(e2, e1), (e1, e2)]
    return None


def preprocess_edges(edges, perform_edge_explanation=False, bidirectional_edge_test=False):
    nodes = set()
    directed_edges = []
    bidirected_edges = []
    directed_edges_test_contradictory = []
    bidirected_edges_test_contradictory = []

    for edge in edges:
        bi_test_contradictory = None
        if bidirectional_edge_test:
            edge, bi_test_contradictory = edge

        if perform_edge_explanation:
            ((n1, n2), answer), explanation = edge
        else:
            (n1, n2), answer = edge

        nodes.add(n1)
        nodes.add(n2)

        direction = normalize_edge_direction(n1, n2, answer)
        if direction:
            is_bidirectional = len(direction) == 2

            if is_bidirectional:
                direction = direction[0]
                direction = [((direction[0],direction[1]), explanation)] if perform_edge_explanation else direction
                bidirected_edges.extend(direction)
                bidirected_edges_test_contradictory.append(bi_test_contradictory)
            else:
                direction = [(direction[0], explanation)] if perform_edge_explanation else direction
                directed_edges.extend(direction)
                directed_edges_test_contradictory.append(bi_test_contradictory)


    return list(nodes), (directed_edges, directed_edges_test_contradictory), (bidirected_edges, bidirected_edges_test_contradictory)


def build_graph(nodes, edges=[], bidirected_edges=[], cycles=[], plot_static_graph=True, directory_name='../graphs', graph_name='mygraph', highlighted_text=None, edge_explanation=False):

    if plot_static_graph:
        plt.figure()
    G = nx.DiGraph()

    G.add_nodes_from(nodes)

    for edge in edges:
        if edge_explanation:
            (e1, e2), explanation = edge
            G.add_edge(e1, e2, title=explanation, color='black', style='solid')
        else:
            e1, e2 = edge
            G.add_edge(e1, e2, color='black', style='solid')

    for cycle in cycles:
        for i in range(len(cycle) - 1):
            G[cycle[i]][cycle[i + 1]]['color'] = 'red'
        G[cycle[-1]][cycle[0]]['color'] = 'red'

    for edge in bidirected_edges:
        if edge_explanation:
            (e1, e2), explanation = edge
            G.add_edge(e1, e2, title=explanation, color='grey', style='dashed')
            G.add_edge(e2, e1, color='grey', style='dashed')
        else:
            e1, e2 = edge
            G.add_edge(e1, e2, color='grey', style='dashed')

    if plot_static_graph:
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_labels(G, pos)

        edge_colors = [G.edges[edge]['color'] for edge in G.edges()]
        edge_styles = [G.edges[edge]['style'] for edge in G.edges()]

        nx.draw(G, pos, node_color='skyblue', node_size=1500,font_size=10, font_weight='bold', arrowsize=20, edge_color=edge_colors, style=edge_styles,width=2)
        plt.title(graph_name)
        plt.show()

    graph_file_name = f'{directory_name}/{graph_name}.html'
    net = Network(directed=True)
    net.from_nx(G)
    net.force_atlas_2based()
    net.show_buttons(filter_=['physics'])
    os.makedirs(directory_name, exist_ok=True)
    net.save_graph(graph_file_name)

    if highlighted_text:
        with open(graph_file_name, 'r', encoding='utf-8') as file:
            graph_html = file.read()

        graph_soup = BeautifulSoup(graph_html, 'html.parser')

        new_paragraph = graph_soup.new_tag('p')
        new_paragraph.append(BeautifulSoup(highlighted_text, 'html.parser'))

        new_section = graph_soup.new_tag('section')
        new_section.append(new_paragraph)

        body = graph_soup.find('body')

        if body:
            body.insert(0, new_section)

        with open(graph_file_name, 'w', encoding='utf-8') as file:
            file.write(graph_soup.prettify())



def causal_discovery_pipeline(text_title, text, entities=[], use_gpt_4=True, use_text_in_causal_discovery=False, use_LLM_pretrained_knowledge_in_causal_discovery=False, causal_discovery_query_for_bidirected_edges=True, perform_edge_explanation=False, reverse_edge_for_variable_check=False, optimize_found_entities=True, use_text_in_entity_optimization=True, build_causal_graph=False, search_cycles=True, plot_static_graph=True, graph_directory_name='../graphs', verbose=False):
    start = time.time()

    if verbose and text:
        print('Text:')
        print(text)
        print('--')

    if verbose:
        print(f'Entities ({len(entities)}): {entities}')
        print('--')
        
    graph_edges = gpt_causal_discovery(entities, text=(text if use_text_in_causal_discovery else None), use_pretrained_knowledge=use_LLM_pretrained_knowledge_in_causal_discovery, reverse_variable_check=reverse_edge_for_variable_check, query_for_bidirected_edges=causal_discovery_query_for_bidirected_edges)

    edges = extract_edge_answers(graph_edges)
    if verbose:
        print('Edges:')
        print(edges)
        print('--')

    contradictory_edge_test = []

    if reverse_edge_for_variable_check:    
        valid_edges, invalid_edges = check_invalid_answers(edges)
        if verbose:
            print('Valid Edges:')
            print(valid_edges)
            print('--')
            print('Invalid Edges:')
            print(invalid_edges)
            print('--')
        
        edge_correction_response = correct_invalid_edges(invalid_edges, text, use_pretrained_knowledge=use_LLM_pretrained_knowledge_in_causal_discovery, query_for_bidirected_edges=causal_discovery_query_for_bidirected_edges)
        corrected_edges = extract_edge_answers(edge_correction_response)
        if verbose:
            print('Edge Correction Response:')
            print(corrected_edges)
            print('--')

        contradictory_edge_test.extend([False]*len(valid_edges))
        contradictory_edge_test.extend([True]*len(invalid_edges))
        
        valid_edges.extend(corrected_edges)
        edges = valid_edges
    
    if reverse_edge_for_variable_check:
        edges = [(edge, contradictory_edge_test[i]) for i, edge in enumerate(edges)]

    nodes, directed_edges, bidirected_edges = preprocess_edges(edges, perform_edge_explanation=perform_edge_explanation, bidirectional_edge_test=reverse_edge_for_variable_check)

    if verbose:
        print('Nodes:')
        print(nodes)
        print('--')
        print('Processed Edges:')
        print(directed_edges[0])
        print('--')
    
    if build_causal_graph:
        build_graph(nodes=nodes, edges=directed_edges[0], bidirected_edges=bidirected_edges[0], plot_static_graph=plot_static_graph, directory_name=graph_directory_name, graph_name=text_title, edge_explanation=perform_edge_explanation)

    elapsed_seconds = time.time() - start
    if verbose:
        print_edges(graph_edges)
        print(f'exec time : {time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))}')

    return nodes, directed_edges, bidirected_edges, elapsed_seconds, perform_edge_explanation


def main():
    data = pd.read_csv('../data/xai4sci/test - full.csv')

    data['entities'] = data['entities'].apply(eval)
    l = len(data)
    
    init(False)

    reverse_edge_check=True
    edge_explanation=False
    results = pd.DataFrame(columns=['sentence', 'entities', 'gt_relation', 'pred_relation', 'pred_edge', 'is_edge_test_contradictory', 'exec_time'])

    file_name = f'../results/xai4sci/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} - {default_model} - results.csv'
    for i, row in enumerate(data.iterrows()):
        row = row[1]
        nodes, directed_edges, bidirected_edges, elapsed_seconds, edge_explanation = causal_discovery_pipeline(text_title=i, text=row['sentence_no_tags'], entities=row['entities'], use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=False, reverse_edge_for_variable_check=reverse_edge_check, causal_discovery_query_for_bidirected_edges=False, perform_edge_explanation=edge_explanation, optimize_found_entities=False, use_text_in_entity_optimization=True, search_cycles=False, plot_static_graph=False, verbose=False)
        
        pred_rel = 0
        pred_edge = []
        gt_edge = row['relation']
        is_edge_test_contradictory = False

        directed_edges, is_edge_tests_contradictory = directed_edges
        
        if len(directed_edges) == 0:
            pred_rel = '-1'
        elif len(directed_edges) == 1:
            if edge_explanation:
                pred_edge = directed_edges[0][0]
            else:
                pred_edge = directed_edges[0]

            if pred_edge[0] == row['entities'][0]:
                pred_rel = '0'
            else:
                pred_rel = '1'
            is_edge_test_contradictory = is_edge_tests_contradictory[0]       

        
        new_row = pd.DataFrame({'sentence': row['sentence_no_tags'], 'entities': [row['entities']], 'gt_relation': str(row['relation']), 'pred_relation': pred_rel, 'pred_edge': [pred_edge], 'is_edge_test_contradictory':is_edge_test_contradictory, 'exec_time': time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))}, index=[0])
        results = pd.concat([results, new_row]).reset_index(drop=True)
        results.to_csv(file_name, index=False)
        print(f'{i}/{l}')


    print(classification_report([(i if i in ['0', '1'] else '-1') for i in results['gt_relation'].values], results['pred_relation'].values, labels=[-1, 0, 1], target_names=['No relation', 'X -> Y', 'X <- Y']))


if __name__ == "__main__":
    main()