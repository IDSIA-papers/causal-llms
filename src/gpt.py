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
    'pandas', 'matplotlib', 'requests', 'bs4', 'lxml', 'tqdm', 'torch'
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
import json

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


def gpt_ner(text):
    system_msg = 'You are a helpful assistant for Named Entity Recognition of medical texts.' 
        
    user_msg = f'''
    You will be provided with an abstract of a medical research paper delimited by the <Text></Text> xml tags. 
    Please read the provided abstract carefully to comprehend the context and content. Analyze the provided text 
    and identify all the meaningful entities that could contribute to understanding cause-and-effect relationships 
    between factors such as diseases, medications, treatments, interventions, symptoms, outcomes, effects, or risk factors. 

    Text:
    <Text>{text}</Text>
    
    Avoid including entities that are synonyms or can be used interchangeably to already identified ones. For example, if text contains 
    both "hospital" and "medical center" (which are synonyms and can be used interchangeably) and you already extracted "hospital" as final entity, 
    do not include "medical center" as well.
    
    Your response should highlight entities that are crucial for establishing causal relationships in the medical context.

    Answer listing the found entities within the tags <Answer><Entity>[entity1]</Entity><Entity>[entity2]</Entity></Answer>
    (e.g., <Answer><Entity>diabetes</Entity><Entity>hypertension</Entity></Answer>) and with the original text in which you highlight
    the found entities using the "<mark></mark>" HTML tag.
    Follow this example to understand the expected output:

    Example:
    Text:
    <Text>Smoking increases the risk of respiratory disease, and both can cause lung cancer.</Text>

    Extracted entities: 
    <Answer><Entity>Smoking</Entity><Entity>respiratory disease</Entity><Entity>lung cancer</Entity></Answer>

    Highlighted text:
    <HighlightedText><mark>Smoking</mark> increases the risk of <mark>respiratory disease</mark>, and both can cause <mark>lung cancer</mark>.</HighlightedText>
    '''
    
    response = gpt_request(system_msg, user_msg)
    while not response:
        response = gpt_request(system_msg, user_msg)

    if not response:
        print('No response from GPT')
        return [], ''
    
    answer_text = response
    
    soup = BeautifulSoup(answer_text, 'html.parser')
    entities = [entity.text for entity in soup.find_all('entity')]

    text_with_no_tags = soup.find('highlightedtext')
    highlighted_text = ''.join(str(tag) for tag in text_with_no_tags.contents)

    return entities, highlighted_text


def pick_random_causal_verb():
    verbs = ['provokes', 'triggers', 'causes', 'leads to', 'induces', 'results in', 'generates', 'produces', 'stimulates', 'instigates', 'engenders', 'promotes', 'gives rise to', 'sparks']
    return random.choice(verbs)


def gpt_causal_discovery(entities, text=None, use_pretrained_knowledge=False, reverse_variable_check=False, query_for_bidirected_edges=True):

    graph_edges = []    

    system_msg = 'You are a helpful assistant for causal reasoning and cause-and-effect relationship discovery.'
 
    intro_msg = f'''
You will be provided with {"an abstract of a medical research paper delimited by the <Text></Text> xml tags, and " if text else ""}\
a pair of entities delimited by the <Entity></Entity> xml tags representing medical entities {"extracted from the given abstract" if text else ""}, such
as medications, treatments, symptoms, diseases, outcomes, side effects, or other medical factors.
            {f"""
            Text:  
            <Text>{text}</Text>""" if text else ""}'''
    instructions_msg = f'''
{"Please read the provided abstract carefully to comprehend the context and content." if text else ""}
Examine the roles, interactions, and details surrounding the entities {"within the abstract" if text else ""}.
Based {"only " if text and not use_pretrained_knowledge else ""}on {"the information in the text " if text else ""}{"and " if text and use_pretrained_knowledge else ""}\
{"your pretrained knowledge" if use_pretrained_knowledge or not text else ""}, determine the most likely cause-and-effect \
relationship between the entities from the following listed options (A, B, C{", D" if query_for_bidirected_edges else ""}):\
    '''
    option_choice_msg = f'''
Your response should accurately reflect the likely causal connection between the two entities based on the 
information {"presented in the text" if text else ""} {"and any additional knowledge" if text and use_pretrained_knowledge else ""} {"you are aware of" if use_pretrained_knowledge or not text else ""}.
If no clear causal relationship is apparent, select the appropriate option accordingly.
Then provide your final answer within the tags <Answer>[answer]</Answer>, (e.g. <Answer>C</Answer>).
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


def optimize_entities(entities, text=None):
    system_msg = 'You are a helpful assistant for medical entity optimization, by accurately identifying and pairing synonyms, redundant entities, or entities that can be used interchangeably'

    entities_text = '\n'.join([f'<Entity>{entity}</Entity>' for entity in entities])

    user_msg = f'''
You will be provided with {'an abstract of a medical research paper delimited by the <Text></Text> xml tags, and ' if text else ''} \
a list of named entities representing medical entities, each one of them delimited by the <Entity></Entity> xml tags. \

{f"""Text:
<Text>{text}</Text>""" if text else ""}

Entities:
{entities_text}

Your task is to optimize this entity list by identifying synonyms within the entities and grouping them accordingly. \
Your goal is to create a JSON object where the keys represent the root word entities, and each key is associated with \
an array of its synonyms, i.e., words or entities that can be used interchangeably to the root word.\
If a root word entity has no synonyms, its value in the JSON should be an empty array.

Ensure that each entity appears only once in the dictionary, either as key (i.e. root word) or as element in the value arrays (i.e. the synonyms): \
an entity must must not appear as key if it is the synonym (i.e. in the value array) of another entity, and the other way around (i.e. must not be \
in the value array of an entity if it is already a key of the JSON object).
An entity must not be a synonym of itself.

You should efficiently process the given list of entities and produce the desired dictionary structure.
The output JSON object should accurately represent the optimized entities and their synonyms based on the provided list.\

Then provide your final JSON object answer within the tags <Answer>[json_object]</Answer>, (e.g. <Answer>{{
    "smoking": ["tobacco", "nicotine", "cigarettes", "cigar"],
    "cancer": ["lung cancer"],
    "tumors": []
}}</Answer>).

You can follow these steps to complete the task:
1. Read carefully the provided list of entities.
2. Loop through the list of entities and identify synonyms. For each entity, find if there are any other entities that are synonyms of it.
3. If the entity has synonyms, check if any of these are already present in the JSON object as key elements: if it is the case, add this entity \
    as an element of the value array of the key entity. If not, add the entity as a key element and add its synonyms as elements of its value array.
4. If the entity has no synonyms, check if it is already present in the JSON object as key element: if it is the case, do nothing. If not, add \
    the entity as a key element and add an empty array as its value.

Follow the examples below to understand the expected output.

Example:

Given the initial list of entities:
<Entity>smoking</Entity>
<Entity>lung cancer</Entity>
<Entity>tumors</Entity>
<Entity>cancer</Entity>
<Entity>tobacco</Entity>
<Entity>nicotine</Entity>
<Entity>cigarettes</Entity>
<Entity>cigar</Entity>

You should pair the synonyms, generate the following JSON object and provide it as your answer:
<Answer>
{{
    "smoking": ["tobacco", "nicotine", "cigarettes", "cigar"],
    "cancer": ["lung cancer"],
    "tumors": []
}}
</Answer>

Note that every entity appears only once in the output JSON object, either as key or as element in the value arrays.

After you have finished building the JSON object, check and make sure that every entity appears only once in the output JSON object, \
either as key or as element in the value arrays.

After you have finished building the JSON object, perform an 

If you realize you have made a mistake by adding entities twice in the JSON object, you must go back and correct it.
For example, if you have added "lung cancer" as a synonym of "cancer" and "lung cancer" also appears as a key element, you should remove 
the redundant "lung cancer" element from the JSON object keys, while keeping it in the synonyms array of "cancer".

Follow the example below to understand how you should behave in case you realize an entity appears multiple times, as both key element and synonym in the value arrays.

Example:
If you create an output JSON object that looks like this:
<Answer>
{{
    "smoking": ["tobacco", "nicotine", "cigarettes", "cigar"],
    "cancer": ["lung cancer"],
    "lung cancer": ["cancer"],
    "tumors": []
}}
</Answer>

You should perform a careful check and realize that "lung cancer" appears both as key element and in the value arrays, as synonym of the "cancer" entity. \
You should remove the redundant "lung cancer" element from the JSON object keys, while keeping it in the synonyms array of "cancer", and provide the following JSON 
object as your answer:
<Answer>
{{
    "smoking": ["tobacco", "nicotine", "cigarettes", "cigar"],
    "cancer": ["lung cancer"],
    "tumors": []
}}
</Answer>
'''
    
    response = gpt_request(system_msg, user_msg, temperature=1)
    if response:
        soup = BeautifulSoup(response, 'html.parser')
        answer = soup.find('answer').text
        try:
            opt_entities = json.loads(answer)
            if opt_entities:
                return opt_entities
        except (json.JSONDecodeError, TypeError):
            pass

    return entities



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
You will be provided with {"an abstract of a medical research paper delimited by the <Text></Text> xml tags, and " if text else ""}\
a pair of entities delimited by the <Entity></Entity> xml tags representing medical entities {"extracted from the given abstract" if text else ""} (such
as medications, treatments, symptoms, diseases, outcomes, side effects, or other medical factors), and two answers you previously gave to this same request\
that are incoherent with each other, delimited by the <Answer></Answer> xml tags.
            {f"""
Text:  
<Text>{text}</Text>""" if text else ""}'''
    instructions_msg = f'''
{"Please read the provided abstract carefully to comprehend the context and content." if text else ""}
Consider the previous answers you gave to this same request that are incoherent with each other, and the entities they refer to in order to give a correct answer.
Examine the roles, interactions, and details surrounding the entities {"within the abstract" if text else ""}.
Based {"only " if text and not use_pretrained_knowledge else ""}on {"the information in the text " if text else ""}{"and " if text and use_pretrained_knowledge else ""}\
{"your pretrained knowledge" if use_pretrained_knowledge or not text else ""}, determine the most likely cause-and-effect \
relationship between the entities from the following listed options (A, B, C, D):\
    '''
    option_choice_msg = f'''
Your response should accurately reflect the likely causal connection between the two entities based on the 
information {"presented in the text" if text else ""} {"and any additional knowledge" if text and use_pretrained_knowledge else ""} {"you are aware of" if use_pretrained_knowledge or not text else ""}.
If no clear causal relationship is apparent, select the appropriate option accordingly.
Then provide your final answer within the tags <Answer>[answer]</Answer>, (e.g. <Answer>C</Answer>).
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


def gpt_edge_explanation(edges, text=None):
    graph_edges = []
    
    system_msg = 'You are a helpful assistant for causal reasoning and cause-and-effect relationship discovery.'
 
    intro_msg = f'''
You will be provided with {"a text of a medical research paper delimited by the <Text></Text> xml tags, " if text else ""}\
a pair of entities delimited by the <Entity></Entity> xml tags representing medical entities {"extracted from the given text" if text else ""} (such
as medications, treatments, symptoms, diseases, outcomes, side effects, or other medical factors), and the causal relationship found between the two entities\
delimited by the <Relationship></Relationship> xml tags, that has been previously extracted {"from the given medical text" if text else ""}.

    {f"""
    Text:
    <Text>{text}</Text>""" if text else ""}
    '''
    instructions_msg = f'''
Your first task is to read carefully the provided text and the causal relationship found between the two entities.
Then, given the causal relationship {"and the provided text" if text else ""}, you should {"extract the piece of text or sentence that explains" if text else "provide a textual explanation of"} 
the causal relationship between the two entities, representing the reason why that particular causal relationship was previously chosen.
{"Make sure that the explanation of the causal relationship is actually extracted from the given text. Once you have an answer, check that is it extracted from the given text." if text else ""}

Provide your final answer within the tags <Answer>[answer]</Answer>, (e.g. <Answer>Smoking increases the risk of respiratory disease</Answer>).
If the two entities are not causally related, simply provide an empty answer within the tags <Answer></Answer>.

Follow the example below to better understand the task and the expected output.

EXAMPLE:

Text:
<Text>Smoking increases the risk of respiratory disease, and both can cause lung cancer.</Text>

Entities:
<Entity>smoking</Entity>
<Entity>respiratory disease</Entity>

Relationship:
<Relationship>"smoking" causes "respiratory disease"</Relationship>

You should extract the piece of text that explains the causal relationship between the two entities, which in this case is:
<Answer>Smoking increases the risk of respiratory disease</Answer>
    '''

    for (e1, e2), answer in edges:
        textual_relationship = get_textual_answers(e1, e2, answer)

        user_msg = f'''\
        {intro_msg}

        Entities:
        <Entity>{e1}</Entity>
        <Entity>{e2}</Entity>

        Relationship:
        <Relationship>{textual_relationship}</Relationship>

        {instructions_msg}
        '''

        response = gpt_request(system_msg, user_msg)

        if response:
            soup = BeautifulSoup(response, 'html.parser')
            gpt_explanation = soup.find('answer').text
            try:
                graph_edges.append((((e1, e2), answer), gpt_explanation))
            except (json.JSONDecodeError, TypeError):
                pass

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


def preprocess_edges(edges, perform_edge_explanation=False):
    nodes = set()
    directed_edges = []
    bidirected_edges = []

    for edge in edges:
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
            else:
                direction = [(direction[0], explanation)] if perform_edge_explanation else direction
                directed_edges.extend(direction)

    return list(nodes), directed_edges, bidirected_edges


def find_cycles(nodes=[], edges=[], return_first_100_cycles=True, edge_explanation=False):
    if not nodes or not edges:
        return []

    g = gt.Graph(directed=True)

    nodes_ids = {}
    v_prop = g.new_vertex_property("string")
    for n in nodes:
        v = g.add_vertex()
        v_prop[v] = n
        nodes_ids[n] = v

    for edge in edges:
        (n1, n2), expl = edge if edge_explanation else (edge, None)
        e = g.add_edge(nodes_ids[n1], nodes_ids[n2])

    cycles = []
    for i, c in enumerate(gt.all_circuits(g)):
        if return_first_100_cycles and i >= 100:
            break
        cycles.append([v_prop[v] for v in c])

    return cycles


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



def causal_discovery_pipeline(text_title, text, entities=[], use_gpt_4=True, use_text_in_causal_discovery=False, use_LLM_pretrained_knowledge_in_causal_discovery=False, causal_discovery_query_for_bidirected_edges=True, perform_edge_explanation=False, reverse_edge_for_variable_check=False, optimize_found_entities=True, use_text_in_entity_optimization=True, search_cycles=True, plot_static_graph=True, graph_directory_name='../graphs', verbose=False):
    start = time.time()

    init(causal_discovery_query_for_bidirected_edges)

    if verbose and text:
        print('Text:')
        print(text)
        print('--')

    text_with_highlighted_variables = None
    if entities == []:
        entities, text_with_highlighted_variables = gpt_ner(text)
    else:
        if verbose:
            print('Skipping NER operation. Using provided entities.')
            print('--')

    if verbose:
        print(f'Entities: ({len(entities)} = {entities})')
        print('--')

    if optimize_found_entities:
        opt_entities = optimize_entities(entities, text=(text if use_text_in_entity_optimization else None))
        entities = list(opt_entities.keys())

        if verbose:
            print(f'Optimized Entities: ({len(entities)} = {entities})')
        
    graph_edges = gpt_causal_discovery(entities, text=(text if use_text_in_causal_discovery else None), use_pretrained_knowledge=use_LLM_pretrained_knowledge_in_causal_discovery, reverse_variable_check=reverse_edge_for_variable_check, query_for_bidirected_edges=causal_discovery_query_for_bidirected_edges)

    edges = extract_edge_answers(graph_edges)
    if verbose:
        print('Edges:')
        print(edges)
        print('--')

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

        valid_edges.extend(corrected_edges)
        edges = valid_edges
    
    if perform_edge_explanation:
        edges = gpt_edge_explanation(edges, text=(text if use_text_in_causal_discovery else None))

    nodes, directed_edges, bidirected_edges = preprocess_edges(edges, perform_edge_explanation)

    cycles = []
    if search_cycles:
        cycles = find_cycles(nodes=nodes, edges=directed_edges, edge_explanation=perform_edge_explanation)

    if verbose:
        print('Nodes:')
        print(nodes)
        print('--')
        print('Processed Edges:')
        print(directed_edges)
        print('--')
    
    build_graph(nodes=nodes, edges=directed_edges, bidirected_edges=bidirected_edges, cycles=cycles, plot_static_graph=plot_static_graph, directory_name=graph_directory_name, graph_name=text_title, highlighted_text=text_with_highlighted_variables, edge_explanation=perform_edge_explanation)
    
    if verbose:
        if cycles:
            print('GRAPH IS CYCLIC')
        else:
            print('Graph is acyclic')
        
        print('--')

    elapsed_seconds = time.time() - start
    if verbose:
        print_edges(graph_edges)
        print(f'exec time : {time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))}')

    return nodes, directed_edges, bidirected_edges, cycles, elapsed_seconds, perform_edge_explanation


# Example text for test
def example_test(directory='../results/'):
    text = 'Excessive alcohol consumption can cause liver cirrhosis, and both can lead to death.'
    text_title = 'Example test'
    bidirected_edges=False
    entity_optimization=True
    edge_explanation=False
    return causal_discovery_pipeline(text_title, text, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=False, reverse_edge_for_variable_check=False, causal_discovery_query_for_bidirected_edges=bidirected_edges, perform_edge_explanation=edge_explanation, optimize_found_entities=entity_optimization, use_text_in_entity_optimization=True, search_cycles=False, plot_static_graph=False, graph_directory_name=directory, verbose=False)