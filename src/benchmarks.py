import numpy as np
import pandas as pd
import gpt
import networkx as nx
import plotly as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from cdt.metrics import SHD, precision_recall
from pyvis.network import Network
import random
import os

class Algorithm:
    BASELINE = 0
    GPT = 1


def plot_causal_graph(nodes, edges, title, graph_path):

    net = Network(directed=True, notebook=True)
    net.force_atlas_2based()
    net.show_buttons(filter_=['physics']) 

    node_ids = {}

    for i, node in enumerate(nodes):
        net.add_node(i, label=node)
        node_ids[node] = i
    
    for e1, e2 in edges:
        net.add_edge(source=node_ids[e1], to=node_ids[e2])

    net.show(f'{graph_path}/{title}.html')



def f1_score(precision, recall):
    try:
        return 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        return None


def benchmark_prediction(benchmark_title, ground_truth_nodes, ground_truth_edges, SHD_double_for_anticausal=True, save_graphs=True, graphs_directory='../graphs', algorithm=Algorithm.GPT, verbose=False):

    if ground_truth_nodes is None or ground_truth_edges is None:
        print("Ground truth nodes or edges are None.")
        return None, None, None

    prediction_edges = []
    cycles = []
    if algorithm == Algorithm.BASELINE:
        for i, node1 in enumerate(ground_truth_nodes):
                for j, node2 in enumerate(ground_truth_nodes):
                    if i > j:
                        random_edge  = random.randint(0, 3)
                        if random_edge  == 0:
                            prediction_edges.append((node1, node2))
                        elif random_edge  == 1:
                            prediction_edges.append((node2, node1))
                        elif random_edge  == 2:
                            prediction_edges.extend([(node1, node2), (node2, node1)])
        plot_causal_graph(ground_truth_nodes, prediction_edges, f'{benchmark_title} - Baseline', graphs_directory)
    elif algorithm == Algorithm.GPT:
        nodes, prediction_directed_edges, prediction_bidirected_edges, cycles = gpt.causal_discovery_pipeline(f'{benchmark_title} - Prediction', '', entities=ground_truth_nodes, use_text_in_causal_discovery=False, use_LLM_pretrained_knowledge_in_causal_discovery=False, reverse_edge_for_variable_check=False, optimize_found_entities=False, use_text_in_entity_optimization=False, search_cycles=True, plot_static_graph=False, graph_directory_name=graphs_directory, verbose=False)
        prediction_edges = prediction_directed_edges + prediction_bidirected_edges
        
    if verbose:
        print(prediction_edges)

    return nodes, prediction_edges, cycles


def evaluate_predictions(ground_truth_nodes, ground_truth_edges, prediction_edges, SHD_double_for_anticausal=True, verbose=False):
    ground_truth_graph = nx.DiGraph()
    ground_truth_graph.add_nodes_from(ground_truth_nodes)
    ground_truth_graph.add_edges_from(ground_truth_edges)

    prediction_graph = nx.DiGraph()
    prediction_graph.add_nodes_from(ground_truth_nodes)
    prediction_graph.add_edges_from(prediction_edges)

    shd = SHD(ground_truth_graph, prediction_graph, double_for_anticausal=SHD_double_for_anticausal)

    aupr, curve = precision_recall(ground_truth_graph, prediction_graph)

    extra_edges = []
    missing_edges = []
    correct_direction = []
    incorrect_direction = []

    for e in prediction_edges:
        if e not in ground_truth_edges:
            extra_edges.append(e)
        if (e in ground_truth_edges and (e[1], e[0]) not in ground_truth_edges and (e[1], e[0]) not in prediction_edges) or (e in ground_truth_edges and (e[1], e[0]) in ground_truth_edges and (e[1], e[0]) in prediction_edges):
            correct_direction.append(e)
        else:
            incorrect_direction.append(e)

    for e in ground_truth_edges:
        if e not in prediction_edges:
            missing_edges.append(e)
    
    if len(curve) == 2:
        precision, recall = curve[0][0], curve[0][1]
    else:
        precision, recall = curve[1][0], curve[1][1]

    f1 = f1_score(precision, recall)

    return shd, aupr, curve, precision, recall, f1, prediction_edges, missing_edges, extra_edges, correct_direction, incorrect_direction


def precision_recall_curve_plot(titles, curves, graph_path):
    fig = go.Figure()

    for i, curve_point in enumerate(curves):
        precision_values = [point[0] for point in curve_point]
        recall_values = [point[1] for point in curve_point]

        fig.add_trace(go.Scatter(
                x=recall_values,
                y=precision_values,
                text=f'F1 score = {f1_score(precision_values[1], recall_values[1]):.2f}',
                mode='lines+markers',
                name=titles[i]
            ))

    fig.add_trace(go.Scatter(
                x=[0.0, 1.0, 1.0],
                y=[1.0, 1.0, 0.0],
                text='F1 score = 1.0',
                mode='lines+markers',
                name='Ideal PR line',
                line = dict(dash='dash'))
            )

    fig.update_layout(
        title='Prediction Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        xaxis=dict(range=[-0.1, 1.1]),
        yaxis=dict(categoryorder='total ascending'),
    )

    fig.write_html(f'{graph_path}/precision_recall_curve.html')


def f1_score_hist(titles, curves, graph_path):
    fig = go.Figure()

    best_points = [points[1] for points in curves]
    f1s = []
    for i, (precision, recall) in enumerate(best_points):
        f1 = f1_score(precision, recall)
        f1s.append(f1)
        fig.add_trace(go.Bar(
            y=[titles[i]],
            x=[f1],
            orientation='h',
            text=f'{f1:.2f}',
            textposition='inside',
            hoverinfo='x',
            name=titles[i]
        ))

    avg = np.mean(f1s)
    fig.add_trace(go.Bar(
            y=['Average'],
            x=[avg],
            orientation='h',
            text=f'{avg:.2f}',
            textposition='inside',
            hoverinfo='x',
            name='Average'
    ))

    fig.update_layout(
        title='F1 Scores for Benchmarks',
        xaxis_title='F1 Score',
        yaxis_title='Benchmark',
        xaxis=dict(range=[0, 1.1]),
        yaxis=dict(categoryorder='total ascending'),
    )

    fig.write_html(f'{graph_path}/f1_scores.html')


def shd_hist(shd_values, benchmark_titles, graph_path):
    
    shds = shd_values.copy()
    titles = benchmark_titles.copy()

    shds.append(np.mean(shds))
    avg_title = 'Average'
    benchmark_titles.append(avg_title)

    sorted_shds, sorted_titles = zip(*sorted(zip(shds, titles)))

    fig = px.bar(x=sorted_titles, y=sorted_shds, title='Structural Hamming Distance for benchmarks')

    fig.update_xaxes(title_text='Benchmarks')
    fig.update_yaxes(title_text='Structural Hamming Distance')
    fig.write_html(f'{graph_path}/shd_scores.html')





def run_benchmarks(benchmark_algorithm=Algorithm.GPT):

    ground_truth_graphs = [
                            ('Asia_benchmark', ['visit to Asia', 'tubercolosis', 'lung cancer', 'bronchitis', 'dyspnoea', 'smoking', 'positive X-ray'], [('visit to Asia', 'tubercolosis'), ('smoking', 'lung cancer'), ('smoking', 'bronchitis'), ('bronchitis', 'dyspnoea'), ('lung cancer', 'dyspnoea'), ('tubercolosis', 'dyspnoea'), ('lung cancer', 'positive X-ray'), ('tubercolosis', 'positive X-ray')]),
                            ('Smoking_benchmark', ['smoking', 'tobacco fumes', 'lung cancer', 'tumors'], [('smoking', 'tobacco fumes'), ('smoking', 'lung cancer'), ('smoking', 'tumors'), ('tobacco fumes', 'lung cancer'), ('tobacco fumes', 'tumors'), ('lung cancer', 'tumors'), ('tumors', 'lung cancer')]),
                            ('Alcohol_benchmark', ['alcohol', 'liver cirrhosis', 'death'], [('alcohol', 'liver cirrhosis'), ('liver cirrhosis', 'death'), ('alcohol', 'death')]),
                            ('Cancer_benchmark', ['smoking', 'respiratory disease', 'lung cancer', 'asbestos exposure'], [('smoking', 'respiratory disease'), ('respiratory disease', 'lung cancer'), ('asbestos exposure', 'lung cancer'), ('asbestos exposure', 'respiratory disease'), ('smoking', 'lung cancer')]),
                            ('Diabetes_benchmark', ['lack of exercise', 'body weight', 'diabetes', 'diet'], [('lack of exercise', 'body weight'), ('lack of exercise', 'diabetes'), ('body weight', 'diabetes'), ('diet', 'diabetes'), ('diet', 'body weight')]),
                            ('Obesity_benchmark', ['obesity', 'mortality', 'heart failure', 'heart defects'], [('obesity', 'mortality'), ('obesity', 'heart failure'), ('heart failure', 'mortality'), ('heart defects', 'heart failure'), ('heart defects', 'mortality')]),
                            ]

    titles = []
    shds = []
    auprs = []
    curves = []
    pred_edges = []
    pred_cycles = []


    main_directory_path = f'../benchmarks/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    graphs_directory_path = f'{main_directory_path}/graphs'
    os.makedirs(graphs_directory_path, exist_ok=True)
    
    data_list = []

    for title, ground_truth_nodes, ground_truth_edges in ground_truth_graphs:
        nodes, prediction_edges, prediction_cycles = benchmark_prediction(title, ground_truth_nodes, ground_truth_edges, save_graphs=True, graphs_directory=graphs_directory_path, algorithm=benchmark_algorithm, verbose=True)
        shd, aupr, curve, precision, recall, f1, prediction_edges, missing_edges, extra_edges, correct_direction, incorrect_direction = evaluate_predictions(ground_truth_nodes, ground_truth_edges, prediction_edges)
        
        titles.append(title)
        shds.append(shd)
        auprs.append(aupr)
        curves.append(curve)
        pred_edges.append(prediction_edges)
        pred_cycles.append(prediction_cycles)

        print(f'{title} completed:')
        print(f'    SHD                 = {shd}')
        print(f'    Ground Truth edges  = {len(ground_truth_edges)} edges')
        print(f'    Prediction edges    = {len(prediction_edges)} edges')
        print(f'    Missing edges       = {len(missing_edges)} missing edges')
        print(f'    Extra edges         = {len(extra_edges)} extra edges')
        print(f'    Correct direction   = {len(correct_direction)} correct edges')
        print(f'    Incorrect direction = {len(incorrect_direction)} incorrect edges')
        print(f'    Precision           = {precision}')
        print(f'    Recall              = {recall}')
        print(f'    F1 score            = {f1}')
        print(f'    Area PRC            = {aupr}')
        print(f'    PRC point           = {curve}')
        print(f'    Cycles              = {prediction_cycles}')
        print('')

        row_data = {
            'title': title,
            'SHD': shd,
            'Ground Truth edges': len(ground_truth_edges),
            'Prediction edges': len(prediction_edges),
            'Missing edges': len(missing_edges),
            'Extra edges': len(extra_edges),
            'Correct direction': len(correct_direction),
            'Incorrect direction': len(incorrect_direction),
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUPR': aupr,
            'PRC point': curve,
            'Prediction': prediction_edges,
            'Cycles': prediction_cycles
        }
        data_list.append(row_data)

    results = pd.DataFrame(data_list)
    results.to_csv(f'{main_directory_path}/results.csv')

    print('Benchmarks completed')

    precision_recall_curve_plot(titles, curves, main_directory_path)
    f1_score_hist(titles, curves, main_directory_path)
    shd_hist(shds, titles, main_directory_path)