import sys
import argparse
import scraping
import gpt
import pandas as pd
import re
from datetime import datetime
import benchmarks
import json
import os
import time

def sanitize_string(string, max_length=100):
    string = re.sub(r'[\\/:*?"<>|]', '_', string)
    return string[:max_length] if max_length else string


def causal_analysis(data, file_name=None, directory_name=None, causal_discovery_query_for_bidirected_edges=True, perform_edge_explanation=False, optimize_found_entities=True, use_short_abstracts=False, max_abstract_length=200):

    print('CAUSAL ANALYSIS PROCESS')

    print(f'Starting at : {datetime.now().strftime("%H:%M:%S %d/%m/%Y")}')

    if file_name:
        file_name = sanitize_string(file_name)        
        file_name = f'{sanitize_string(file_name).split(".")[0]}.csv'
    else:
        file_name = f'causal_analysis_results.csv'
    
    directory = f'../results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    if directory_name:
        directory = f'{directory} - {directory_name}'

    file_name = f'{directory}/{file_name}'
    graphs_directory = f'{directory}/graphs'
    os.makedirs(directory, exist_ok=True)
    os.makedirs(graphs_directory, exist_ok=True)

    results = pd.DataFrame(columns=['id', 'title', 'abstract', 'exec_time'])
    data_len = len(data)
    print(data_len)

    for i, row in data.iterrows():
        if use_short_abstracts and len(row['abstract'].split(' ')) > max_abstract_length:
            continue

        title = sanitize_string(row['title'], 35)
        article_ref = f'{row["id"]}-{title}'

        print(f'\n-------- {row["title"]} --------\n')
        nodes, directed_edges, bidirected_edges, cycles, elapsed_seconds, edge_explanation = gpt.causal_discovery_pipeline(article_ref, row['abstract'], causal_discovery_query_for_bidirected_edges=causal_discovery_query_for_bidirected_edges, perform_edge_explanation=perform_edge_explanation, use_text_in_causal_discovery=True, use_LLM_pretrained_knowledge_in_causal_discovery=False, reverse_edge_for_variable_check=False, optimize_found_entities=optimize_found_entities, use_text_in_entity_optimization=True, search_cycles=True, plot_static_graph=False, graph_directory_name=graphs_directory, verbose=False)

        new_row = pd.DataFrame({'id': row['id'], 'title': row['title'], 'abstract': row['abstract'], 'exec_time': time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))}, index=[0])
        results = pd.concat([results, new_row]).reset_index(drop=True)
        results.to_csv(file_name, index=False)

        graph_data = {'nodes': nodes, 'directed_edges': directed_edges, 'bidirected_edges':bidirected_edges, 'edge_explanation':edge_explanation, 'cycles': cycles, 'exec_time': time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))}
        with open(f'{graphs_directory}/{article_ref}.json', "w") as json_file:
            json.dump(graph_data, json_file, indent=4)
        
        print(f'{i+1}/{data_len} - {time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))}')


    return results


def pubmed_scraping():
    print('SCRAPING PROCESS')
    print('------------------\n')

    scraping.main()


def scraping_and_causal_analysis():
    data = scraping.main(return_data=True)
    if data is None:
        print('ERROR: No data')
        sys.exit()

    causal_analysis(data)



def run_benchmarks(model=benchmarks.Algorithm.GPT):
    print('BENCHMARKS')
    print('------------------\n')

    benchmarks.run_benchmarks(model)


def evaluate_results(ground_truth, prediction, results_directory=None):
    print('EVALUATE RESULTS')

    try:
        with open(ground_truth, 'r') as json_file:
            gt_graph = json.load(json_file)
    except FileNotFoundError:
        print(f"JSON file not found: {ground_truth}")
        return
    except pd.errors.ParserError:
        print(f"Error parsing JSON file: {ground_truth}")
        return
    try:
        with open(prediction, 'r') as json_file:
            pred_graph = json.load(json_file)
    except FileNotFoundError:
        print(f"JSON file not found: {prediction}")
        return
    except pd.errors.ParserError:
        print(f"Error parsing JSON file: {prediction}")
        return

    directory = f'../evaluations/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}' if results_directory is None else results_directory
    os.makedirs(directory, exist_ok=True)

    shd, aupr, curve, precision, recall, f1, prediction_edges, missing_edges, extra_edges, correct_direction, incorrect_direction = benchmarks.evaluate_predictions(gt_graph['nodes'], gt_graph['edges'], pred_graph['edges'])

    results = pd.DataFrame({
            'Ground Truth Graph': ground_truth,
            'Prediction Graph': prediction,
            'SHD': shd,
            'Ground Truth edges': len(gt_graph['edges']),
            'Prediction edges': len(prediction_edges),
            'Missing edges': len(missing_edges),
            'Extra edges': len(extra_edges),
            'Correct direction': len(correct_direction),
            'Incorrect direction': len(incorrect_direction),
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUPR': aupr,
            'PRC point': [curve],
            'Prediction': [prediction_edges]
        }, index=[0])
    results.to_csv(f'{directory}/results.csv')


def run_example_test():
    print('EXAMPLE TEST')
    print('------------------\n')
    directory = f'../results/TEST - {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(directory , exist_ok=True)

    nodes, directed_edges, bidirected_edges, cycles, elapsed_seconds, edge_explanation = gpt.example_test(directory)

    graph_data = {'nodes': nodes, 'directed_edges': directed_edges, 'bidirected_edges':bidirected_edges, 'edge_explanation':edge_explanation, 'cycles': cycles, 'exec_time': time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))}
    with open(f'{directory}/Example test.json', "w") as json_file:
        json.dump(graph_data, json_file, indent=4)

    print('\n--\nTEST COMPLETE')


def plot_graph(graph_path):
    try:
        with open(graph_path, 'r') as json_file:
            graph = json.load(json_file)
    except FileNotFoundError:
        print(f"JSON file not found: {graph_path}")
        return
    except pd.errors.ParserError:
        print(f"Error parsing JSON file: {graph_path}")
        return
    
    
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if '/' in graph_path:
        graph_name = graph_path.split('/')[-1].split('.')[0]
        directory = '/'.join(graph_path.split('/')[:-1])
    elif '\\' in graph_path:
        graph_name = graph_path.split('\\')[-1].split('.')[0]
        directory = '\\'.join(graph_path.split('\\')[:-1])
    else:
        graph_name = 'mygraph'
        directory = f'../graphs/{datetime_str}'

    graph_name = f'{graph_name} - {datetime_str}'
    

    gpt.build_graph(nodes=graph['nodes'], edges=graph['directed_edges'], bidirected_edges=graph['bidirected_edges'], cycles=graph['cycles'], graph_name=graph_name, directory_name=directory, edge_explanation=graph['edge_explanation'], plot_static_graph=False)


class MyArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        if file is None:
            file = sys.stdout

        custom_help = """
Usage: causal_llms.py <action> [options]

Description:
  This script performs various tasks related to causal discovery.

Actions:
  ex       Run the example test.
  s        Run the scraping process.
  c        Perform causal analysis.
  sc       Run scraping and causal analysis.
  b        Run the benchmark tests.
  e        Evaluate prediction against ground truth graph.
  p        Plot interactive causal graph.

Options:
  --help   Show this help message and exit.

Examples:
  python causal_llms.py ex                               # Run the example test.
  python causal_llms.py s                                # Run the scraping process.
  python causal_llms.py c --data-path </path/to/data>    # Perform causal analysis with specified data path.
  python causal_llms.py sc                               # Run scraping and causal analysis.
  python causal_llms.py b --algorithm {baseline|gpt}     # Run the benchmark tests with the specified algorithm.
  python causal_llms.py e --gt </path/to/ground_truth> --pred </path/to/prediction> # Evaluate prediction.
  python causal_llms.py p --graph-path </path/to/graph>  # Plot interactive causal graph.

The `algorithm` parameter specifies the algorithm to use for the benchmark tests.
The possible values are:
* `baseline`: Baseline algorithm
* `gpt`: GPT LLM
"""
        file.write(custom_help +"\n")



def main():

    parser = MyArgumentParser()
    parser.add_argument("action", choices=["ex", "s", "c", "sc","b", "e", "p", "t"], help="Action to perform.")
    parser.add_argument("--data-path", help="Path to the data for causal analysis.")
    parser.add_argument("--algorithm", help="Algorithm to use for causal analysis on benchmarks.")
    parser.add_argument("--gt", help="Path to the ground truth graph for prediction evaluation.")
    parser.add_argument("--pred", help="Path to the prediction graph for results evaluation.")  
    parser.add_argument("--graph-path", help="Path to the graph-path graph to be plotted.")  


    try:

        args = parser.parse_args()

        if args.action == "b":
            if args.algorithm.upper() in [attr for attr in dir(benchmarks.Algorithm) if attr.isupper()]:
                run_benchmarks(model=getattr(benchmarks.Algorithm, args.algorithm.upper()))
            else:
                run_benchmarks()
        elif args.action == "ex":
            run_example_test()
        elif args.action == "s":
            pubmed_scraping()
        elif args.action == "c":
            if args.data_path:
                data = None
                try:
                    data = pd.read_csv(args.data_path)
                except FileNotFoundError:
                    print(f"CSV file not found: {args.data_path}")
                    return
                except pd.errors.ParserError:
                    print(f"Error parsing CSV file: {args.data_path}")
                    return
                except pd.errors.EmptyDataError:
                    print(f"CSV file is empty: {args.data_path}")
                    return
                except UnicodeDecodeError:
                    print(f"Error decoding CSV file: {args.data_path}")
                    return
                except PermissionError:
                    print(f"Permission error: {args.data_path}")
                    return
                except IOError:
                    print(f"I/O error: {args.data_path}")
                    return

                bidirected_edges=False
                entity_optimization=True
                edge_explanation=True
                causal_analysis(data, causal_discovery_query_for_bidirected_edges=bidirected_edges, perform_edge_explanation=edge_explanation, optimize_found_entities=entity_optimization)
            else:
                print("Please provide the path to the data for causal analysis using the --data-path option.")
        elif args.action == "sc":
            scraping_and_causal_analysis()
        elif args.action == "e":
            evaluate_results(args.gt, args.pred)
        elif args.action == "p":
            plot_graph(args.graph_path)
        elif args.action == "t":
            data_path = '../data/xai_abstracts/'
            files = ['ageing.csv', 'ARoUTI.csv', 'cardiovascular diseases.csv', 'diabetes.csv']
            for file in files:
                name = file.split('.')[0]
                data = pd.read_csv(data_path + file)
                bidirected_edges=False
                entity_optimization=True
                edge_explanation=True
                causal_analysis(data=data, file_name=name, directory_name=name, causal_discovery_query_for_bidirected_edges=bidirected_edges, perform_edge_explanation=edge_explanation, optimize_found_entities=entity_optimization)
        else:
            raise argparse.ArgumentError
    except argparse.ArgumentError:
        print("Invalid action. Use --help for available options.")

if __name__ == "__main__":
    main()