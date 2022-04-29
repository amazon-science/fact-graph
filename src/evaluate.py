import torch
import logging
import sys
from torch.utils.data import (DataLoader, RandomSampler)
from datasets import load_dataset
import random
from models import load_model_and_tokenizer
import numpy as np
import argparse
from tqdm import tqdm
from utils import calculate_metrics
from glob import glob
from utils_evaluate import preprocess_function, preprocess_edge_function

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def preprocess_factgraph(examples):
    result = preprocess_function(examples, tokenizer, num_document_graphs)

    return result


def preprocess_factgraph_edge(examples):
    result = preprocess_edge_function(examples, tokenizer, num_document_graphs)

    return result


def load_data(data_file, model_type):
    logging.info("loading file...")
    dataset = load_dataset("json", data_files=data_file)
    logging.info("preprocessing...")
    if model_type == 'factgraph':
        dataset = dataset.map(preprocess_factgraph, batched=True,
                              load_from_cache_file=False,
                              remove_columns=["graphs", "graph_summary"],
                              num_proc=1)
        dataset.set_format(columns=["input_ids", "attention_mask", "graphs", "graphs_attn",
                                    "mask_graph", "edge_index", "edge_type"])
    else:
        dataset = dataset.map(preprocess_factgraph_edge, batched=True,
                              load_from_cache_file=False,
                              remove_columns=["graphs", "graph_summary", "summary", "article"],
                              num_proc=1)
        dataset.set_format(
            columns=['input_ids', 'attention_mask', 'head', 'tail', 'head_graph', 'tail_graph', 'label_ann',
                     'head_mask', 'tail_mask', 'head_mask_graph', 'tail_mask_graph', 'input_ids_graph',
                     'attention_mask_graph', 'edge_index', 'edge_type', 'mask_graph'])



    return dataset




def eval_factgraph(model, dataloader_val, val_data):
    label_to_id = {1: "CORRECT", 0: "INCORRECT"}
    model.eval()
    with torch.no_grad():
        pred_labels = []
        for step, input in enumerate(tqdm(dataloader_val)):

            graph_structure = {"num_graphs": num_document_graphs + 1, "mask_graph": input["mask_graph"],
                               "edge_index": input["edge_index"], "edge_type": input["edge_type"]}

            outputs = model(input["input_ids"], input["attention_mask"], input["graphs"],
                            input["graphs_attn"], graph_structure)

            preds = outputs
            preds = np.argmax(preds.cpu().numpy(), axis=1)

            pred_labels.extend(preds)

        summaries = val_data["train"]['summary']
        articles = val_data["train"]['article']

        assert len(summaries) == len(articles) == len(pred_labels)

        for summary, article, pred in zip(summaries, articles, pred_labels):
            print('article:', article)
            print('summary:', summary)
            print('Prediction:', label_to_id[int(pred)])
            print("--")


def eval_factgraph_edge(model, dataloader_val, val_data):
    label_to_id = {1: "CORRECT", 0: "INCORRECT"}
    model.eval()
    with torch.no_grad():

        pred_edge_labels = []
        ref_edge_labels = []
        preds_sent = []
        for step, input in enumerate(tqdm(dataloader_val)):

            input_ids = input['input_ids']
            attn = input['attention_mask']
            data_ann = [input['head'], input['tail'], input['head_mask'], input['tail_mask'],
                        input['head_graph'], input['tail_graph'], input['head_mask_graph'], input['tail_mask_graph'],
                        tokenizer, input['input_ids_graph'], input['attention_mask_graph'],
                        input['edge_index'], input['edge_type'], input['mask_graph']]

            outputs = model(input_ids, attn, data_ann)

            edge_preds = outputs
            edge_preds = np.argmax(edge_preds.cpu().numpy(), axis=2)

            for idx_datapoint, datapoint in enumerate(edge_preds):
                # correct prediction
                pred_datapoint = 1
                pred_edge_label = []
                for idx_edge, edge_pred in enumerate(datapoint):
                    label_edge = input['label_ann'][idx_datapoint][idx_edge].cpu().numpy()
                    if label_edge == -100:
                        continue

                    pred_edge_label.append(label_to_id[int(edge_pred)])
                    if edge_pred == 0:
                        pred_datapoint = 0
                preds_sent.append(pred_datapoint)
                pred_edge_labels.append(pred_edge_label)


        head_nodes_graph = val_data['train']['head_nodes_graph']
        tail_nodes_graph = val_data['train']['tail_nodes_graph']
        head_all_words = val_data['train']['head_words']
        tail_all_words = val_data['train']['tail_words']
        summaries = val_data['train']['summary']
        articles = val_data['train']['article']
        preds_sent = np.array(preds_sent)

        for label_sum, summary, article, label_edges, head_nodes, tail_nodes, head_words, tail_words in \
                zip(preds_sent, summaries, articles, pred_edge_labels, head_nodes_graph, tail_nodes_graph,
                    head_all_words, tail_all_words):
            print("Article:", article)
            print("Summary:", summary)

            for label, head_n, tail_n, head_w, tail_w in zip(label_edges, head_nodes, tail_nodes, head_words, tail_words):
                print('head node:', ' '.join(head_n), ', tail node:', ' '.join(tail_n))
                print('head word(s):', ' '.join(head_w), ', tail word(s):', ' '.join(tail_w))
                print('prediction:', label)
                print("")
            print("Sentence level predction:", label_to_id[label_sum])
            print("---")
            print("")




def collate_fn(batch):
    """
       data: is a list of tuples with (example, label, length)
             where "example" is a tensor of arbitrary shape
             and label/length are scalars
    """

    data = {}
    for key in batch[0].keys():
        data[key] = [item[key] for item in batch]

    ignore_fields = {'edge_index', 'edge_type', 'summary', 'article', 'head_nodes_graph',
                     'tail_nodes_graph', 'head_words', 'tail_words'}
    for key in data.keys():
        if key not in ignore_fields:
            data[key] = torch.tensor(data[key], dtype=torch.long).cuda()

    return data



def test(path_model, model_type, test_data, cuda=True):

    dirs = list(glob(path_model + "/*/"))
    if not dirs:
        raise Exception("Model not found.")

    path_model = dirs[0]
    model = torch.load(path_model + "/model.pt")

    if cuda:
        model.to("cuda")

    dataloader_test = DataLoader(test_data["train"], batch_size=8, collate_fn=collate_fn)

    if model_type == 'factgraph':
        eval_factgraph(model, dataloader_test, test_data)
    else:
        eval_factgraph_edge(model, dataloader_test, test_data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for the entailment model")
    parser.add_argument("--model_name_or_path", type=str, help="Name or path of the huggingface model/checkpoint to use")
    parser.add_argument("--test_data_file", type=str, help="Path pre-tokenized data")
    parser.add_argument("--model_dir", type=str, help="Directory to load the model")
    parser.add_argument("--model_type", type=str, help="Type of model")
    parser.add_argument("--save_every_k_step", type=int, help="Every step save")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--adapter_size", type=int, help="adapter size", default=32)
    parser.add_argument("--num_epoch", type=int, help="number of epochs")
    parser.add_argument("--num_labels", type=int, help="number of labels for classification head", default=2)
    parser.add_argument("--num_document_graphs", type=int, help="number of document graphs", default=5)
    args = parser.parse_args()

    logging.info("model type: %s", args.model_type)
    logging.info("model dir: %s", args.model_dir)
    model, tokenizer = load_model_and_tokenizer(args.model_type, args.model_name_or_path, args.num_labels, args.adapter_size)

    max_seq_length = tokenizer.model_max_length
    num_document_graphs = args.num_document_graphs

    test_data = args.test_data_file

    logging.info("test file: %s", test_data)
    test_data = load_data(test_data, args.model_type)

    test(args.model_dir, args.model_type, test_data)

