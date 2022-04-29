import json
import torch
import os
import logging
import sys
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import (DataLoader, RandomSampler)
from datasets import load_dataset
import random
import math
from utils import maybe_save_checkpoint, save_metrics
from models import load_model_and_tokenizer
from preprocess import align_triples, generate_edge_tensors
import numpy as np
import argparse
from tqdm import tqdm
from models import EDGES_AMR
from utils import calculate_metrics
from glob import glob

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

label_to_id = {"CORRECT": 1, "INCORRECT": 0}

def preprocess_function(examples):

    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    max_seq_length = 512
    args = (examples["summary"], examples["article"])
    result = tokenizer(*args, padding="max_length", max_length=max_seq_length, truncation=True)

    result["graphs"] = []
    result["graphs_attn"] = []
    result["mask_graph"] = []
    result["edge_index"] = []
    result["edge_type"] = []

    max_seq_length = 300

    for graphs, graph_summary in zip(examples["graphs"], examples["graph_summary"]):

        if graph_summary["amr_simple"]:
            claim_graph_string_tok = tokenizer.tokenize("[CLS] " + graph_summary["amr_simple"] + " [SEP]")
            triples_claim_graph = align_triples(claim_graph_string_tok, "[CLS] " + graph_summary["amr_simple"] + " [SEP]",
                                                    json.loads(graph_summary["triples"]), 1)
        else:
            logging.warning("No claim graph.")
            claim_graph_string_tok = "[CLS] [SEP]".split()
            triples_claim_graph = []

        input_ids_graph_claim = tokenizer.convert_tokens_to_ids(claim_graph_string_tok)

        assert len(claim_graph_string_tok) == len(input_ids_graph_claim), "length mismatched"

        padding_length_a = max_seq_length - len(claim_graph_string_tok)
        input_ids_graph_claim = input_ids_graph_claim + ([pad_token] * padding_length_a)
        graph_claim_attention_mask = [1] * len(claim_graph_string_tok) + ([0] * padding_length_a)

        input_ids_graph_claim = input_ids_graph_claim[:max_seq_length]
        graph_claim_attention_mask = graph_claim_attention_mask[:max_seq_length]

        edge_index, edge_types = generate_edge_tensors(triples_claim_graph, max_seq_length)

        enc_graphs = [input_ids_graph_claim]
        enc_graphs_attn = [graph_claim_attention_mask]
        mask_graphs = [1]
        graph_edge_index = [edge_index]
        graph_edge_type = [edge_types]

        # tokenize document graphs
        for g in graphs[:num_document_graphs]:
            doc_graph_string_tok = tokenizer.tokenize("[CLS] " + g["amr_simple"] + " [SEP]")

            triples = json.loads(g["triples"])
            all_triples = align_triples(doc_graph_string_tok, "[CLS] " + g["amr_simple"] + " [SEP]", triples, 1)
            edge_index, edge_types = generate_edge_tensors(all_triples, max_seq_length)

            input_ids_graph = tokenizer.convert_tokens_to_ids(doc_graph_string_tok)
            assert len(doc_graph_string_tok) == len(input_ids_graph), "length mismatched"
            padding_length_a = max_seq_length - len(doc_graph_string_tok)
            input_ids_graph = input_ids_graph + ([pad_token] * padding_length_a)
            graph_attention_mask = [1] * len(doc_graph_string_tok) + ([0] * padding_length_a)

            input_ids_graph = input_ids_graph[:max_seq_length]
            graph_attention_mask = graph_attention_mask[:max_seq_length]

            enc_graphs.append(input_ids_graph)
            enc_graphs_attn.append(graph_attention_mask)
            mask_graphs.append(1)
            graph_edge_index.append(edge_index)
            graph_edge_type.append(edge_types)

        # pad document graphs
        while len(enc_graphs) < num_document_graphs + 1:
            doc_sent_tok = tokenizer.tokenize("[CLS] [SEP]")
            padding_length_a = max_seq_length - len(doc_sent_tok)
            input_ids_sent = tokenizer.convert_tokens_to_ids(doc_sent_tok)
            input_ids_sent = input_ids_sent + ([pad_token] * padding_length_a)
            sent_attention_mask = [1] * len(doc_sent_tok) + ([0] * padding_length_a)
            enc_graphs.append(input_ids_sent)
            enc_graphs_attn.append(sent_attention_mask)
            mask_graphs.append(0)
            edge_index, edge_types = generate_edge_tensors([], max_seq_length)
            graph_edge_index.append(edge_index)
            graph_edge_type.append(edge_types)

        result["graphs"].append(enc_graphs)
        result["graphs_attn"].append(enc_graphs_attn)
        result["mask_graph"].append(mask_graphs)
        result["edge_index"].append(graph_edge_index)
        result["edge_type"].append(graph_edge_type)

    result["label"] = [label_to_id[l] for l in examples["label"]]

    assert len(result["graphs"]) == len(result["graphs_attn"]) == len(result["label"])\
           == len(result["input_ids"]) == len(result["edge_index"]) == len(result["edge_type"])

    return result


def load_data(data_file, dev=False):
    logging.info("loading file...")
    dataset = load_dataset("json", data_files=data_file)
    logging.info("preprocessing...")
    if dev:
        dataset = dataset.map(preprocess_function, batched=True,
                              load_from_cache_file=False,
                              remove_columns=["id", "article", "summary", "graphs", "graph_summary"],
                              num_proc=1)
    else:
        dataset = dataset.map(preprocess_function, batched=True,
                                load_from_cache_file=False,
                                remove_columns=["id", "article", "summary"],
                                num_proc=8)


    dataset.set_format(columns=["input_ids", "attention_mask", "graphs", "graphs_attn",
                                "label", "mask_graph", "edge_index", "edge_type"])
    return dataset




def eval_model(model, dataloader_val, val_data, global_step=None, best_metric=None):
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

        best_metric, metrics = calculate_metrics(global_step, pred_labels, val_data, best_metric)

        return best_metric, metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """
       data: is a list of tuples with (example, label, length)
             where "example" is a tensor of arbitrary shape
             and label/length are scalars
    """

    data = {}
    for key in batch[0].keys():
        data[key] = [item[key] for item in batch]

    for key in data.keys():
        if key != "edge_index" and key != "edge_type":
            data[key] = torch.tensor(data[key], dtype=torch.long).cuda()

    return data


def train(model, training_data, validation_data, save_dir, test_data=None, lr: float = 1e-4,
          warmup: float = 0, num_epoch: int = 5, save_every_k_step: int = 50000,
          cuda=True, batch_size=4):

    # First check save_dir, if not empty, throw warning
    assert os.path.exists(save_dir), "save_dir does not exist!"
    assert len(os.listdir(save_dir)) == 0, "save_dir must be empty!"

    logging.info("num epochs: %s", num_epoch)
    logging.info("leaning rate: %s", lr)
    logging.info("batch size: %s", batch_size)
    logging.info("warmup: %s", warmup)

    train_sampler = RandomSampler(training_data["train"])
    train_dataloader = DataLoader(training_data["train"],
                                  sampler=train_sampler,
                                  batch_size=batch_size, collate_fn=collate_fn)

    dataloader_val = DataLoader(validation_data["train"], batch_size=8, collate_fn=collate_fn)

    params = []
    params_name = []
    params_name_frozen = []

    num_params = 0
    num_params_frozen = 0

    trained_params = ["adapter", "final_classifier", "pooler_g", "embeddings"]

    for n, p in model.named_parameters():
        trained = False
        for trained_param in trained_params:
            if trained_param in n:
                if trained_param != 'embeddings':
                    num_params += p.numel()
                    trained = True
                params.append(p)
                params_name.append(n)
        if not trained:
            num_params_frozen += p.numel()
            params_name_frozen.append(n)

    logging.info("Frozen parameters: %s", params_name_frozen)
    logging.info("Learned parameters: %s", params_name)
    logging.info("# parameters: %f", num_params)
    logging.info("# frozen parameters: %f", num_params_frozen)
    logging.info("percentage learned parameters: %.4f", num_params / num_params_frozen)

    num_steps = num_epoch * math.ceil(len(train_dataloader))
    weight_decay = 0
    warmup_steps = warmup
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in zip(params_name, params) if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in zip(params_name, params) if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps
    )

    global_step = 0
    model.zero_grad()
    model.train()

    # TODO: allow setting device
    if cuda:
        model.to("cuda")

    best_metric = [0, 0]
    for epc in range(num_epoch):
        logging.info("Epoch #{}: \n".format(epc))
        epoch_iterator = tqdm(train_dataloader, desc="Training Steps")

        for step, input in enumerate(epoch_iterator):


            # call model
            graph_structure = {"num_graphs": num_document_graphs + 1, "mask_graph": input["mask_graph"],
                               "edge_index": input["edge_index"], "edge_type": input["edge_type"]}
            output_logits = model(input["input_ids"], input["attention_mask"], input["graphs"],
                                  input["graphs_attn"], graph_structure)


            # calculate loss
            entailment_lbl = input["label"]
            entailment_loss_f = torch.nn.CrossEntropyLoss()
            if cuda:
                entailment_lbl = entailment_lbl.to("cuda")
            entailment_loss = entailment_loss_f(output_logits, entailment_lbl)
            loss = entailment_loss
            loss.backward()

            # only train edge embeddings
            model.plm.embeddings.word_embeddings.weight.grad[:-len(EDGES_AMR)] = 0
            model.plm.embeddings.position_embeddings.weight.grad[:-len(EDGES_AMR)] = 0
            model.plm.embeddings.token_type_embeddings.weight.grad[:-len(EDGES_AMR)] = 0

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            epoch_iterator.set_description("loss={:.8f}, lr={:.5E}".format(loss.item(), optimizer.param_groups[0]["lr"]))

            global_step += 1

            # evaluate and save
            if global_step % save_every_k_step == 0:
                best_metric, metrics = eval_model(model, dataloader_val, validation_data, global_step, best_metric)
                metrics['loss'] = loss.item()
                logging.info(metrics)
                save_metrics(metrics, os.path.join(save_dir, "log_metrics.json"))
                maybe_save_checkpoint(metrics, save_dir, global_step, model, tokenizer, "avg_bacc")
                model.train()

    best_metric, metrics = eval_model(model, dataloader_val, validation_data, global_step, best_metric)
    metrics['loss'] = loss.item()
    logging.info(metrics)
    save_metrics(metrics, os.path.join(save_dir, "log_metrics.json"))
    maybe_save_checkpoint(metrics, save_dir, global_step, model, tokenizer, "avg_bacc")
    if test_data:
        test(save_dir, test_data)


def test(path_model, test_data, cuda=True):

    logging.info("test file: %s", test_data)
    test_data = load_data(test_data, dev=True)
    dirs = list(glob(path_model + "/*/"))
    if not dirs:
        raise Exception("Model not found.")

    path_model = dirs[0]
    model = torch.load(path_model + "/model.pt")

    if cuda:
        model.to("cuda")

    dataloader_test = DataLoader(test_data["train"], batch_size=8, collate_fn=collate_fn)
    _, metrics = eval_model(model, dataloader_test, test_data)
    logging.info('test: %s', metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for the entailment model")
    parser.add_argument("--model_name_or_path", type=str, help="Name or path of the huggingface model/checkpoint to use")
    parser.add_argument("--train_data_file", type=str, help="Path pre-tokenized data")
    parser.add_argument("--val_data_file", type=str, help="Path pre-tokenized data")
    parser.add_argument("--test_data_file", type=str, help="Path pre-tokenized data")
    parser.add_argument("--test", help="test model", action="store_true")
    parser.add_argument("--save_dir", type=str, help="Directory to save the model")
    parser.add_argument("--save_every_k_step", type=int, help="Every step save")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--adapter_size", type=int, help="adapter size", default=32)
    parser.add_argument("--num_epoch", type=int, help="number of epochs")
    parser.add_argument("--num_labels", type=int, help="number of labels for classification head", default=2)
    parser.add_argument("--num_document_graphs", type=int, help="number of document graphs", default=5)
    parser.add_argument("--pretrained_model_adapters", type=str, help="path pretrained model")
    parser.add_argument("--pretrained_model_graph_adapters", type=str, help="path pretrained model")
    parser.add_argument("--seed", type=int, help="seed", default=200)
    args = parser.parse_args()

    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer("factgraph", args.model_name_or_path, args.num_labels, args.adapter_size)

    logging.info(model)

    max_seq_length = tokenizer.model_max_length
    num_document_graphs = args.num_document_graphs

    if args.test:
        test(args.save_dir, args.test_data_file)

    else:
        # load adapters
        if args.pretrained_model_adapters:
            # load text model partially
            model_pre_text = torch.load(args.pretrained_model_adapters)
            pretrained_dict = model_pre_text
            model_dict = model.state_dict()
            pretrained_dict_text = {k: v for k, v in pretrained_dict.items() if 'adapter.' in k or 'adapter_bottom.' in k}
            model_dict.update(pretrained_dict_text)
            model.load_state_dict(model_dict)

        if args.pretrained_model_graph_adapters:
            # load graph model partially
            model_pre = torch.load(args.pretrained_model_graph_adapters)
            pretrained_dict = model_pre
            model_dict = model.state_dict()
            pretrained_dict_graph = {k.replace(".electra", ""): v for k, v in pretrained_dict.items() if 'adapter_graph' in k or '.embeddings' in k}
            model_dict.update(pretrained_dict_graph)
            model.load_state_dict(model_dict)


        train_data = load_data(args.train_data_file)
        val_data = load_data(args.val_data_file, dev=True)

        logging.info("train file: %s", args.train_data_file)
        logging.info("dev file: %s", args.val_data_file)
        logging.info("dir: %s", args.save_dir)
        logging.info("seed: %s", args.seed)
        logging.info("adapter size: %s", args.adapter_size)
        logging.info("number of document graphs: %s", num_document_graphs)

        train(model, train_data, val_data, args.save_dir, test_data=args.test_data_file, num_epoch=args.num_epoch,
              save_every_k_step=args.save_every_k_step, batch_size=args.batch_size)