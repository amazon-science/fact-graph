import torch
from tqdm import tqdm
from models import EDGES_AMR
import os
import logging
from glob import glob
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.utils.data import (DataLoader, RandomSampler, WeightedRandomSampler)
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score
from datasets import load_dataset
import random
import argparse
import sys
import math
from utils import maybe_save_checkpoint, save_metrics
from models import load_model_and_tokenizer
from utils import processing_edge_level_data

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def preprocess_function(examples):

    outputs = processing_edge_level_data(examples, tokenizer, num_document_graphs, max_seq_length)

    return outputs


def load_data(data_file):
    dataset = load_dataset('json', data_files=data_file)
    dataset = dataset.map(preprocess_function, batched=True,
                            load_from_cache_file=False,
                          remove_columns=['id', 'article', 'summary', 'summary_tok', 'graph_summary', 'graphs', 'hallucinations', 'hallucination_amr', 'sentences'],
                            num_proc=10)
    dataset.set_format(columns=['input_ids', 'attention_mask', 'head', 'tail', 'head_graph', 'tail_graph', 'label_ann', 'label', 'head_mask',
                                'tail_mask', 'head_mask_graph', 'tail_mask_graph', 'input_ids_graph',
                                'attention_mask_graph', 'edge_index', 'edge_type', 'mask_graph'])
    return dataset


def eval_model(model, dataloader_val, val_data, global_step=0, best_bacc=0):
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

                for idx_edge, edge_pred in enumerate(datapoint):
                    label_edge = input['label_ann'][idx_datapoint][idx_edge].cpu().numpy()
                    if label_edge == -100:
                        continue

                    pred_edge_labels.append(edge_pred)
                    ref_edge_labels.append(label_edge)
                    if edge_pred == 0:
                        pred_datapoint = 0
                preds_sent.append(pred_datapoint)


        labels_sent = np.array(val_data["train"]["label"])
        preds_sent = np.array(preds_sent)
        assert len(labels_sent) == len(preds_sent)

        metrics = {}
        metrics["step"] = global_step
        metrics["accuracy"] = np.round_((preds_sent == labels_sent).astype(np.float32).mean().item(), 4)
        metrics["bacc"] = np.round_(balanced_accuracy_score(y_true=labels_sent, y_pred=preds_sent), 4)

        if metrics["bacc"] > best_bacc:
            best_bacc = metrics["bacc"]

        metrics["best_bacc"] = best_bacc

        metrics["f1"] = np.round_(f1_score(y_true=labels_sent, y_pred=preds_sent, average="micro"), 4)
        metrics["f1_macro"] = np.round_(f1_score(y_true=labels_sent, y_pred=preds_sent, average="macro"), 4)


        labels_edge = np.array(ref_edge_labels)
        preds_edge = np.array(pred_edge_labels)
        assert len(labels_edge) == len(preds_edge)

        metrics["accuracy_edge"] = np.round_((preds_edge == labels_edge).astype(np.float32).mean().item(), 4)
        metrics["bacc_edge"] = np.round_(balanced_accuracy_score(y_true=labels_edge, y_pred=preds_edge), 4)
        metrics["f1_edge"] = np.round_(f1_score(y_true=labels_edge, y_pred=preds_edge, average="micro"), 4)

        return best_bacc, metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    # import pdb
    # pdb.set_trace()

    data = {}
    for key in batch[0].keys():
        data[key] = [item[key] for item in batch]

    for key in data.keys():
        if key != 'edge_index' and key != 'edge_type':
            data[key] = torch.tensor(data[key], dtype=torch.long).cuda()

    return data


def train(model, train_data, val_data, save_dir, test_data=None, lr: float = 1e-4, warmup: float = 0,
          num_epoch: int = 5, save_every_k_step: int = 50, cuda=True, batch_size=4):

    logging.info("num epochs: %s", num_epoch)
    logging.info("leaning rate: %s", lr)
    logging.info("batch size: %s", batch_size)
    logging.info("warmup: %s", warmup)

    weighted_training = True

    if weighted_training:
        num_neg = 0.
        num_pos = 0.
        for tensor in train_data['train']:
            sent_label = tensor['label']
            if sent_label == 0:
                num_neg += 1
            else:
                num_pos += 1
            # print(sent_label)

        weights = []
        w_neg = (num_pos * 10) / (num_pos + num_neg)
        w_pos = (num_neg * 10) / (num_pos + num_neg)
        for tensor in train_data['train']:
            sent_label = tensor['label']
            if sent_label == 0:
                weights.append(w_neg)
            else:
                weights.append(w_pos)

        train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights) * 5)
    else:
        train_sampler = RandomSampler(train_data['train'])
    train_dataloader = DataLoader(train_data['train'],
                                  sampler=train_sampler,
                                  batch_size=batch_size, collate_fn=collate_fn)

    dataloader_val = DataLoader(val_data['train'], batch_size=8, collate_fn=collate_fn)

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

    logging.info("Frozen parameters: %f", params_name_frozen)
    logging.info("Learned parameters: %f", params_name)
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
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps
    # )
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps
    )


    global_step = 0
    model.zero_grad()
    model.train()

    max_grad_norm = 1

    # TODO: allow setting device
    if cuda:
        model.to("cuda")

    best_bacc = 0
    for epc in range(num_epoch):
        logging.info("Epoch #{}: \n".format(epc))
        epoch_iterator = tqdm(train_dataloader, desc="Training Steps")

        for step, input in enumerate(epoch_iterator):
            # import pdb
            # pdb.set_trace()

            input_ids = input['input_ids']
            attn = input['attention_mask']
            data_ann = [input['head'], input['tail'], input['head_mask'], input['tail_mask'],
                        input['head_graph'], input['tail_graph'], input['head_mask_graph'], input['tail_mask_graph'],
                        tokenizer, input['input_ids_graph'], input['attention_mask_graph'],
                        input['edge_index'], input['edge_type'], input['mask_graph']]

            output_logits = model(input_ids, attn, data_ann)

            output_logits = output_logits.view((-1, output_logits.size(-1)))

            entailment_lbl = input['label_ann']
            entailment_lbl = entailment_lbl.view((-1))

            entailment_loss_f = torch.nn.CrossEntropyLoss()

            entailment_loss = entailment_loss_f(output_logits, entailment_lbl)

            loss = entailment_loss

            loss.backward()

            model.plm.embeddings.word_embeddings.weight.grad[:-len(EDGES_AMR)] = 0
            model.plm.embeddings.position_embeddings.weight.grad[:-len(EDGES_AMR)] = 0
            model.plm.embeddings.token_type_embeddings.weight.grad[:-len(EDGES_AMR)] = 0

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            epoch_iterator.set_description("loss={:.8f}, lr={:.5E}".format(loss.item(), optimizer.param_groups[0]['lr']))

            global_step += 1

            if global_step % save_every_k_step == 0:
                best_bacc, metrics = eval_model(model, dataloader_val, val_data, global_step, best_bacc)

                metrics['loss'] = loss.item()
                logging.info(metrics)

                model.train()

                save_metrics(metrics, os.path.join(save_dir, "log_metrics.json"))
                maybe_save_checkpoint(metrics, save_dir, global_step, model, tokenizer, "bacc")
    best_metric, metrics = eval_model(model, dataloader_val, val_data, global_step, best_bacc)
    metrics['loss'] = loss.item()
    logging.info(metrics)
    save_metrics(metrics, os.path.join(save_dir, "log_metrics.json"))
    maybe_save_checkpoint(metrics, save_dir, global_step, model, tokenizer, "bacc")
    if test_data:
        test(save_dir, test_data)


def test(path_model, test_data, cuda=True):

    logging.info("test file: %s", test_data)
    test_data = load_data(test_data)
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
    parser.add_argument('--num_epochs', type=int, help="epochs")
    parser.add_argument("--num_labels", type=int, help="number of labels for classification head", default=2)
    parser.add_argument("--pretrained_model_adapters", type=str, help="path pretrained model")
    parser.add_argument("--pretrained_model_graph_adapters", type=str, help="path pretrained model")
    parser.add_argument("--num_document_graphs", type=int, help="number of document graphs", default=5)
    parser.add_argument('--seed', type=int, help="seed", default=200)
    args = parser.parse_args()

    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer("factgraph-edge", args.model_name_or_path, args.num_labels, args.adapter_size)

    logging.info(model)

    max_seq_length = tokenizer.model_max_length
    num_document_graphs = args.num_document_graphs

    if args.test:
        test(args.save_dir, args.test_data_file)

    else:

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
            pretrained_dict_graph = {k.replace(".electra", ""): v for k, v in pretrained_dict.items() if
                                     'adapter_graph' in k or '.embeddings' in k}
            model_dict.update(pretrained_dict_graph)
            model.load_state_dict(model_dict)

        max_seq_length = tokenizer.model_max_length
        train_data = load_data(args.train_data_file)

        val_data = load_data(args.val_data_file)

        logging.info("train file: %s", args.train_data_file)
        logging.info("dev file: %s", args.val_data_file)
        logging.info("dir: %s", args.save_dir)
        logging.info("seed: %s", args.seed)
        logging.info("adapter size: %s", args.adapter_size)
        logging.info("number of document graphs: %s", num_document_graphs)

        train(model, train_data, val_data, args.save_dir, test_data=args.test_data_file,
              save_every_k_step=args.save_every_k_step, batch_size=args.batch_size, num_epoch=args.num_epochs)