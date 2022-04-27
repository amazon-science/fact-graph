from transformers import AutoTokenizer, AutoConfig
from transformers import ElectraAdapterModel
from torch import nn
import torch
import numpy as np
from utils import transform_graph_geometric

EDGES_AMR = ["have-rel-role", "have-degree", "all-over", "distance-quantity", "date-entity", ":ARG0", ":ARG0-of",
                  ":ARG1", ":ARG1-of", ":ARG2", ":ARG2-of", ":ARG3", ":ARG3-of", ":ARG4",
                  ":ARG4-of", ":ARG5", ":ARG5-of", ":ARG6", ":ARG6-of", ":ARG7", ":accompanier", ":accompanier-of",
                  ":age", ":age-of", ":beneficiary", ":beneficiary-of", ":century", ":concession", ":concession-of",
                  ":condition", ":condition-of", ":conj-as-if", ":consist", ":consist-of", ":day", ":dayperiod",
                  ":dayperiod-of", ":decade", ":degree", ":degree-of", ":destination", ":destination-of", ":direction",
                  ":direction-of", ":domain", ":domain-of", ":duration", ":duration-of", ":era", ":example", ":example-of",
                  ":extent", ":extent-of", ":frequency", ":frequency-of", ":instrument", ":instrument-of", ":li", ":location",
                  ":location-of", ":manner", ":manner-of", ":medium", ":medium-of", ":mod", ":mod-of", ":mode", ":month",
                  ":name", ":op1", ":op1-of", ":op10", ":op11", ":op12", ":op12_<lit>", ":op13", ":op14", ":op14_<lit>_:",
                  ":op15", ":op16", ":op17", ":op18", ":op19", ":op19_<lit>_:", ":op1_<lit>", ":op2", ":op2-of", ":op20",
                  ":op21", ":op22", ":op23", ":op24", ":op25", ":op25_<lit>_:", ":op26", ":op27", ":op27_<lit>_.", ":op28",
                  ":op29", ":op3", ":op3-of", ":op30", ":op31", ":op32", ":op33", ":op34", ":op35", ":op36", ":op37",
                  ":op38", ":op39", ":op4", ":op40", ":op41", ":op5", ":op6", ":op7", ":op8", ":op9", ":ord", ":ord-of",
                  ":part", ":part-of", ":path", ":path-of", ":polarity", ":polarity-of", ":polite", ":poss", ":poss-of",
                  ":prep-a", ":prep-about", ":prep-after", ":prep-against", ":prep-against-of", ":prep-along-to",
                  ":prep-along-with", ":prep-amid", ":prep-among", ":prep-around", ":prep-as", ":prep-at", ":prep-back",
                  ":prep-between", ":prep-by", ":prep-down", ":prep-for", ":prep-from", ":prep-in", ":prep-in-addition-to",
                  ":prep-into", ":prep-of", ":prep-off", ":prep-on", ":prep-on-behalf", ":prep-on-behalf-of", ":prep-on-of",
                  ":prep-on-side-of", ":prep-out-of", ":prep-over", ":prep-past", ":prep-per", ":prep-through", ":prep-to",
                  ":prep-toward", ":prep-under", ":prep-up", ":prep-upon", ":prep-with", ":prep-without", ":purpose", ":purpose-of",
                  ":quant", ":quant-of", ":quant101", ":quant102", ":quant104", ":quant113", ":quant114", ":quant115", ":quant118",
                  ":quant119", ":quant128", ":quant141", ":quant143", ":quant146", ":quant148", ":quant164", ":quant165", ":quant166",
                  ":quant179", ":quant184", ":quant189", ":quant194", ":quant197", ":quant208", ":quant214", ":quant217", ":quant228",
                  ":quant246", ":quant248", ":quant274", ":quant281", ":quant305", ":quant306", ":quant308", ":quant309", ":quant312",
                  ":quant317", ":quant324", ":quant329", ":quant346", ":quant359", ":quant384", ":quant396", ":quant398", ":quant408",
                  ":quant411", ":quant423", ":quant426", ":quant427", ":quant429", ":quant469", ":quant506", ":quant562", ":quant597",
                  ":quant64", ":quant66", ":quant673", ":quant675", ":quant677", ":quant74", ":quant754", ":quant773", ":quant785", ":quant787",
                  ":quant79", ":quant797", ":quant801", ":quant804", ":quant86", ":quant870", ":quarter", ":range", ":scale", ":season",
                  ":snt1", ":snt12", ":snt2", ":snt3", ":snt4", ":snt5", ":snt6", ":snt7", ":snt8", ":source", ":source-of", ":subevent",
                  ":subevent-of", ":time", ":time-of", ":timezone", ":timezone-of", ":topic", ":topic-of", ":unit", ":value", ":weekday",
                  ":weekday-of", ":year", ":year2"]


def load_model_and_tokenizer(type_model, model_name_or_path, number_labels, adapter_size):
    """Load model and tokenizer."""

    config = AutoConfig.from_pretrained(model_name_or_path)

    # number of labels for the classification head
    config.num_labels = number_labels

    # number of attention heads for pooler
    config.pooler_attention_heads = 2
    config.adapter_size = adapter_size

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    new_tokens_vocab = {"additional_special_tokens": []}

    # sort by edge labels
    tokens_amr = sorted(EDGES_AMR, reverse=True)

    # add edges labels to model embeddings matrix
    for t in tokens_amr:
        new_tokens_vocab["additional_special_tokens"].append(t)
    num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)
    print(num_added_toks, "tokens added.")

    if type_model == "factgraph":
        model = FactGraphModel(model_name_or_path, config, tokenizer)
    elif type_model == "factgraph-edge":
        model = FactGraphEdgeLevelModel(model_name_or_path, config, tokenizer)
    else:
        raise Exception("Model not supported.")

    return model, tokenizer


class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn


class MultiheadAttPoolLayer(nn.Module):
    """MultiheadAtt Pooling layer."""

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)

        return output, attn

def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling - Take attention mask into account for correct averaging.
    model_output: tensor of shape (b, l, d_k_original)
    attention_mask: tensor of shape (b, l)
    returns: tensor of shape (b, d_k_original)
    """

    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def max_pooling(model_output, attention_mask):
    """
    Max Pooling - Take the max value over time for every dimension.
    model_output: tensor of shape (b, l, d_k_original)
    attention_mask: tensor of shape (b, l)
    returns: tensor of shape (b, d_k_original)
    """
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]


class FactGraphModel(nn.Module):
    """FactGraph model."""
    def __init__(self, model_name_or_path, config, tokenizer):
        super().__init__()

        print(model_name_or_path)

        self.plm = ElectraAdapterModel.from_pretrained(model_name_or_path, config=config)
        self.plm.resize_token_embeddings(len(tokenizer))

        config.input_dim = config.hidden_size * 2
        self.final_classifier = nn.Linear(config.input_dim, 2)

        self.pooler_g = MultiheadAttPoolLayer(config.pooler_attention_heads, config.hidden_size, config.hidden_size)


    def forward(self, input_ids, attention_mask, input_ids_graphs, attn_graphs, graph_structure=None):

        # encode text [document, summary]
        input_ids = input_ids
        attention_mask = attention_mask
        transformer_outputs = self.plm(input_ids, attention_mask=attention_mask)
        output_text = transformer_outputs.last_hidden_state
        output_text = output_text[:, 0, :]  # take <s> token (equiv. to [CLS])

        # encode graphs
        bz, sents, dim_graph = input_ids_graphs.size()
        input_ids_graphs = input_ids_graphs.view(bz * sents, dim_graph)
        attn_graphs = attn_graphs.view(bz * sents, dim_graph)
        edge_index = graph_structure["edge_index"]
        edge_type = graph_structure["edge_type"]

        # adjust graph edges to batch
        new_edge_index = []
        new_edge_type = []
        for graphs_doc, type_doc in zip(edge_index, edge_type):
            for graph, type_graph in zip(graphs_doc, type_doc):
                new_edge_index.append(graph)
                new_edge_type.append(type_graph)

        # import pdb
        # pdb.set_trace()
        # transform graph into pytorch Geometric format
        graph_batch = transform_graph_geometric(input_ids_graphs, new_edge_index, new_edge_type)

        transformer_outputs = self.plm(input_ids_graphs, attention_mask=attn_graphs, graph=graph_batch)

        # node pooling
        pooling_graphs = mean_pooling(transformer_outputs, attn_graphs)
        pooling_graphs = pooling_graphs.view(bz, sents, -1)

        mask = graph_structure["mask_graph"][:, 1:]
        mask = (mask == 0)

        # pooler of the graph representations
        output_graph, pool_attn = self.pooler_g(pooling_graphs[:, 0, :], pooling_graphs[:, 1:, :])

        # final classification
        output = torch.cat([output_text, output_graph], dim=1)
        logits = self.final_classifier(output)

        return logits


class FactGraphEdgeLevelModel(nn.Module):
    def __init__(self, model_name_or_path, config, tokenizer):
        super().__init__()

        self.plm = ElectraAdapterModel.from_pretrained(model_name_or_path, config=config)
        self.plm.resize_token_embeddings(len(tokenizer))

        self.final_classifier = nn.Linear(4 * config.hidden_size, 2)

        self.dropout = nn.Dropout(0.3)


    def forward(self, input_ids, attention_mask, data_ann=None):


        input_ids = input_ids
        attention_mask = attention_mask
        transformer_outputs = self.plm(input_ids, attention_mask=attention_mask)
        output_text = transformer_outputs.last_hidden_state
        output_text_s = output_text[:, 0, :]  # take <s> token (equiv. to [CLS])

        head, tail, head_mask, tail_mask, head_graph, tail_graph, head_mask_graph, tail_mask_graph, \
        tok, input_ids_graph, attention_mask_graph, edge_index, edge_type, mask_graphs = data_ann


        # encode graphs
        bz, sents, dim_graph = input_ids_graph.size()
        input_ids_graph = input_ids_graph.view(bz * sents, dim_graph)
        attention_mask_graph = attention_mask_graph.view(bz * sents, dim_graph)

        # adjust graph edges to batch
        new_edge_index = []
        new_edge_type = []
        for graphs_doc, type_doc in zip(edge_index, edge_type):
            for graph, type_graph in zip(graphs_doc, type_doc):
                new_edge_index.append(graph)
                new_edge_type.append(type_graph)

        # import pdb
        # pdb.set_trace()

        # transform graph into pytorch Geometric format
        graph_batch = transform_graph_geometric(input_ids_graph, new_edge_index, new_edge_type)

        # import pdb
        # pdb.set_trace()

        transformer_outputs = self.plm(input_ids_graph, attention_mask=attention_mask_graph, graph=graph_batch)
        output_graph = transformer_outputs.last_hidden_state
        output_graph = output_graph.view(bz, sents, dim_graph, -1)
        output_graph_s = output_graph[:, 0, :]  # take <s> token (equiv. to [CLS])

        batch_size = input_ids.size(0)
        add = torch.arange(batch_size) * input_ids.size(1)
        add = add.unsqueeze(1).to("cuda")
        add = add.unsqueeze(1).to("cuda")

        head = head + add
        tail = tail + add

        # import pdb
        # pdb.set_trace()

        outputs_edges = self.dropout(output_text)
        #outputs_edges = output_text
        outputs_edges = outputs_edges.view((-1, outputs_edges.size(-1)))
        head_embeddings = outputs_edges[head]
        tail_embeddings = outputs_edges[tail]
        #rel_embeddings = outputs_edges[rel]

        tail_embeddings = tail_mask.unsqueeze(3) * tail_embeddings
        head_embeddings = head_mask.unsqueeze(3) * head_embeddings

        head_embeddings = torch.sum(head_embeddings, dim=1)
        tail_embeddings = torch.sum(tail_embeddings, dim=1)
        # import pdb
        # pdb.set_trace()

        head_embeddings = head_embeddings.view(batch_size, -1, head_embeddings.size(-1))
        tail_embeddings = tail_embeddings.view(batch_size, -1, tail_embeddings.size(-1))
        #rel_embeddings = rel_embeddings.view(batch_size, -1, rel_embeddings.size(-1))

        # import pdb
        # pdb.set_trace()

        # import pdb
        # pdb.set_trace()

        ##### graph

        input_ids_graph = input_ids_graph.view(bz, sents, dim_graph)
        add = torch.arange(batch_size) * input_ids_graph.size(2)
        add = add.unsqueeze(1).to("cuda")
        add = add.unsqueeze(1).to("cuda")

        head_graph = head_graph + add
        tail_graph = tail_graph + add

        # import pdb
        # pdb.set_trace()

        outputs_edges = self.dropout(output_graph)

        mask_graphs = mask_graphs.unsqueeze(2).to("cuda")
        mask_graphs = mask_graphs.unsqueeze(2).to("cuda")
        outputs_edges = outputs_edges * mask_graphs
        outputs_edges = torch.sum(outputs_edges, dim=1)
        outputs_edges = outputs_edges / mask_graphs.sum(1)
        # import pdb
        # pdb.set_trace()

        #outputs_edges = output_graph
        outputs_edges = outputs_edges.view((-1, outputs_edges.size(-1)))
        head_embeddings_graph = outputs_edges[head_graph]
        tail_embeddings_graph = outputs_edges[tail_graph]

        tail_embeddings_graph = tail_mask_graph.unsqueeze(3) * tail_embeddings_graph
        head_embeddings_graph = head_mask_graph.unsqueeze(3) * head_embeddings_graph

        head_embeddings_graph = torch.sum(head_embeddings_graph, dim=1)
        tail_embeddings_graph = torch.sum(tail_embeddings_graph, dim=1)

        head_embeddings_graph = head_embeddings_graph.view(batch_size, -1, head_embeddings_graph.size(-1))
        tail_embeddings_graph = tail_embeddings_graph.view(batch_size, -1, tail_embeddings_graph.size(-1))

        #####


        final_embeddings = torch.cat([head_embeddings, head_embeddings_graph, tail_embeddings, tail_embeddings_graph], dim=2)
        logits_all = self.final_classifier(final_embeddings)

        # import pdb
        # pdb.set_trace()

        return logits_all