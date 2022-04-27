import unidecode
import torch

def align_triples(graph_string_tok, graph_string, triples, size_claim_graph=None):
    graph_string = graph_string.split()

    map = {}
    idx_graph_string = 0
    map[idx_graph_string] = []

    try:
        tokenized_node = ""
        for idx, tok in enumerate(graph_string_tok):

            tokenized_node += unidecode.unidecode(tok.replace("##", "").lower())
            unaccented_string = unidecode.unidecode(graph_string[idx_graph_string].lower())

            if not unaccented_string.startswith(tokenized_node):
                idx_graph_string += 1
                map[idx_graph_string] = []
                tokenized_node = ""
                tokenized_node += tok.replace("##", "")

            map[idx_graph_string].append(idx)

        assert len(map) == len(graph_string)

        # check if the tokenized graph is aligned
        for k in map.keys():
            original_word = graph_string[k]
            original_word = unidecode.unidecode(original_word)

            recovered_node = ""
            for tok_idx in map[k]:
                tok = graph_string_tok[tok_idx]
                recovered_node += tok.replace("##", "")

            recovered_node = unidecode.unidecode(recovered_node)
            assert original_word.lower() == recovered_node.lower()


        # update triples with tokenized graph
        updated_triples = []
        for t in triples:
            if size_claim_graph:
                heads = map[t[0] + size_claim_graph]
                tails = map[t[1] + size_claim_graph]
            else:
                heads = map[t[0]]
                tails = map[t[1]]

            relation = t[2]

            for head in heads:
                for tail in tails:
                    updated_triples.append((head, tail, relation))

    except:
        raise Exception("Error when converting graph to tokenized version.")

    return updated_triples


def update_triples(all_triples, triples, graph_string):
    size_string = len(graph_string.split())

    updated_triples = []
    for t in triples:
        updated_triples.append((t[0] + size_string, t[1] + size_string, t[2]))

    return all_triples + updated_triples


def generate_edge_tensors(triples, max_seq_length_graph):

    set_edges = {"d": 0, "r": 1}

    edge_index_head = []
    edge_index_tail = []
    edge_types = []

    for t in triples:
        head = t[0]
        tail = t[1]
        relation = t[2]

        if head >= max_seq_length_graph or tail >= max_seq_length_graph:
            continue

        edge_index_head.append(head)
        edge_index_tail.append(tail)
        edge_types.append(set_edges[relation])

    edge_index = torch.tensor([edge_index_head, edge_index_tail], dtype=torch.long)
    edge_types = torch.tensor(edge_types, dtype=torch.long)

    return edge_index, edge_types



