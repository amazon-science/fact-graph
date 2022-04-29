import re
import penman
import torch
from torch import nn
import os
import json
import csv
from collections import defaultdict
import spacy
nlp = spacy.load("en_core_web_sm")

TYPES_AMR = ['person', 'family', 'animal', 'language', 'nationality', 'ethnic-group', 'regional-group',
         'political-movement', 'religious-group', 'organization', 'company', 'government-organization',
         'military', 'criminal-organization', 'political-party', 'market-sector',
         'school', 'university', 'research-institute', 'team', 'league', 'location', 'city', 'city-district',
         'county', 'state', 'province', 'territory', 'country', 'local-region',
         'country-region', 'world-region', 'continent', 'ocean', 'sea', 'lake', 'river', 'gulf', 'bay',
        'strait', 'canal', 'peninsula', 'mountain', 'volcano', 'valley',
        'facility', 'airport', 'station', 'port', 'tunnel', 'bridge', 'road', 'railway-line', 'canal', 'building',
         'theater', 'museum', 'palace', 'hotel', 'worship-place', 'market', 'sports-facility', 'park', 'zoo', 'amusement-park',
        'event', 'incident', 'natural-disaster', 'earthquake', 'war', 'conference', 'game', 'festival',
        'product', 'vehicle', 'ship', 'aircraft', 'aircraft-type', 'spaceship', 'car-make', 'work-of-art', 'picture',
        'music', 'show', 'broadcast-program', 'have-org-role-91',
        'publication', 'book', 'newspaper', 'magazine', 'journal', 'natural-object',
        'canyon', 'island', 'desert', 'forest moon', 'planet', 'star', 'constellation',
        'award', 'law', 'court-decision', 'treaty', 'music-key', 'musical-note', 'food-dish', 'writing-script', 'variable', 'program']

def simplify(tokens, v2c):
    SENSE_PATTERN = re.compile('-[0-9][0-9]$')
    mapping = {}
    new_tokens = []
    for idx, tok in enumerate(tokens):
        # ignore instance-of
        if tok.startswith('('):
            new_tokens.append('(')
            last_map = tok.replace("(", "")
            continue
        elif tok == '/':
            save_map = True
            continue
        # predicates, we remove any alignment information and parenthesis
        elif tok.startswith(':'):

            new_tok = tok.strip(')')
            new_tok = new_tok.split('~')[0]
            new_tokens.append(new_tok)

            count_ = tok.count(')')
            for _ in range(count_):
                new_tokens.append(')')

        # concepts/reentrancies, treated similar as above
        else:
            new_tok = tok.strip(')')
            new_tok = new_tok.split('~')[0]

            if new_tok == "":
                continue

            # now we check if it is a concept or a variable (reentrancy)
            if new_tok in v2c:
                # reentrancy: replace with concept
                if new_tok not in mapping:
                    mapping[new_tok] = set()
                mapping[new_tok].add(len(new_tokens))

                if v2c[new_tok] is not None:
                    new_tok = v2c[new_tok]


            # check number
            elif new_tok.isnumeric():
                new_tok = new_tok
            # remove sense information
            elif re.search(SENSE_PATTERN, new_tok):
                new_tok = new_tok[:-3]
            # remove quotes
            elif new_tok[0] == '"' and new_tok[-1] == '"':
                new_tok = new_tok[1:-1]

            if new_tok != "":
                new_tokens.append(new_tok)

            if save_map:
                if last_map not in mapping:
                    mapping[last_map] = set()

                mapping[last_map].add(len(new_tokens) - 1)
                save_map = False

            count_ = tok.count(')')
            for _ in range(count_):
                new_tokens.append(')')

    return new_tokens, mapping



def simplify_nopar(tokens, v2c):
    SENSE_PATTERN = re.compile('-[0-9][0-9]$')
    mapping = {}
    new_tokens = []
    for idx, tok in enumerate(tokens):
        # ignore instance-of
        if tok.startswith('('):
            #new_tokens.append('(')
            last_map = tok.replace("(", "")
            continue
        elif tok == '/':
            save_map = True
            continue
        # predicates, we remove any alignment information and parenthesis
        elif tok.startswith(':'):

            new_tok = tok.strip(')')
            new_tok = new_tok.split('~')[0]
            new_tokens.append(new_tok)

            count_ = tok.count(')')
            # for _ in range(count_):
            #     new_tokens.append(')')

        # concepts/reentrancies, treated similar as above
        else:
            new_tok = tok.strip(')')
            new_tok = new_tok.split('~')[0]

            if new_tok == "":
                continue

            # now we check if it is a concept or a variable (reentrancy)
            if new_tok in v2c:
                # reentrancy: replace with concept
                if new_tok not in mapping:
                    mapping[new_tok] = set()
                mapping[new_tok].add(len(new_tokens))

                if v2c[new_tok] is not None:
                    new_tok = v2c[new_tok]


            # check number
            elif new_tok.isnumeric():
                new_tok = new_tok
            # remove sense information
            elif re.search(SENSE_PATTERN, new_tok):
                new_tok = new_tok[:-3]
            # remove quotes
            elif new_tok[0] == '"' and new_tok[-1] == '"':
                new_tok = new_tok[1:-1]

            if new_tok != "":
                new_tokens.append(new_tok)

            if save_map:
                if last_map not in mapping:
                    mapping[last_map] = set()

                mapping[last_map].add(len(new_tokens) - 1)
                save_map = False

            count_ = tok.count(')')
            # for _ in range(count_):
            #     new_tokens.append(')')

    return new_tokens, mapping


def get_positions(new_tokens, src):
    pos = []
    for idx, n in enumerate(new_tokens):
        if n == src:
            pos.append(idx)

    return pos


def get_line_amr_graph(graph, new_tokens, mapping, roles_in_order, amr):
    triples = []
    nodes_to_print = new_tokens

    graph_triples = graph.triples

    edge_id = -1
    triples_set = set()
    count_roles = 0
    for triple in graph_triples:
        src, edge, tgt = triple

        # try:

        if edge == ':instance' or edge == ':instance-of':
            continue

        # print(triple)

        # if penman.layout.appears_inverted(graph_penman, v):
        if "-of" in roles_in_order[count_roles] and "-off" not in roles_in_order[count_roles]:
            if edge != ':consist-of':
                edge = edge + "-of"
                old_tgt = tgt
                tgt = src
                src = old_tgt

        try:
            assert roles_in_order[count_roles] == edge
        except:
            print(roles_in_order)
            print(count_roles)
            print(edge)

        count_roles += 1

        if edge == ':wiki':
            continue


        src = str(src).replace("\"", "")
        tgt = str(tgt).replace("\"", "")

        try:
            if src not in mapping:
                src_id = get_positions(new_tokens, src)
            else:
                src_id = sorted(list(mapping[src]))
            # check edge to verify
            edge_id = get_edge(new_tokens, edge, edge_id, triple, mapping, graph)

            if tgt not in mapping:
                tgt_id = get_positions(new_tokens, tgt)
            else:
                tgt_id = sorted(list(mapping[tgt]))
        except:
            print(graph_triples)
            print(src, edge, tgt)
            print("error")

            print(" ".join(new_tokens))


        for s_id in src_id:
            if (s_id, edge_id, 'd') not in triples_set:
                triples.append((s_id, edge_id, 'd'))
                triples_set.add((s_id, edge_id, 'd'))
                triples.append((edge_id, s_id, 'r'))
        for t_id in tgt_id:
            if (edge_id, t_id, 'd') not in triples_set:
                triples.append((edge_id, t_id, 'd'))
                triples_set.add((edge_id, t_id, 'd'))
                triples.append((t_id, edge_id, 'r'))

    if nodes_to_print == []:
        # single node graph, first triple is ":top", second triple is the node
        triples.append((0, 0, 's'))
    return nodes_to_print, triples


def get_edge(tokens, edge, edge_id, triple, mapping, graph):
    for idx in range(edge_id + 1, len(tokens)):
        if tokens[idx] == edge:
            return idx


def create_set_instances(graph_penman):
    instances = graph_penman.instances()
    # print(instances)
    dict_insts = {}
    for i in instances:
        dict_insts[i.source] = i.target
    return dict_insts


def get_roles_penman(graph_triples, roles_in_order):
    roles_penman = []
    count_roles = 0
    for v in graph_triples:
        role = v[1]
        if role == ':instance' or role == ':instance-of':
            continue
        if "-of" in roles_in_order[count_roles]:
            role = role + "-of"
        roles_penman.append(role)
        count_roles += 1

    return roles_penman


def check_triple(graph_triples, triple, amr_data):
    src, edge, tgt = triple
    try:
        if tgt in TYPES_AMR:
            #print("tgt", tgt)
            for triple2 in graph_triples:
                s, e, t = triple2
                if e == ':name' and s == src:
                    name_id = t
            name_entity = []
            for triple2 in graph_triples:
                s, e, t = triple2
                if s == name_id and e != ':instance':
                    t = t.replace("\"", "")
                    name_entity.append(t)
            return " ".join(name_entity)
    except:
        return None
    return None


def get_line_graph(graph, new_tokens, mapping, roles_in_order, amr_data, data_hal_amr, v2c_penman, doc_graph):

    hallucinated_nodes, hallucinated_words, map_nodes_tokens = data_hal_amr

    triples = set()
    words_amr = set()
    hal = False
    nodes_to_print = new_tokens

    graph_triples = graph.triples

    for triple in graph_triples:
        src, edge, tgt = triple
        if edge == ":instance" or edge == ":instance-of":
            name_entity = check_triple(graph_triples, triple, amr_data)
            if name_entity:
                v2c_penman[src] = name_entity

    edge_id = -1
    count_roles = 0
    for triple in graph_triples:
        src, edge, tgt = triple

        if edge == ':instance' or edge == ':instance-of':
            continue
        if edge == ':wiki':
            continue

        if "-of" in roles_in_order[count_roles] and "-off" not in roles_in_order[count_roles]:
            if edge != ':consist-of':
                edge = edge + "-of"
                old_tgt = tgt
                tgt = src
                src = old_tgt

        try:
            assert roles_in_order[count_roles] == edge
        except:
            print(roles_in_order)
            print(count_roles)
            print(edge)
        count_roles += 1

        src = str(src).replace("\"", "")
        tgt = str(tgt).replace("\"", "")

        try:
            if src not in mapping:
                src_id = get_positions(new_tokens, src)
            else:
                src_id = sorted(list(mapping[src]))
            # check edge to verify
            edge_id = get_edge(new_tokens, edge, edge_id, triple, mapping, graph)

            if tgt not in mapping:
                tgt_id = get_positions(new_tokens, tgt)
            else:
                tgt_id = sorted(list(mapping[tgt]))
        except:
            print(graph_triples)
            print(src, edge, tgt)
            print("error")


        if src not in mapping:
            src_print = src
        else:
            src_print = v2c_penman[src]

        if tgt not in mapping:
            tgt_print = tgt
        else:
            tgt_print = v2c_penman[tgt]

        if src_print in hallucinated_nodes or tgt_print in hallucinated_nodes:
            haluc = 0
        else:
            haluc = 1

        for s_id in src_id:
            for t_id in tgt_id:
                triples.add((s_id, t_id, edge_id, haluc))

        if doc_graph:
            continue

        hal_word = None
        idx_words1 = set()
        for s_print in src_print.split():
            for idx_word1 in map_nodes_tokens[s_print]:

                if idx_word1 in hallucinated_words:
                    hal_word = 0
                    hal = True

                idx_words1.add(idx_word1)

        idx_words2 = set()
        for t_print in tgt_print.split():
            for idx_word2 in map_nodes_tokens[t_print]:

                if idx_word2 in hallucinated_words:
                    hal_word = 0
                    hal = True

                idx_words2.add(idx_word2)

        if hal_word == None:
            hal_word = 1

        if idx_words1 and idx_words2:
            idx_words1 = list(idx_words1)
            idx_words2 = list(idx_words2)

            idxs_words1 = " ".join([str(idx_word) for idx_word in idx_words1])
            idxs_words2 = " ".join([str(idx_word) for idx_word in idx_words2])

            words_amr.add((idxs_words1, idxs_words2,
                    " ".join([amr_data.tokens[idx_word] for idx_word in idx_words1]),
                    " ".join([amr_data.tokens[idx_word] for idx_word in idx_words2]),
                           hal_word, edge,
                    " ".join([node for node in src_print.split()]),
                    " ".join([node for node in tgt_print.split()]),
                    " ".join([str(node) for node in src_id]),
                    " ".join([str(node) for node in tgt_id])
                           ))
        elif idx_words1 and src_id and hal_word == 0:
            idx_words1 = list(idx_words1)
            idx_words2 = list(idx_words1)

            idxs_words1 = " ".join([str(idx_word) for idx_word in idx_words1])
            idxs_words2 = " ".join([str(idx_word) for idx_word in idx_words2])

            words_amr.add((idxs_words1, idxs_words2,
                    " ".join([amr_data.tokens[idx_word] for idx_word in idx_words1]),
                    " ".join([amr_data.tokens[idx_word] for idx_word in idx_words2]),
                           hal_word, edge,
                    " ".join([node for node in src_print.split()]),
                    " ".join([node for node in src_print.split()]),
                    " ".join([str(node) for node in src_id]),
                    " ".join([str(node) for node in src_id])
                           ))
        elif idx_words2 and tgt_id and hal_word == 0:
            idx_words1 = list(idx_words2)
            idx_words2 = list(idx_words2)

            idxs_words1 = " ".join([str(idx_word) for idx_word in idx_words1])
            idxs_words2 = " ".join([str(idx_word) for idx_word in idx_words2])

            words_amr.add((idxs_words1, idxs_words2,
                    " ".join([amr_data.tokens[idx_word] for idx_word in idx_words1]),
                    " ".join([amr_data.tokens[idx_word] for idx_word in idx_words2]),
                           hal_word, edge,
                    " ".join([node for node in tgt_print.split()]),
                    " ".join([node for node in tgt_print.split()]),
                    " ".join([str(node) for node in tgt_id]),
                    " ".join([str(node) for node in tgt_id])
                           ))

        #print(idx_words1, idx_words2, src_print, tgt_print, haluc, hal_word_log)

    if nodes_to_print == []:
        triples.append((0, 0, 's', haluc))
    return nodes_to_print, list(triples), hal, list(words_amr)


def simplify_amr_hal(amr_data, data_hal_amr, doc_graph=False):
    try:
        amr = amr_data.graph_string()
        graph_penman = penman.decode(amr)
        v2c_penman = create_set_instances(graph_penman)

        amr_penman = penman.encode(graph_penman)
        amr_penman = amr_penman.replace('\t', '')
        amr_penman = amr_penman.replace('\n', '')
        tokens = amr_penman.split()
    except:
        raise Exception("Error while converting AMR graph.")

    try:
        new_tokens, mapping = simplify_nopar(tokens, v2c_penman)
    except Exception as e:
        raise Exception("Error while simplyfying AMR graph.")

    roles_in_order = []
    for token in amr_penman.split():
        if token.startswith(":"):
            if token == ':instance-of':
                continue
            roles_in_order.append(token)

    nodes, triples, hal, words_amr = get_line_graph(graph_penman, new_tokens, mapping, roles_in_order, amr_data, data_hal_amr, v2c_penman, doc_graph)
    try:
        triples = sorted(triples)

        return nodes, triples, hal, words_amr
    except:
        raise Exception("Error while processing AMR graph.")


def simplify_amr_nopar(amr):
    try:
        graph_penman = penman.decode(amr)
        v2c_penman = create_set_instances(graph_penman)

        amr_penman = penman.encode(graph_penman)
        amr_penman = amr_penman.replace('\t', '')
        amr_penman = amr_penman.replace('\n', '')
        tokens = amr_penman.split()
    except:
        print('error')
        exit()
        return None

    try:
        new_tokens, mapping = simplify_nopar(tokens, v2c_penman)
    except Exception as e:
        print(e.message, e.args)
        print('error simply')
        #exit()
        return None

    roles_in_order = []
    for token in amr_penman.split():
        if token.startswith(":"):
            if token == ':instance-of':
                continue
            roles_in_order.append(token)

    nodes, triples = get_line_amr_graph(graph_penman, new_tokens, mapping, roles_in_order, amr)

    triples = sorted(triples)

    return nodes, triples



def simplify_amr_triples(amr):
    graph_penman = penman.decode(amr)
    v2c_penman = create_set_instances(graph_penman)

    graph = ''
    for t in graph_penman.triples:
        if t[1] != ':instance':
            try:
                head = t[0]
                tail = t[2]
                if t[0] in v2c_penman.keys():
                    head = v2c_penman[t[0]]
                if t[2] in v2c_penman.keys():
                    tail = v2c_penman[t[2]]

                graph += ' <H> ' + head + ' <R> ' + t[1] + ' <T> ' + tail
            except Exception as e:
                print(e)
                print(graph_penman.triples)
                print(graph_penman.instances())


    return graph


def read_csv(file):
    # Read CSV file
    with open(file) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        # next(reader, None)  # skip the headers
        data_read = [row for row in reader]

        return data_read


def map_nodes(graph, align_data):

    amr = graph
    nodes = amr.nodes
    sent_tok = amr.tokens

    map_nodes_tokens = defaultdict(list)
    for a in align_data[amr.id]:
        toks = a.tokens
        nds = a.nodes

        toks_alin = []
        for tok in toks:
            selected_tok = sent_tok[tok]
            toks_alin.append(selected_tok)

        nds_alin = []
        for n in nds:
            nds_alin.append(nodes[n])

            n = nodes[n].replace("\"", "")
            if toks:
                map_nodes_tokens[n].extend(toks)


    return map_nodes_tokens


def check_hallucinations(datapoint, graph, align_data, hal_data):

    amr = graph
    nodes = amr.nodes
    sent_tok = amr.tokens

    hal_spans = hal_data[datapoint['id']]

    hals = json.loads(datapoint['hallucinations'])
    incorrect_hals = []
    for h in hals:
        # incorect hal
        if h[2] == 0:
            incorrect_hals.extend(h[1].split())

    filtered_hals = []
    for h in incorrect_hals:
        for span in hal_spans:
            if h in span:
                filtered_hals.append(h)

    hals = list(set(filtered_hals))
    for idx, h in enumerate(hals):
        sent_nlp = nlp(h)
        tokens = [token.text for token in sent_nlp]
        tokens = ' '.join(tokens)
        hals[idx] = tokens.lower()

    hallucinated_nodes = set()
    hals_id = set()
    map_nodes_tokens = defaultdict(list)
    for a in align_data[amr.id]:
        toks = a.tokens
        nds = a.nodes

        hallucination = False

        toks_alin = []
        for tok in toks:
            selected_tok = sent_tok[tok]
            toks_alin.append(selected_tok)

            for h in hals:
                if selected_tok in h:
                    hallucination = True
                    hals_id.add(tok)
                    break

        nds_alin = []
        for n in nds:
            nds_alin.append(nodes[n])

            n = nodes[n].replace("\"", "")
            if toks:
                map_nodes_tokens[n].extend(toks)

        if hallucination:
            for n in nds:
                n = nodes[n].replace("\"", "")
                hallucinated_nodes.add(n)

    hallucinated_nodes = list(hallucinated_nodes)
    hals_id = list(hals_id)

    return hallucinated_nodes, hals_id, map_nodes_tokens, hals