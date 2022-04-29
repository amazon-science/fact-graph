import unidecode
from utils import *
import sys

file = sys.argv[1]
cnndm = load_source_docs(file)

processed_data = []
labels_cont = []
id_order = 0

for idx, example in enumerate(cnndm):

    for s in example['summary_sentences']:
        new_example = {}
        new_example['summary'] = unidecode.unidecode(s['sentence'])
        new_example['article'] = unidecode.unidecode(example['article'])
        new_example['domain'] = 'cnndm'
        new_example['source'] = 'qags'
        new_example['id'] = 'cnndm_qags_' + str(id_order)
        new_example['id_order'] = id_order
        id_order += 1

        reps = [r['response'] for r in s['responses']]
        label = most_common(reps)
        if label == 'yes':
            new_example['label'] = 'CORRECT'
        elif label == 'no':
            new_example['label'] = 'INCORRECT'
        else:
            print('error')
            exit()
        processed_data.append(new_example)
        labels_cont.append(new_example['label'])


file = sys.argv[2]
xsum = load_source_docs(file)


labels_cont = []
for idx, example in enumerate(xsum):

    for s in example['summary_sentences']:
        new_example = {}
        new_example['summary'] = unidecode.unidecode(s['sentence'])
        new_example['article'] = unidecode.unidecode(example['article'])
        new_example['domain'] = 'xsum'
        new_example['source'] = 'qags'
        new_example['id'] = 'xsum_qags_' + str(id_order)
        new_example['id_order'] = id_order
        id_order += 1

        reps = [r['response'] for r in s['responses']]
        label = most_common(reps)
        if label == 'yes':
            new_example['label'] = 'CORRECT'
        elif label == 'no':
            new_example['label'] = 'INCORRECT'
        else:
            print('error')
            exit()
        processed_data.append(new_example)
        labels_cont.append(new_example['label'])


output_file = sys.argv[3]
save_data(processed_data, output_file)
