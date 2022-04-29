import json
import os
from tqdm import tqdm
import augmentation_ops as ops
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import set_start_method
import sys

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def apply_transformation_parallel(data, operation, transformation, num_sents, size_chunks, workers):

    data_list = list(chunks(data, size_chunks))
    set_start_method('spawn', force=True)
    final_datapoints = []
    with tqdm(total=len(data_list)) as pbar:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, data in enumerate(data_list):
                job = executor.submit(transformation, data, operation, num_sents, idx)
                futures[job] = idx

            for job in as_completed(futures):
                datapoint = job.result()
                r = futures[job]
                pbar.update(1)
                final_datapoints.extend(datapoint)
                del futures[job]
    return final_datapoints


def load_source_docs(file_path, to_dict=False):
    with open(file_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    if to_dict:
        data = {example["id"]: example for example in data}
    return data


def save_data(data, file, name_suffix):
    output_file = os.path.splitext(file)[0] + "-" + name_suffix + ".json"

    with open(output_file, "w", encoding="utf-8") as fd:
        for example in data:
            example = dict(example)
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")


def apply_transformation(data, operation, num_sents, idx_process):
    for idx, example in enumerate(data):
        try:
            new_example = operation.transform(example, number_sents=num_sents)
            if new_example:
                data[idx] = new_example
        except Exception as e:
            print("Caught exception:", e)
    return data


def apply_transformation_list(data, operation, idx):
    new_data = []
    for example in data:
        try:
            new_examples = operation.transform(example)
            if new_examples:
                for new_example in new_examples:
                    if new_example:
                        new_data.append(new_example)
        except Exception as e:
            print("Caught exception:", e)
    return new_data


def main(file, num_sents):

    data = load_source_docs(file, to_dict=False)
    print("Loaded %d source documents." % len(data))

    sent_op = ops.SelectSentencesScore()
    data = apply_transformation_parallel(data, sent_op, apply_transformation, num_sents, 1000, 5)

    print(len(data))
    save_data(data, file, "sents")


if __name__ == "__main__":
    file = sys.argv[1]
    num_sents = int(sys.argv[2])
    main(file, num_sents)
