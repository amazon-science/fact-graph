import random

import spacy
import signal


from sentence_transformers import SentenceTransformer, util
from LexRank import degree_centrality_scores
import numpy as np
import collections
import logging
import json
import unidecode
logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)


class TimeoutException(Exception):  # Custom exception class
    pass


def timeout_handler(signum, frame):  # Custom signal handler
    raise TimeoutException


# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

LABEL_MAP = {True: "CORRECT", False: "INCORRECT"}




def make_new_example(eid=None, text=None, positive_example=None, sentences=None):
    # Embed example information in a json object.
    return {
        "id": eid,
        "text": text,
        "positive_example": positive_example,
        "sentences": sentences,
    }


class Transformation():
    # Base class for all data transformations

    def __init__(self):
        # Spacy toolkit used for all NLP-related substeps
        self.spacy = spacy.load("en_core_web_sm")

    def transform(self, example):
        # Function applies transformation on passed example
        pass



class SelectSentencesScore(Transformation):
    # Embed document as Spacy object and sample one sentence as claim
    def __init__(self, min_sent_len=8, model_type='paraphrase-mpnet-base-v2'):
        super().__init__()
        self.min_sent_len = min_sent_len
        self.count_at_random = 0
        self.model_sentence_transf = SentenceTransformer(model_type)

    def transform(self, example, number_sents):
        assert example["article"] is not None, "Text must be available"

        self.model_sentence_transf.encode(['Test'], convert_to_tensor=True)

        example["article"] = unidecode.unidecode(example["article"])
        example["summary"] = unidecode.unidecode(example["summary"])

        page_text = example["article"].replace("\n", "")
        summaries = example["summary"].replace("\n", "")

        page_doc = self.spacy(page_text, disable=["tagger"])
        #sents = [sent for sent in page_doc.sents if len(sent) >= self.min_sent_len]
        sents = [sent for sent in page_doc.sents]

        page_sum = self.spacy(summaries, disable=["tagger"])
        sents_sum = [sent for sent in page_sum.sents]

        sents_text = [sent.text for sent in sents]
        sents_text_sum = [sent.text for sent in sents_sum]

        assert len(sents) == len(sents_text)

        # import pdb
        # pdb.set_trace()

        signal.alarm(15)

        try:
            embs = self.model_sentence_transf.encode(sents_text, convert_to_tensor=True)
            embs_sum = self.model_sentence_transf.encode(sents_text_sum, convert_to_tensor=True)

            # Compute the pair-wise cosine similarities
            cos_scores = util.pytorch_cos_sim(embs, embs_sum).cpu().detach().numpy()

            centrality_scores = np.mean(cos_scores, axis=1)

            # We argsort so that the first element is the sentence with the highest score
            most_central_sentence_indices = np.argsort(-centrality_scores)

            #assert len(most_central_sentence_indices) == len(sents)

            best_sents = []
            for idx in most_central_sentence_indices[:number_sents]:
                tmp_claim = sents[idx]
                claim_text = tmp_claim.text
                best_sents.append((claim_text, int(idx), float(centrality_scores[idx])))


        except TimeoutException:
            self.count_at_random += 1
            print("return none error", self.count_at_random)
            return None
        else:
            # Reset the alarm
            signal.alarm(0)
            if not best_sents:
                self.count_at_random += 1
                print("return none, no sents", self.count_at_random)
                return None

        example['sentences'] = json.dumps(best_sents)
        return example