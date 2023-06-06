import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from transformers import AutoTokenizer, AutoModelForMaskedLM


def dna_bert6():
    model_name = "zhihan1996/DNA_bert_6"
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return model
