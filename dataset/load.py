"""
Load raw data (.fasta) into sequences and labels (.in)
"""

from itertools import islice
from Bio import SeqIO

# paths to fetch the raw file
DATA_PATH = './data'
FILE_PATH = '/chip-seq-toy/test.fasta'


def read_fasta(file, limit=None):
    """
    Read .fasta file into sequences and labels arrays,
    assuming the names of records are labels.
    """
    with open(file, 'r') as f:
        records = [
            (record.seq._data.upper(), int(record.name))
            for record in islice(SeqIO.parse(f, 'fasta'), limit)
        ]

    l = map(list, zip(*records))
    sequences, labels = next(l), next(l)
    print(len(sequences), 'samples loaded')

    return sequences, labels


if __name__ == '__main__':
    sequences, labels = read_fasta(DATA_PATH+FILE_PATH)
    # for i in range(5):
    #     print(sequences[i].decode(), type(sequences[i]), len(sequences[i].decode()))
    #     print(labels[i], type(labels[i]))
    f1 = open("./sequence-toy_test.in", "w")
    for seq in sequences:
        f1.write(seq.decode() + "\n")
    f1.close()

    f2 = open("./label-toy_test.in", "w")
    for label in labels:
        f2.write(str(label) + "\n")
    f2.close()
