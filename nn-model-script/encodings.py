# encoding sequences as kmers
def seq2kmer(seq, k=3) -> str:
    """
    converts original sequences to kmers
    
    seq: str, original sequence
    k: int, the length of each kmer

    returns str: kmers separated by spaces
    """
    kmers = [seq[x: x + k] for x in range(len(seq) - k)]

    return " ".join(kmers)
