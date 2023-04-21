import msprime
import tskit
# from IPython.display import SVG

# TODO: determine the parameters
# parameters for simulation: (TBD)
# the length of a single sequence
seq_length = 100
# the sample size
sample_size = 1000

# other parameters: (not sure if needed or how to determine)
# recombination_rate=1e-8
# population_size=1e4
# model, demography, ploidy, discrete_genome, etc.

# step 1: the initial tree sequence object
ts = msprime.sim_ancestry(sample_size/2, sequence_length=seq_length)

# parameters for mutation:
mutation_rate = 0.1
# TODO: still different mutations every time, not sure why
mutation_seed = 456
# model, discrete_genome

# step 2: mutations
mts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=mutation_seed)

# step 3: random reference sequence alignment
refseq = tskit.random_nucleotides(mts.sequence_length, seed=123)

# step 4: output the sequence
f = open("../dataset/sequence-simple.in", "w")
for _ in mts.alignments(reference_sequence=refseq):
    # print(_)
    f.write(_)
f.close()
