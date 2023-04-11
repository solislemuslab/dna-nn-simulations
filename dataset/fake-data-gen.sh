# pick from {A,T,C,G}
var="ATCG"
# define the fake sequence and label input filename
seq_filename="sequence-fake01.in"
lab_filename="label-fake01.in"
# define the sample size and sequence length
sample_size=1000
seq_length=500

# We generate {$sample_size} fake samples
for (( i=1; i<=$sample_size; ++i ))
do
	# For each sample, the length of sequence is {$seq_length}
	seq=""
	for (( j=1; j<=$seq_length; ++j ))
	do
		# Pick a random character from {ATCG} and append to the sequence
		new_char="${var:$(( RANDOM % ${#var} )):1}"
		seq="${seq}${new_char}"
	done

	# Append the sequence to the sequence file
	echo $seq >> $seq_filename
	# randomly choose a number from {0,1} and append to the label file
	echo $(($RANDOM%2)) >> $lab_filename
done

