from queueing_rnn import RNNCell, RNNCellTraining

""" This script executes our RNN-learning methodology. We have to detail
a directory that contains the learning traces such that we can read from them
and an output file location such that we can save our learned parameters. To re-create
our experiments, change the matdir and the output_file to the directories that
you have saved the traces in and that you want to save the .txt file with the learned
parameters in respectively.

This code has been inspired and adapted from the following source:
Garbi, G et al. (2020). Learning Queueing Networks by Recurrent Neural Networks.
Accessible at:
https://pdfs.semanticscholar.org/7f7c/12bcc23ba098ad5a4a0ad251bd92e9b9c27a.pdf."""

matdir = '/Users/alfredtingey/RNN_queueing/learning_traces/synthetic/net_bottle_generated'
output_file = '/Users/alfredtingey/RNN_queueing/models_learnt/model_bottle_generated.txt'

# Initialize the RNN cell and training cell using the traces in the directories.
td = RNNCellTraining(matdir, lambda init_s: RNNCell(init_s))

# Load the trace files
td.load_file()
# Build the RNN network
td.makeNN(lr=0.05)
# Start the learning
td.learn()
# Save the results to the output file
td.saveResults(output_file)
