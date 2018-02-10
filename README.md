# Neural-based models in NLG

Experiments using neural network RNN(char and word level) and GAN(char-level) in Natural Language Generation and their performance

### raw_corpus/

The Big Bang Theory S1-S10: https://github.com/skashyap7/TBBTCorpus/tree/master/preprocessing

Crawl from here: https://bigbangtrans.wordpress.com/

### data/

Training data used for the experiment. Extracted all Sheldon transcripts from raw\_corpus

### doc/

All related report and presentation

### preprocess/

Codes for crawling data and preprocessing

### rnn_based_solution/

Codes for char and word level generation using RNN

### GAN_result/

Results of sample output and training logs after training GAN in char level with sequence length of 80, codes are in the reference

## Reference

1. RNN model:

- https://github.com/martin-gorner/tensorflow-rnn-shakespeare

- https://github.com/jamesrequa/TV-Script-RNN

- https://github.com/shawnwun/RNNLG

2. GAN:

- https://github.com/amirbar/rnn.wgan

3. Crawler:

- https://github.com/skashyap7/TBBTCorpus/tree/master/preprocessing
