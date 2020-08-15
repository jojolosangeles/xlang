28*28 => mnist_p53 => 10 probabilities
- dense 512 relu


784 => mnist_p63 => 10 probabilities
- dense 32 relu


text 10k => imdb_p72 => probability
- dense 16 relu
- dense


text 10k => reuters_p80 => 46 probabilities
- dense 64 relu
- dense


text 10k => reuters_p84 => 46 probabilities
- dense 64 relu
- dense 4


features => boston_p86 => float
- dense 64 relu
- dense


text 10k => imdb_p105 => probability
- dense 16 relu
- dense


text 10k => imdb_p105b => probability
- dense 4 relu
- dense


text 10k => imdb_p106 => probability
- dense 512 relu
- dense


text 10k => imdb_p108 => probability
- dense 16 relu l2 0.001
- dense


text 10k => imdb_p110 => probability
- dense 16 relu
- dropout 0.5
- dense
- dropout


28x28x1 => mnist_p120 => 10 probabilities
- conv 32 3x3 relu
- maxpool 2x2
- conv 64
- maxpool
- conv
- flatten
- dense 64 relu


150x150x3 => catdog_p134 => probability
- conv 32 3x3 relu
- maxpool 2x2
- conv 64
- maxpool
- conv 128
- maxpool
- conv 128
- maxpool
- flatten
- dense 512 relu


150x150x3 => catdog_p141 => probability
- conv 32 3x3 relu
- maxpool 2x2
- conv 64
- maxpool
- conv 128
- maxpool
- conv 128
- maxpool
- flatten
- dropout 0.5
- dense 512 relu


4*4*512 => catdog_p148 => probability
- dense 256 relu
- dropout 0.5


150x150x3 => catdog_p150 => probability
- convbase VGG16 imagenet
- flatten
- dense 256 relu


text 10k => imdb_p187 => probability
- embed 8 input_length=20
- flatten


text 10k => imdb_p191 => probability
- embed 100 input_length=100
- flatten
- dense 32 relu


text 10k => imdb_p200 => probability
- embed 32
- simpleRNN 32


text 10k => imdb_p205 => probability
- embed 32
- LSTM 32


timed 32x7 => jena_p213 => float
- flatten
- dense 32 relu


time 7 => jena_p215 => float
- GRU 32


time 7 => jena_p217 => float
- GRU 32 dropout=0.2 recurrent_dropout=0.2


time 7 => jena_p218 => float
- GRU 32 dropout=0.2 recurrent_dropout=0.2
- GRU 64


text 10k => imdb_p220 => probability
- embed 128
- LSTM 32


text 10k => imdb_p221 => probability
- embed 32
- bidirectional LSTM 32


time 7 => jena_p227 => float
- embed 128 input_length=500
- conv 32 7 relu
- maxpool 5
- conv
- global maxpool


time 7 => jena_p228 => float
- conv 32 5 relu
- maxpool 3
- conv
- maxpool
- conv
- global maxpool


time 7 => jena_p230 => float
- conv 32 5 relu
- maxpool 3
- conv
- GRU 32 dropout=0.1 recurrent_dropout=0.5


64x64x3 => image_p262 => 10 probabilities
with conv=separableconv
- conv 32 3x3 relu
- conv 64
- maxpool 2x2
- conv 64
- conv 128
- maxpool 2x2
- conv 64
- conv 128
- global_average_pool
- dense 32 relu


60x26 => nietz_p275 => 26 probabilities
- LSTM 128
