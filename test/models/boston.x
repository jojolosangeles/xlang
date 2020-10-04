features => boston_test_1 => float
- dense 64 relu
- dense

load boston_housing

train rmsprop mae
- 10 epochs, show loss mae
- batch 512
