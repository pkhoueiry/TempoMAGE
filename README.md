# TempoMAGE
TEMPOral data Prediction of histone Modifications in Accessible GEnome

## Dependencies
To train or test TempoMAGE, you need to have install the following dependencies:
- tensorflow 2.1.0
- matplotlib
- skleanr
- pandas
- numpy
- Bio
- cython
- pyranges
- mlxtend

## Dataset
A dataset is available in the repository, in the `dataset` directory and it contains the following:
- training_data.tar.xz: this is the training_set that can be used to train the model. this is only used to demonstrate how the model performs from epoch to another.
- test_data.tar.xz: this is the test set used to test the already saved model.
- tempoMAGE_savedmodel.tar.xz: the saved model the can be used to predict or test the model.

## Train TempoMAGE on training data
Use the following command to train the model on th training dataset:
`python3 train_tempoMAGE.py -t training_data/ -o output_dir/ -e 1 -s yes`

## Test TempoMAGE on test data
`python3 test_tempoMAGE.py -t testing_data/ -o out_test/ -m tempoMAGE_savedmodel/`