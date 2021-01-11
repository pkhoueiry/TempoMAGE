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
A dataset is available in the repository: `training_testing_and_saved_model.tar.xz`.
Just extract its content and you will find the following:
- training_data: this dataset that can be used to train the model. this is only used to demonstrate how the model performs from epoch to another.
- test_data: this dataset is used to test the already saved model (included in the tar ball).
- tempoMAGE_savedmodel: the saved model the can be used to predict or test the model.

## Train TempoMAGE on training data
Use the following command to train the model on th training dataset:
`python3 /data/project/ml_mah19/train_tempoMAGE.py -t /data/project/ml_mah19/training_data/ -o /data/Downloads/new_test/ -e 1 -s yes`

## Test TempoMAGE on test data
`python3 /data/project/ml_mah19/test_tempoMAGE.py -t /data/project/ml_mah19/testing_data/ -o /data/Downloads/new_test/out_test/ -m /data/Downloads/new_test/`