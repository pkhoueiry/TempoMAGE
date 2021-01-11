# TempoMAGE
TEMPOral data Prediction of histone Modifications in Accessible GEnome

## Dependencies
To train or test TempoMAGE, you need to install the following dependencies:
- tensorflow 2.1.0
- matplotlib
- sklearn
- pandas
- numpy
- Bio
- cython
- pyranges
- mlxtend

## Dataset
A dataset is available in the repository, in the `dataset` directory and it contains the following:
- `training_data.tar.xz`: this is the training_set that can be used to train the model. this is only used to demonstrate how the model performs from epoch to another.
- `testing_data.tar.xz`: this is the test set used to test the already saved model.
- `tempoMAGE_savedmodel.tar.xz`: the saved model the can be used to predict or test the model.

## Usage
`python3 train_tempoMAGE.py`
```
train_tempoMAGE.py 
     -t/--training_data <full path for the training dataset FILES>
     -o/--output_dir <output directory for plots>
     -e/--epochs <EPOCHS>
     -s/--save <whether or not to save the trained model>
```

`python3 test_tempoMAGE.py`
```
test_tempoMAGE.py 
     -t/--testing_data <full path for the testing dataset FILES>
     -o/--output_dir <output directory for plots>
     -m/--model <full path for the model already trained>
```

## Train TempoMAGE on training data
To train the model on the training set, extract `training_data.tar.xz` and use the following command:

```
python3 train_tempoMAGE.py \
    -t training_data/ \
    -o output_dir/ \
    -e 1 \
    -s yes
```

## Test TempoMAGE on test data
To test the already saved model, extract `testing_data.tar.xz` and `tempoMAGE_savedmodel.tar.xz` and use the following command:

```
python3 test_tempoMAGE.py \
    -t testing_data/ \
    -o out_test/ \
    -m tempoMAGE_savedmodel/
```