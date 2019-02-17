Code for IDAO muon detection first stage competition
- Functions to train model for two tracks are
    - Track 1: train_model_track1()
    - Track 2: train_model_track2()

Above two functions needs data as prepared by prepare_data()

Evaluator can either run the idao_data_broom.py after specifying the path to data as data_path
Or, they can run individual functions, once train, test_private data are loaded with closest hit features. For, this to
work prepare_data() should be called first to prepare the data in right format.

feature: list: utils.SIMPLE_FEATURE_COLUMNS + closest_features
folder `best_model` contains saved model for both the tracks

