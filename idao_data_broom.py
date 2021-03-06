import os
import utils
import pandas as pd
import numpy as np
import swifter
import scoring
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool


def read_train_test_data(data_path):
    print("Reading train and test data")
    train, test = utils.load_full_test_csv(data_path)
    print("Creating closest features for train data")
    train_closest_hits_features = train.swifter.apply(utils.find_closest_hit_per_station,
                                                      result_type="expand", axis=1)
    print("Creating closest features for test data")
    test_closest_hits_features = test.swifter.apply(utils.find_closest_hit_per_station,
                                                    result_type="expand", axis=1)
    # changing column name to more readable format
    new_col_names = {i: f'closest_feature_{i}' for i in range(24)}
    train_closest_hits_features.rename(columns=new_col_names, inplace=True)
    test_closest_hits_features.rename(columns=new_col_names, inplace=True)

    features = utils.SIMPLE_FEATURE_COLUMNS
    # extending features list to add closest features
    features.extend(list(new_col_names.values()))
    print("Combining simple features and closest features")
    train_concat = pd.concat(
        [train.loc[:, utils.SIMPLE_FEATURE_COLUMNS + utils.TRAIN_COLUMNS], train_closest_hits_features], axis=1)
    test_concat = pd.concat([test.loc[:, utils.SIMPLE_FEATURE_COLUMNS], test_closest_hits_features], axis=1)

    return train_concat, test_concat, features


def read_private_test_data(data_path):
    print("Reading private test data")
    test_private = utils.load_full_private_test_csv(data_path)
    print("Creating closest features for private test data")
    test_closest_hits_features = test_private.swifter.apply(utils.find_closest_hit_per_station,
                                                            result_type="expand", axis=1)

    # changing column name to more readable format
    new_col_names = {i: f'closest_feature_{i}' for i in range(24)}
    test_closest_hits_features.rename(columns=new_col_names, inplace=True)
    test_concat = pd.concat([test_private.loc[:, utils.SIMPLE_FEATURE_COLUMNS], test_closest_hits_features], axis=1)

    return test_concat


def prepare_data(train, features):
    """
    Function to prepare data to train catboost model
    :param train: pd.DataFrame: train data with closest hit features
    :param features: list: utils.SIMPLE_FEATURE_COLUMNS + closest_features
    :return:
           train_part: pd.DataFrame: train data split into validation and train
           validation: pd.DataFrame: part of train data, to be used for validation
           train_pool:  catboost.Pool(), to be used to train the model. This object is created using train_part
           validation_pool: catboost.Pool(), to evaluate model performance while training. This is created using validation
    """
    train_part, validation = train_test_split(train, test_size=0.2, random_state=100)
    train_weight_min = train_part.loc[train_part['weight'] > 0, 'weight'].min()
    train_part_mod = train_part.copy()
    validation_mod = validation.copy()
    # Replacing negative weight with minimum positive weight from train data.
    # Above is done because catboost does not support negative weight
    train_part_mod['weight'] = np.where(train_part_mod['weight'] > 0, train_part_mod['weight'], train_weight_min)
    validation_mod['weight'] = np.where(validation_mod['weight'] > 0, validation_mod['weight'], train_weight_min)
    # Creating Pool of train and validation, this will be used during model training to assess the performance
    train_pool = Pool(data=train_part_mod.loc[:, features].values, label=train_part_mod.label.values,
                      weight=train_part_mod.weight.values)
    validation_pool = Pool(data=validation_mod.loc[:, features].values, label=validation_mod.label.values,
                           weight=validation_mod.weight.values)
    return train_part, validation, train_pool, validation_pool


def train_model_track2(train_pool, validation_pool, validation, features, data_path):
    """
    Function to train CatBoostClassifier, with the best set of parameters for track 2, as determined during first stage
    :param train_pool: catboost.Pool() object, as created by prepare_data()
    :param validation_pool: catboost.Pool() object, as created by prepare_data()
    :param validation: pd.DataFrame: as created by prepare_data(). Needed to get the score at rejection90
    :param features: list: utils.SIMPLE_FEATURE_COLUMNS + closest_features
    :param data_path: os.path() to which model will be saved
    :return: None
    """
    cat = CatBoostClassifier(iterations=1200,
                             loss_function='Logloss',
                             l2_leaf_reg=2,
                             random_seed=100,
                             scale_pos_weight=11.92984045,
                             eval_metric='AUC',
                             use_best_model=True,
                             early_stopping_rounds=100,
                             max_depth=7,
                             max_bin=100)

    cat.fit(train_pool, eval_set=validation_pool)

    valid_pred_prob = cat.predict_proba(validation.loc[:, features].values)[:, 1]
    valid_score_90 = scoring.rejection90(validation.label.values, valid_pred_prob,
                                         sample_weight=validation.weight.values)
    print(f"Score at rejection 90 for validation {valid_score_90}")
    model_file = os.path.join(data_path, 'track_2_best_model.cbm')
    print(f"Track 2 best model is being saved at {model_file}")
    cat.save_model(model_file, format='cbm')


def train_model_track1(train_pool, validation_pool, validation, test_private, features, data_path):
    """
        Function to train CatBoostClassifier, with the best set of parameters for track 1, as determined during first stage
        :param train_pool: catboost.Pool() object, as created by prepare_data()
        :param validation_pool: catboost.Pool() object, as created by prepare_data()
        :param validation: pd.DataFrame: as created by prepare_data(). Needed to get the score at rejection90
        :param test_private: pd.DataFrame: private test data, as created by read_private_test_data()
        :param features: list: utils.SIMPLE_FEATURE_COLUMNS + closest_features
        :param data_path: os.path() to which model will be saved
        :return: None
        """

    cat = CatBoostClassifier(iterations=3000,
                             loss_function='Logloss',
                             l2_leaf_reg=2,
                             random_seed=100,
                             scale_pos_weight=11.92984045,
                             eval_metric='AUC',
                             use_best_model=True,
                             early_stopping_rounds=100,
                             max_depth=7,
                             max_bin=100
                             )

    cat.fit(train_pool, eval_set=validation_pool)
    valid_pred_prob = cat.predict_proba(validation.loc[:, features].values)[:, 1]
    valid_score_90 = scoring.rejection90(validation.label.values, valid_pred_prob,
                                         sample_weight=validation.weight.values)
    # 0.771923225
    print(f"Score at rejection 90 {valid_score_90}")
    predictions = cat.predict_proba(test_private.loc[:, features].values)[:, 1]
    prediction_file = os.path.join(data_path, "test_private.csv")
    print(f"Track 1 prediction on private test data is present at {prediction_file}")
    pd.DataFrame(data={"prediction": predictions}, index=test_private.index).to_csv(prediction_file,
                                                                                    index_label=utils.ID_COLUMN)
    model_file = os.path.join(data_path, 'track_1_best_mode.cbm')
    print(f"Track 1 best model is saved at {model_file}")
    cat.save_model(model_file, format='cbm')


def main(data_path):
    """
    Main function which calls read, prepare and model train functions
    :param data_path: os.path() - location where all data are present
    :return:
    """
    # -------------- Start of Block 1 ----------------
    # If all steps - data reading, processing and model training has to be tested than use below block
    print("Reading Data")
    train, test_public, features = read_train_test_data(data_path)
    print("Reading private test data")
    test_private = read_private_test_data(data_path)
    train_part, validation, train_pool, validation_pool = prepare_data(train, features)
    print("Training and predicting for track 1")
    train_model_track1(train_pool, validation_pool, validation, test_private, features, data_path)
    print("Track 1 completed")
    print("Training and saving model for track 2")
    train_model_track2(train_pool, validation_pool, validation, features, data_path)
    print("Completed successfully")
    # -------------- End of Block 1 ----------------

    # -------------- Start of Block 2 ----------------
    # If train, test and test_private data is already present in csv format, please comment above block and un-comment
    # below block and use it
    #train_file = 'train_advance_feature.csv.gz'
    #test_file = 'test_advance_feature.csv.gz'
    #test_private_file = 'test_private_v3_track_1.csv.gz'
    #train = pd.read_csv(os.path.join(data_path, train_file), index_col=False)
    #test = pd.read_csv(os.path.join(data_path, test_file), index_col=utils.ID_COLUMN)
    #test_private = pd.read_csv(os.path.join(data_path, test_private_file), index_col=utils.ID_COLUMN)
    #print("Preparing data for training model")
    #features = utils.SIMPLE_FEATURE_COLUMNS
    #features.extend([i for i in range(24)])

    #train_part, validation, train_pool, validation_pool = prepare_data(train, features)
    #print("Training and predicting for track 1")
    #train_model_track1(train_pool, validation_pool, validation, test_private, features, data_path)
    #print("Track 1 completed")
    #print("Training and saving model for track 2")
    #train_model_track2(train_pool, validation_pool, validation, features, data_path)
    #print("Completed successfully")
    # -------------- End of Block 2 -----------------


if __name__ == "__main__":
    data_path = 'Data/'
    main(data_path)
