import time
import autokeras as ak
import random

import tensorflow as tf
import numpy as np
import time
from keras_tuner import Objective

from autokeras import ImageClassifier

from autolens.utils import handle_dataset
from autolens.utils.create_resources_folder import resources
from autolens.utils.handle_autokeras_folder import replace_classifier_folder
from autolens.utils.handle_results import save_results

def main(
        path_metadata: str,
        path_dataset: str,
        steps: int,
        target_size: tuple,
        test_size: float, 
        valid_size: float 
        ):

    """
    :param path_metadata:
    :param path_dataset:
    :param steps:
    :param target_size:
    :param test_size:
    :param valid_size
    """

    # To make sure resources folder is created
    resources()

    print('Time --> Start')
    start_time = time.time()

    # Handling folder created by AutoKeras
    replace_classifier_folder()

    # TODO Check this, make it configurable
    batch_size = 32

    print('Building Architecture')
    model = ImageClassifier(
        project_name='autokeras_model',
        directory='resources/autokeras',
        metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
        # F1-Score not available in this keras version
        max_trials=1,
        objective=Objective('val_auc', direction="max") 
        )

    print('Loading Data')
    
    train_data = ak.image_dataset_from_directory(
        path_dataset,
        # Use 20% data as testing data.
        validation_split=valid_size,
        subset="training",
        # Set seed to ensure the same split when loading testing data.
        seed=random.randint(1, 10000),
        image_size=(150, 150),
        batch_size=batch_size,
    )

    test_data = ak.image_dataset_from_directory(
        path_dataset,
        validation_split=test_size,
        subset="validation",
        seed=random.randint(1, 10000),
        image_size=(150, 150),
        batch_size=batch_size,
    )

    predictions = []
    prob_predictions = []

    print('Fitting Model')
    history = model.fit(train_data, epochs=1)
    score = model.evaluate(test_data)

    # Model is not saved automatically
    best_model = model.export_model()
    best_model.save("resources/autokeras/autokeras_model.keras")
            
    print('Predictions')
    # Predicted labels
    y_predic = model.predict(test_data).astype("int32")
    y_predic = y_predic.flatten()

    predictions.append(y_predic)
    predictions = predictions[0].tolist()

    # Probabilities of the predicted labels
    for i in range(0, len(test_data[0][1]), 32):
        y_prob_int = model.export_model()(test_data[0][1][i:i+32]) # works as tensorflow keras model
        y_prob_int = np.array(y_prob_int).flatten().tolist()
        y_prob_int_rounded = [round(prob, 4) for prob in y_prob_int]
        prob_predictions.extend(y_prob_int_rounded)

    # Actual labels
    y_test = test_data[1][1].tolist()
    print('true labels', len(y_test))

    print('Time --> Stop')
    elapsed_time = time.time() - start_time
    print('Time:', elapsed_time)

    print('Saving Results...')
    all_results = save_results(history, score, predictions, prob_predictions, y_test, elapsed_time)
    all_results.to_csv('resources/autokeras_results.csv', mode='a', index=False)
    print('Results Ready!')