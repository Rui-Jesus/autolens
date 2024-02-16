import time
import random

from autogluon.multimodal import MultiModalPredictor

from autolens.utils import calculate_metrics
from autolens.utils import handle_dataset
from autolens.utils.create_resources_folder import resources
from autolens.utils.handle_autogluon_folder import replace_classifier_folder
from autolens.utils.handle_results import save_results
from autolens.dataset.build_2 import Dataset

def main(path_metadata: str,
        path_dataset: str,
        steps: int,
        target_size: tuple,
        test_size: float,
        valid_size:float
        ):

    """
    :param path_metadata:
    :param path_dataset:
    :param steps:
    :param target_size:
    :param test_size:
    :param valid_size:
    :return:
    """
    
    resources()

    print('Time --> Start')
    start_time = time.time()

    # Handling folders created by AutoGluon
    replace_classifier_folder()

    print('Loading Data')
    dataset = Dataset(path_dataset=path_dataset, 
                    test_size=test_size, 
                    valid_size=valid_size)
    
    train, test, valid = dataset.to_path(test_size=test_size,
                                        valid_size=valid_size,
                                        test_seed=random.randint(1, 10000),
                                        valid_seed=random.randint(1, 10000))
    
    print('Building Architecture')
    classifier = MultiModalPredictor(
            label="label",
            problem_type='binary',
            path='resources/autogluon',
            verbosity=4,
            eval_metric='roc_auc',
            validation_metric='roc_auc'
            )

    print('Fitting Model')
    history = classifier.fit(
            train_data=train,
            tuning_data=valid,
            presets='high_quality', # medium_quality / high_quality / best_quality
            hyperparameters={
                "env.num_workers": 1,
                "env.num_workers_evaluation": 1, # otherwise arrises an ssh error 
                "env.num_gpus": 1
                },
            time_limit=60*15 # in seconds
            )

    histories = classifier.fit_summary(verbosity=4, show_plot=True)
    
    # Model is automatically saved
    # loaded_predictor = MultiModalPredictor.load(model_path)

    print('Evaluating Model')
    scores = classifier.evaluate(
            data=test, 
            metrics=["accuracy", "precision", "recall", "f1", "roc_auc", "mcc"])

    print(scores)

    print('Predictions')
    # Predicted labels
    y_pred = classifier.predict(test).tolist() 

    # Probabilities of the predicted labels
    prob = classifier.predict_proba(test)
    y_prob = [sublist[0] for sublist in prob]
    #y_prob = (prob[[0, 1]].max(axis=1)).tolist()

    # Actual labels
    y_test = test['label'].tolist()

    print('Time --> Stop')
    elapsed_time = time.time() - start_time
    print('Time:', elapsed_time)

    print('Saving Results...')
    all_results = save_results(histories, scores, y_pred, y_prob, y_test, elapsed_time)
    all_results.to_csv('resources/autogluon_results.csv', mode='a', index=False)
    print('Results Ready!')
