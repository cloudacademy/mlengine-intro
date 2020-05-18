# Adapted from https://github.com/tensorflow/models/tree/master/samples/core/get_started

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd
import tensorflow as tf

def main():
  
  TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
  TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

  CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
  SPECIES = ['Setosa', 'Versicolor', 'Virginica']

  def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

  def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

  (train_x, train_y), (test_x, test_y) = load_data()


  # Feature columns describe how to use the input.
  feature_columns = []
  for key in train_x.keys():
      feature_columns.append(tf.feature_column.numeric_column(key=key))

  # Build 3-layer DNN with 10, 20, 10 units respectively.
  classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    # Three hidden layers
    hidden_units=[10, 20, 10],
    # The model must choose between 3 classes.
    n_classes=3)

  def train_input_fn(features, labels):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(100)

    # Return the dataset.
    return dataset

  classifier.train(
      input_fn=lambda:train_input_fn(train_x, train_y),
      steps=2000)


  # Define the test inputs
  def eval_input_fn(features, labels):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    dataset = dataset.batch(100)

    # Return the dataset.
    return dataset

  # Evaluate the model.
  eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_x, test_y))

  print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


  # Generate predictions from the model
  expected = ['Setosa', 'Versicolor', 'Virginica']
  predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
  }

  predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(predict_x, labels=None))

  template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

  for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(template.format(SPECIES[class_id],
                          100 * probability, expec))


if __name__ == "__main__":
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  tf.logging.set_verbosity(tf.logging.ERROR)
  main()
