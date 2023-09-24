import os

import gdown
#import shutil
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import seaborn as sns
from pylab import rcParams

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer

from bert_model_urls import bert_model_urls
from plot_results import plot_results


def preprocess(data, binarizer):
    features = data.copy()
    labels   = features.pop('intent')
    # trainlabels   = traindf['intent']
    features = features.values
    labels = binarizer.fit_transform(labels.values)

    return features, labels


def make_dir(path):
    root = os.getcwd()
    path.replace('/', '\\')
    dirs = path.split('\\')
    for dir in dirs:
        if dir != '':
            if not os.path.exists(dir):
                os.mkdir(dir)
            os.chdir(dir)

    os.chdir(root)


def build_classifier_model(tfhub_handle_preprocess, tfhub_handle_encoder):
    encoder = [c for c in tfhub_handle_encoder if c not in ('\\/:')]
    root = 'model\\BERT'
    model_name = ''.join(encoder)
    model_path = root + '\\' + model_name

    if not os.path.exists(model_path):

        make_dir(root)

        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

        preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)

        encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)

        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(7, activation=None, name='classifier')(net)

        model = tf.keras.Model(text_input, net)

        print('Model name:', model_path)
        model.save(model_path)

        return model

    print('\nReading BERT model from file.\n')
    return tf.keras.models.load_model(model_path)

def split_data_helper(data, train_data_ratio):
    data = data.sample(frac=1.0)

    num_samples, _ = data.shape

    num_train_samples     = int(train_data_ratio * num_samples)
    num_validate_samples = int( ((1.0 - train_data_ratio) / 2) * num_train_samples )

    train_samples = data[:, 0:num_train_samples]
    validate_upper = num_train_samples + num_validate_samples
    validate_samples = data[:, num_train_samples:validate_upper]
    test_samples = data[:, validate_upper:]

    return train_samples, validate_samples, test_samples


def split_data(data, train_data_ratio):
    # Separate the classes
    tenants    = data[ data['label'] == 'tenant' ]
    landlords  = data[ data['label'] == 'landlord' ]
    irrelevant = data[ data['label'] == 'irrelevant']

    # Split the data
    train_tenants, validate_tenants, test_tenants = split_data_helper(tenants, train_data_ratio)
    train_landlords, validate_landlords, test_landlords = split_data_helper(landlords, train_data_ratio)
    train_irrelevant, validate_irrelevant, test_irrelevant = split_data_helper(irrelevant, train_data_ratio)

    # Ensure that all classes are equally represented during training

    # Combine the data into test, validation and test data
    train_data    = pd.DataFrame([train_tenants, train_landlords, train_irrelevant])
    validate_data = pd.DataFrame([validate_tenants, validate_landlords, validate_irrelevant])
    test_data     = pd.DataFrame([test_tenants, test_landlords, test_irrelevant])

    train_data = train_data.sample(frac=1.0)

    return train_data, validate_data, test_data


tf.get_logger().setLevel('ERROR')

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ['#01BEFE', '#FFDD00', '#FF7D00', '#FF006D', '#ADFF02', '#8F00FF']
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8


import warnings
warnings.filterwarnings('ignore')


# LOAD DATA
DATA_PATH = '..\\data\\labeled_intent_detection_dataset.csv'
data = pd.read_csv(DATA_PATH)

traindf, validdf, testdf = split_data(data, 0.7)

chart = sns.countplot(x=traindf['intent'], palette=HAPPY_COLORS_PALETTE)
plt.title('Number of texts per intent')
chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right')
# plt.waitforbuttonpress()

binarizer = LabelBinarizer()
train_features, train_labels = preprocess(traindf, binarizer)
test_features, test_labels   = preprocess(testdf, binarizer)
validate_features, validate_labels = preprocess(validdf, binarizer)

# Load BERT model from Tensorflow Hub
model_name = 'small_bert/bert_en_uncased_L-8_H-512_A-8'
tfhub_handle_encoder, tfhub_handle_preprocess = bert_model_urls(model_name)
print(f'BERT model selected: {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

trained_model_dir = 'model/fine_tuned'

retrain = False
fine_tuned_model_name = model_name.replace('/', '-')
fine_tuned_model_name = fine_tuned_model_name.replace('\\', '-')

trained_model_path = trained_model_dir + '\\' + fine_tuned_model_name

# DEFINE THE MODEL: Preprocessing layer, BERT model, dense layer, dropout layer
if retrain or not os.path.exists(trained_model_path):
    print('\nRe-training a fresh model\n')
    classifier_model = build_classifier_model(tfhub_handle_preprocess, tfhub_handle_encoder)
else:
    print('\nLoading a pre-trained model\n')
    classifier_model = tf.keras.models.load_model(trained_model_path)

bert_raw_result = classifier_model(tf.constant(train_features[0]))
print(tf.keras.activations.softmax(bert_raw_result))

# TRAIN THE MODEL
history = None
if retrain or not os.path.exists(trained_model_path):
    print('\nTraining')
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = tf.metrics.CategoricalAccuracy()

    epochs = 5
    optimizer = tf.keras.optimizers.Adam(1e-5)
    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print(f'Training model with {tfhub_handle_encoder}')
    history = classifier_model.fit(x=train_features,
                                    y=train_labels,
                                    validation_data=(validate_features, validate_labels),
                                    batch_size=8,
                                    epochs=epochs)

    if trained_model_dir is not None:
        curr_dir = os.getcwd()
        make_dir(trained_model_dir)
        os.chdir(curr_dir)

    print('\nSaving a trained model\n')
    classifier_model.save(trained_model_path)


# EVALUATE THE MODEL
if retrain:
    print('\nUsing validation data to evaluating a trained model')
    loss, accuracy = classifier_model.evaluate(test_features, test_labels)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}\n')
else:
    # EVALUATE THE MODEL
    print(f'Features: {train_features.shape}, {test_features.shape}, {validate_features.shape}')
    print('Labels: ', test_labels)
    # features = np.concatenate((test_features, train_features, validate_features))
    # labels   = np.concatenate((test_labels, train_labels, validate_features))

    print('\nEvaluate the trained model')
    loss, accuracy = classifier_model.evaluate(train_features, train_labels)
    print('Training')
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}\n')

    loss, accuracy = classifier_model.evaluate(test_features, train_labels)
    print('Test')
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}\n')

    loss, accuracy = classifier_model.evaluate(validate_features, validate_labels)
    print('Validate')
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}\n')

# PLOT ACCURACY AND LOSS OVER TIME
if history is not None:
    plot_results(history)
