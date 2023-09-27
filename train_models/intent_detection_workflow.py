import os

import emoji

# import gdown
# import shutil
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

import matplotlib
import seaborn

from pylab import rcParams

from sklearn.preprocessing import LabelBinarizer

from bert_model_urls import bert_model_urls
from plot_results import plot_results


def clean_data(data):
    corrupted_data = data[ pd.isna(data['posts']) ]
    print("Number of corrupted data points: ", corrupted_data.shape)
    corrupted_data.to_csv("../data/corrupted_data.csv")
    data = data[ pd.notna(data['posts']) ]
    data['posts'] = data['posts'].map( 
            lambda post: emoji.replace_emoji(post, replace='')
        )

    return data

def preprocess(data, binarizer):
    features = data.copy()
    labels   = features.pop('label')
    print(f"Some of the labels: {labels[0:5]}")
    # features = features['posts']
    features = features.pop('posts')
    labels = binarizer.transform(labels.values)
    print(f"Some of the binarized labels: {labels[0:5]}")
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
    root = os.path.join('model', 'BERT')
    model_name = ''.join(encoder)
    model_path = os.path.join(root, model_name)

    if not os.path.exists(model_path):

        make_dir(root)

        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

        preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)

        encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)

        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(3, activation=None, name='classifier')(net)

        model = tf.keras.Model(text_input, net)

        print('Model name:', model_path)
        model.save(model_path)

        return model

    print('\nReading BERT model from file.\n')
    return tf.keras.models.load_model(model_path)


def split_data_helper(data, train_data_ratio):
    # Shuffle
    data = data.sample(frac=1.0)

    num_samples, _ = data.shape

    num_train_samples     = int(train_data_ratio * num_samples)
    num_validate_samples = int( ( num_samples - num_train_samples ) / 2 )

    train_samples = data.iloc[0:num_train_samples, :]
    validate_upper = num_train_samples + num_validate_samples
    validate_samples = data.iloc[num_train_samples:validate_upper, :]
    test_samples = data.iloc[validate_upper:, :]

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
    train_data    = pd.concat([train_tenants, train_landlords, train_irrelevant])
    validate_data = pd.concat([validate_tenants, validate_landlords, validate_irrelevant])
    test_data     = pd.concat([test_tenants, test_landlords, test_irrelevant])

    train_data = train_data.sample(frac=1.0)

    return train_data, validate_data, test_data


def debug_data(features, labels):
    try:
        loss, accuracy = classifier_model.evaluate(features, labels)
        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}\n')
    except:
        for f in features:
            try:
                a = []
                a.append(f)
                classifier_model.predict(tf.constant(a))
            except Exception as e:
                print('Errored out')
                print(f)
                raise e


tf.get_logger().setLevel('ERROR')

seaborn.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ['#01BEFE', '#FFDD00', '#FF7D00', '#FF006D', '#ADFF02', '#8F00FF']
seaborn.set_palette(seaborn.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8


import warnings
warnings.filterwarnings('ignore')


# LOAD DATA
# DATA_PATH = '..\\data\\labeled_intent_detection_dataset.csv'
DATA_PATH = os.path.join('..', 'data', 'labeled_intent_detection_dataset.csv')
data = pd.read_csv(DATA_PATH)

data = clean_data(data)

print('All the data as loaded: ', data.shape)
traindf, validdf, testdf = split_data(data, 0.7)

chart = seaborn.countplot(x=traindf['label'], palette=HAPPY_COLORS_PALETTE)
chart.set_title("Number of texts per intent")
#plt.title('Number of texts per intent')
chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right')
# plt.waitforbuttonpress()

binarizer = LabelBinarizer().fit(traindf["label"])
train_features, train_labels = preprocess(traindf, binarizer)
test_features, test_labels   = preprocess(testdf, binarizer)
validate_features, validate_labels = preprocess(validdf, binarizer)

# Load BERT model from Tensorflow Hub
model_name = 'small_bert/bert_en_uncased_L-8_H-512_A-8'
tfhub_handle_encoder, tfhub_handle_preprocess = bert_model_urls(model_name)
print(f'BERT model selected: {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

# trained_model_dir = 'model/fine_tuned'
trained_model_dir = os.path.join('model', 'fine_tuned')

retrain = False
fine_tuned_model_name = model_name.replace('/', '-')
fine_tuned_model_name = fine_tuned_model_name.replace('\\', '-')

# trained_model_path = trained_model_dir + '\\' + fine_tuned_model_name
trained_model_path = os.path.join(trained_model_dir, fine_tuned_model_name)
# DEFINE THE MODEL: Preprocessing layer, BERT model, dense layer, dropout layer
if retrain or not os.path.exists(trained_model_path):
    print('\nRe-training a fresh model\n')
    classifier_model = build_classifier_model(tfhub_handle_preprocess, tfhub_handle_encoder)
else:
    print('\nLoading a pre-trained model\n')
    classifier_model = tf.keras.models.load_model(trained_model_path)

print(f"Model input shape: {train_features.shape}")
print(f"Some of the data: {train_features[0:5]}")
print(f"One sample: {train_features.iloc[0]}")
# bert_raw_result = classifier_model(tf.constant(train_features[0]))
input = ['any available room near town or in town...atleast where i can take walk from where i stay to town']
bert_raw_result  = classifier_model(tf.constant(input))
bert_raw_result2 = classifier_model.predict(tf.constant(input))
print(tf.keras.activations.softmax(bert_raw_result))
print('\n')
print(binarizer.inverse_transform(bert_raw_result.numpy()))
print(tf.sigmoid(bert_raw_result2))

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
    print('\nUsing test data to evaluating a trained model')
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
    print(f'Train labels: {train_labels[0:5]}')
    print(f'Train labels type: {train_labels.dtype}, shape: {train_labels.shape}')
    print(f'Train input type: {train_features.dtype}, shape: {train_features.shape}')
    # loss, accuracy = classifier_model.evaluate(train_features, train_labels)
    debug_data(train_features, train_labels)

    # loss, accuracy = classifier_model.evaluate(test_features, test_labels)
    debug_data(test_features, test_labels)

    # loss, accuracy = classifier_model.evaluate(validate_features, validate_labels)
    debug_data(validate_features, validate_labels)

# PLOT ACCURACY AND LOSS OVER TIME
if history is not None:
    plot_results(history)
