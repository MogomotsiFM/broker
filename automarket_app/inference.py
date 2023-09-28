import os

from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf

model_name = 'small_bert-bert_en_uncased_L-8_H-512_A-8'
model_path = os.path.join('..', 'fine-tuned', model_name)
print('Reading BERT model from file.')
model = tf.keras.models.load_model(model_path)

def infer_intent(post):
    prediction = tf.sigmoid( model.predict(tf.constant(post)) )
    binarizer = LabelBinarizer.fit( ['irrelevant', 'landlord', 'tenant'] )
    return binarizer.inverse_transform( prediction.numpy() )
    
