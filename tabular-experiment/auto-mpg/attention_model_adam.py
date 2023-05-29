import pandas as pd
import numpy as np
import numpy.random as npr
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from keras.optimizers import Adam
from keras_nlp.layers import PositionEmbedding

from generate_data import *
from baseline_results import *


def get_masked_input_and_labels(encoded_texts, n_cat):
    # For each sentence, mask each word one-by-one
    encoded_texts_masked = []
    y_labels = []
    sample_weights = []

    for encoded_text in encoded_texts:
        for i in range(len(encoded_text)):
            encoded_text_masked = np.copy(encoded_text)
            encoded_text_masked[i] = n_cat
            y_label = encoded_text
            sample_weight = np.zeros(len(encoded_text))
            sample_weight[i] = 1
            encoded_texts_masked.append(encoded_text_masked)
            y_labels.append(y_label)
            sample_weights.append(sample_weight)

    return np.array(encoded_texts_masked), np.array(y_labels), np.array(sample_weights)


# In[6]:


def bert_module(query, key, value, i, embed_dim, ff_dim, num_head):
    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_head,
        key_dim=embed_dim // num_head,
        name="encoder_{}/multiheadattention".format(i),
    )(query, key, value)
    attention_output = layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
        attention_output
    )
    attention_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ],
        name="encoder_{}/ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output


def train_attention_model(train, test, n_cat, embed_dim, ff_dim, num_head, seed):
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    
    x_train = np.array(train)
    x_masked_train, y_masked_labels_train, sample_weights_train = get_masked_input_and_labels(x_train, n_cat)
    x_test = np.array(test)
    x_masked_test, y_masked_labels_test, sample_weights_test = get_masked_input_and_labels(x_test, n_cat)
    
    inputs = layers.Input((x_masked_train.shape[1],), dtype=tf.int64)
    word_embeddings = layers.Embedding(n_cat + 1, embed_dim, name="word_embedding")(inputs)
    position_embeddings = PositionEmbedding(sequence_length=x_masked_train.shape[1])(word_embeddings)
    embeddings = word_embeddings + position_embeddings
    encoder_output = embeddings
    encoder_output = bert_module(encoder_output, encoder_output, encoder_output, 1, embed_dim, ff_dim, num_head)
    mlm_output = layers.Dense(n_cat, name="mlm_cls", activation="softmax")(encoder_output)
    mlm_model = keras.Model(inputs = inputs, outputs = mlm_output)
    adam = Adam()
    mlm_model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)
    history = mlm_model.fit(x_masked_train, y_masked_labels_train, sample_weight=sample_weights_train,
                            epochs=200, batch_size=128, verbose=2)
    
    sl = x_masked_train.shape[1]
        
    temp_train = np.argmax(K.function(inputs = mlm_model.layers[0].input, outputs = mlm_model.layers[-1].output)(x_masked_train) \
        [np.arange(sl-1,x_masked_train.shape[0],sl),sl-1,:], axis = 1)
    temp_test = np.argmax(K.function(inputs = mlm_model.layers[0].input, outputs = mlm_model.layers[-1].output)(x_masked_test) \
        [np.arange(sl-1,x_masked_test.shape[0],sl),sl-1,:], axis = 1)
    
    mse_train =  ((temp_train - train['y'])**2).mean()
    mse_test = ((temp_test - test['y'])**2).mean()
    acc_train = (temp_train == train['y']).mean()
    acc_test = (temp_test == test['y']).mean()
    
    return (mse_train, acc_train, mse_test, acc_test)