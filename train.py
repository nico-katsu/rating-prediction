from pyspark import SparkContext
# %%
import json
import os
from operator import itemgetter

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ['PYSPARK_PYTHON'] = 'python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3.6'
from keras.backend import sqrt, mean, square
from keras.optimizers import Nadam

from keras import Input, Model
from keras.layers import Embedding, Flatten, Dropout, concatenate, Dense, multiply, Softmax, \
    BatchNormalization, ReLU
from tensorflow.keras.regularizers import l2
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # %%
    sc = SparkContext()
    rawRDD = sc.textFile('train_review.json').map(json.loads).map(itemgetter("user_id", 'business_id', 'stars'))
    userRDD = rawRDD.map(itemgetter(0)).distinct()
    user = userRDD.zipWithIndex().collectAsMap()
    deUser = userRDD.collect()
    businessRDD = rawRDD.map(itemgetter(1)).distinct()
    business = businessRDD.zipWithIndex().collectAsMap()
    deBusiness = businessRDD.collect()
    X_user = np.asarray(rawRDD.map(lambda x: user[x[0]]).collect(), dtype=int)
    X_business = np.asarray(rawRDD.map(lambda x: business[x[1]]).collect(), dtype=int)
    y = np.asarray(rawRDD.map(itemgetter(2)).collect(), dtype=float) - 1
    n_business, n_user = len(business), len(user)

    # %%
    dim_embeddings = 10

    user_input = Input(shape=[1], name='user')
    user_embedding = Embedding(n_user + 1, dim_embeddings, name='user-embedding')(user_input)
    user_vec = Flatten(name='user-flatten')(user_embedding)
    # user_vec = Dropout(0.1)(user_vec)

    business_input = Input(shape=[1], name='business')
    business_embedding = Embedding(n_business + 1, dim_embeddings, name='business-embedding')(business_input)
    business_vec = Flatten(name='business-flatten')(business_embedding)
    # business_vec = Dropout(0.1)(business_vec)

    concat = concatenate([user_vec, business_vec])
    # concat_dropout = Dropout(0.2)(concat)
    tdense = [concat]
    for w in [20, 10]:
        tdense.append(
            Dense(w, name=f'fully-connected{w}', activation='relu')(tdense[-1]))

    matrix_product = Flatten()(multiply([user_embedding, business_embedding]))

    concat_2 = concatenate([matrix_product, tdense[-1]])
    mdense = [concat_2]
    for w in [10, 5]:
        mdense.append(
            Dense(w, name=f'fully-connected-m{w}', activation='relu')(mdense[-1]))

    stars = Dense(1, name='stars', activation='relu')(mdense[-1])
    # stars = Softmax()(mdense[-1])
    model = Model([user_input, business_input], stars)
    # model.summary()

    opt_adam = Nadam(lr=0.001)


    def rmse(y_true, y_pred):
        return sqrt(mean(square(y_pred - y_true)))


    model.compile(optimizer=opt_adam, loss=[rmse],
                  metrics=[rmse])

    import keras.datasets.mnist

    x=keras.datasets.mnist.load_data()


    model.fit([X_user, X_business], y, batch_size=256, validation_split=0.005, epochs=2)

    model.save('model.h5')
    with open('indices.json', 'w') as i_out:
        json.dump([user, business], i_out)

    # %%
    testRDD = sc.textFile('test_review_ratings.json').map(json.loads).map(
        itemgetter('user_id', 'business_id', 'stars')).filter(lambda x: x[0] in user and x[1] in business)
    test_X = [np.asarray(testRDD.map(lambda x: user[x[0]]).collect(), dtype=int),
              np.asarray(testRDD.map(lambda x: business[x[1]]).collect(), dtype=int)]
    test_y = np.asarray(testRDD.map(itemgetter(2)).collect(), dtype=float) - 1

    # %%
    model.evaluate(test_X, test_y)
