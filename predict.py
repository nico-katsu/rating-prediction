import json
from operator import itemgetter

from tensorflow.keras.optimizers import Adam
from pyspark import SparkContext
import os, sys
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.backend import sqrt, mean, square

os.environ['PYSPARK_PYTHON'] = 'python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3.6'

test_file, output_file = sys.argv[1:]
if __name__ == '__main__':
    sc = SparkContext()
    with open('indices.json') as i_in, open('user_avg.json') as user_avg_in, open(
            'business_avg.json') as business_avg_in:
        user, business = json.load(i_in)[:2]
        user_avg = json.load(user_avg_in)
        business_avg = json.load(business_avg_in)
    jsonRDD = sc.textFile(test_file).map(json.loads)
    rawRDD = jsonRDD.map(itemgetter('user_id', 'business_id')).filter(
        lambda x: x[0] in user and x[1] in business)
    test_X_user = np.asarray(rawRDD.map(lambda x: user[x[0]]).collect(), dtype=int)
    test_X_business = np.asarray(rawRDD.map(lambda x: business[x[1]]).collect(), dtype=int)


    def rmse(y_true, y_pred):
        return sqrt(mean(square(y_pred - y_true)))


    model = models.load_model('model.h5', compile=False)
    opt_adam = Adam(lr=0.002)
    model.compile(optimizer=opt_adam, loss=[rmse], metrics=[rmse])

    y_pred = np.asarray(model.predict([test_X_user, test_X_business])).argmax(axis=1) / 2
    deUser, deBusiness = list(user.keys()), list(business.keys())
    with open(output_file, 'w') as output:
        for u, b, s in zip(test_X_user, test_X_business, y_pred):
            print(json.dumps(
                {'user_id': deUser[u], 'business_id': deBusiness[b], 'stars': float(s)}),
                file=output)
        empty = jsonRDD.filter(lambda x: x['user_id'] not in user or x['business_id'] not in business).collect()
        for e in empty:
            if e['user_id'] in user_avg:
                e['stars'] = user_avg[e['user_id']]
            elif e['business_id'] in business_avg:
                e['stars'] = business_avg[e['business_id']]
            e['stars'] = 2.5
            print(json.dumps(e), file=output)
