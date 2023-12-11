from tg import expose, TGController, AppConfig
from wsgiref.simple_server import make_server
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import scipy.stats as stats


iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)


model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

class MyController(TGController):
    @expose()
    def index(self):
        return "Hello, TurboGears"

    @expose('json')
    def prediction(self, input_data):# Use the trained model for prediction
        input_array = np.array([input_data])
        prediction = model.predict(input_array)
        return {'prediction': prediction.tolist()}

    @expose('json')
    def plot_histogram(self):
    
        data = np.random.randn(1000)
        plot_data = sns.histplot(data, kde=True).get_lines()[0].get_data()
        return {'plot_data': plot_data.tolist()}

    @expose('json')
    def statistical_analysis(self):
        
        mean, p_value = stats.ttest_1samp(data, 0)
        return {'mean': mean, 'p_value': p_value}

config = AppConfig(minimal=True, root_controller=MyController())


app = config.make_wsgi_app()


server = make_server('', 8080, app)
server.serve_forever()
