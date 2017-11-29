import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import MLPClassifier

custom_data_home = 'D:\Christian-Data\Proyectos\Python\data'
mnist  = datasets.fetch_mldata('MNIST original', data_home=custom_data_home)
# rescale the data, use the traditional train/test split
X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

mlp = MLPClassifier(hidden_layer_sizes=(5,5,5,5), max_iter=5, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))
