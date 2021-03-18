import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

from models.dbn.dbn.tensorflow import SupervisedDBNClassification


# use "from dbn import SupervisedDBNClassification" for computations on CPU with numpy


def run(dataset):

    x, y = dataset[:, :2048], dataset[:, -1:]
    # Splitting data
    x_train, x_train, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Training
    classifier = SupervisedDBNClassification(hidden_layers_structure=[167, 167],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=10,
                                             n_iter_backprop=100,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    classifier.fit(x_train, y_train)

    # Save the model
    classifier.save('D:\\5th\Honours\Code\models\weights\dbn.pkl')

    # Restore it
    classifier = SupervisedDBNClassification.load('model.pkl')

    # Test
    y_pred = classifier.predict(x_test)
    print('Done.\nAccuracy: %f' % accuracy_score(y_test, y_pred))
