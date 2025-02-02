import os
# disable keras loggings
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf

from emotion_recognition import EmotionRecognizer
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, GRU, Dense, Activation, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix

from data_extractor import load_data
from create_csv import write_custom_csv, write_emodb_csv, write_tess_ravdess_csv
from emotion_recognition import EmotionRecognizer
from utils import get_first_letters, AVAILABLE_EMOTIONS, extract_feature, get_dropout_str


import numpy as np
import pandas as pd
import random

class CNNEmotionRecognizer(EmotionRecognizer):
    """
    The Deep Learning version of the Emotion Recognizer.
    This version will use a Convolutional Neural Network (CNN) instead of an RNN.
    """
    def __init__(self, **kwargs):
        """
        params (CNN-specific):
            emotions (list): list of emotions to be used.
            ...
            ==========================================================
            Model params
            n_conv_layers (int): number of convolutional layers, default is 2.
            cnn_filters (int): number of filters for each convolutional layer, default is 64.
            kernel_size (int): kernel size for convolutional layers, default is 3.
            n_dense_layers (int): number of Dense layers, default is 2.
            dense_units (int): number of units in the Dense layers, default is 128.
            dropout (list/float): dropout rate(s); if a float, the same dropout is applied to all layers.
            ==========================================================
            Training params
            batch_size (int): number of samples per gradient update, default is 64.
            epochs (int): number of epochs, default is 100.
            optimizer (str/keras.optimizers.Optimizer instance): optimizer used to train, default is "adam".
            loss (str/keras.losses.Loss instance): loss function to be minimized during training.
        """
        # initialize base class
        super().__init__(**kwargs)

        # CNN-specific parameters:
        self.n_conv_layers = kwargs.get("n_conv_layers", 2)
        self.cnn_filters = kwargs.get("cnn_filters", 64)
        self.kernel_size = kwargs.get("kernel_size", 3)
        self.n_dense_layers = kwargs.get("n_dense_layers", 2)
        self.dense_units = kwargs.get("dense_units", 128)
        
        # Dropout: one value for each conv and dense layer (total layers = n_conv_layers + n_dense_layers)
        self.dropout = kwargs.get("dropout", 0.3)
        if not isinstance(self.dropout, list):
            self.dropout = [self.dropout] * (self.n_conv_layers + self.n_dense_layers)
        
        # Number of output classes (emotions)
        self.output_dim = len(self.emotions)

        # Optimization parameters
        self.optimizer = kwargs.get("optimizer", "adam")
        self.loss = kwargs.get("loss", "categorical_crossentropy")
        
        # Training parameters
        self.batch_size = kwargs.get("batch_size", 64)
        self.epochs = kwargs.get("epochs", 100)

        # Set a model name based on parameters (for saving purposes)
        self.model_name = ""
        self._update_model_name()

        # Placeholder for the model itself
        self.model = None

        # Compute input length (used to build the model)
        self._compute_input_length()

        # Flag to check if the model has been created
        self.model_created = False

    def _update_model_name(self):
        """
        Generates a unique model name based on parameters.
        For the CNN version, the model name reflects that it uses convolutional layers.
        """
        # e.g. "angry-happy-sad" could become "AHS"
        emotions_str = get_first_letters(self.emotions)
        # 'c' for classification, 'r' for regression
        problem_type = 'c' if self.classification else 'r'
        # build dropout string from the list
        dropout_str = get_dropout_str(self.dropout, n_layers=self.n_conv_layers + self.n_dense_layers)
        # Construct a name that includes CNN-specific parameters
        self.model_name = (
            f"{emotions_str}-{problem_type}-CNN-"
            f"convLayers-{self.n_conv_layers}-denseLayers-{self.n_dense_layers}-"
            f"filters-{self.cnn_filters}-denseUnits-{self.dense_units}-"
            f"kernelSize-{self.kernel_size}-dropout-{dropout_str}.h5"
        )

    def _compute_input_length(self):
        """
        Calculates the input shape (the sequence length) to be used to build the model.
        """
        if not self.data_loaded:
            self.load_data()
        # Now, each sample is of shape (feature_length, 1)
        self.input_length = self.X_train[0].shape[0]


    def _verify_emotions(self):
        super()._verify_emotions()
        self.int2emotions = {i: e for i, e in enumerate(self.emotions)}
        self.emotions2int = {v: k for k, v in self.int2emotions.items()}
    
    def _get_model_filename(self):
        """Returns the relative path of this model name"""
        return f"results/{self.model_name}"

    def _model_exists(self):
        """
        Checks if model already exists in disk, returns the filename,
        and returns `None` otherwise.
        """
        filename = self._get_model_filename()
        return filename if os.path.isfile(filename) else None

    def create_model(self):
        """
        Constructs the CNN-based neural network using parameters passed.
        """
        if self.model_created:
            return

        if not self.data_loaded:
            self.load_data()

        model = Sequential()

        # Convolutional layers
        for i in range(self.n_conv_layers):
            if i == 0:
                # First conv layer with input shape defined
                model.add(Conv1D(filters=self.cnn_filters,
                                kernel_size=self.kernel_size,
                                activation="relu",
                                input_shape=(self.input_length, 1),
                                padding="same"))
            else:
                model.add(Conv1D(filters=self.cnn_filters,
                                kernel_size=self.kernel_size,
                                activation="relu",
                                padding="same"))
            model.add(Dropout(self.dropout[i]))
            model.add(MaxPool1D(pool_size=2))

        model.add(GlobalAveragePooling1D())

        for j in range(self.n_dense_layers):
            model.add(Dense(self.dense_units, activation="relu"))
            model.add(Dropout(self.dropout[self.n_conv_layers + j]))

        if self.classification:
            model.add(Dense(self.output_dim, activation="softmax"))
            model.compile(loss=self.loss, metrics=["accuracy"], optimizer=self.optimizer)
        else:
            model.add(Dense(1, activation="linear"))
            model.compile(loss="mean_squared_error", metrics=["mean_absolute_error"], optimizer=self.optimizer)

        self.model = model
        self.model_created = True
        if self.verbose > 0:
            print("[+] Model created")



    def load_data(self):
        """
        Loads and extracts features from the audio files for the db's specified.
        And then reshapes the data.
        """
        super().load_data()  # Load data from the parent class

        # Get original shapes of the training and testing feature arrays.
        # Expected original shapes: (n_samples, feature_length)
        X_train_shape = self.X_train.shape  
        X_test_shape = self.X_test.shape    
        
        # NEW: Reshape so that each sample is a sequence of length 'feature_length'
        # and there is one channel.
        self.X_train = self.X_train.reshape((X_train_shape[0], X_train_shape[1], 1))
        self.X_test = self.X_test.reshape((X_test_shape[0], X_test_shape[1], 1))

        if self.classification:
            # One-hot encode labels for classification.
            # The resulting shape will be (n_samples, num_classes)
            self.y_train = to_categorical([self.emotions2int[str(e)] for e in self.y_train])
            self.y_test = to_categorical([self.emotions2int[str(e)] for e in self.y_test])
        else:
            self.y_train = self.y_train.reshape((-1, 1))
            self.y_test = self.y_test.reshape((-1, 1))




    def train(self, override=False):
        """
        Trains the neural network.
        Params:
            override (bool): whether to override the previous identical model, can be used
                when you changed the dataset, default is False
        """
        # if model isn't created yet, create it
        if not self.model_created:
            self.create_model()

        # if the model already exists and trained, just load the weights and return
        # but if override is True, then just skip loading weights
        if not override:

            model_name = self._model_exists()

            if model_name:
                self.model.load_weights(model_name)
                self.model_trained = True

                if self.verbose > 0:
                    print("[*] Model weights loaded")
                return


        if not os.path.isdir("results"):
            os.mkdir("results")

        if not os.path.isdir("logs"):
            os.mkdir("logs")


        model_filename = self._get_model_filename()
        print(model_filename)
        # print('2')
        # self.checkpointer = ModelCheckpoint(model_filename, save_best_only=True, verbose=1)
        # print('3')
        #self.tensorboard = TensorBoard(log_dir=os.path.join("logs", self.model_name))

        print("begin training")
        self.history = self.model.fit(self.X_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_data=(self.X_test, self.y_test),
                        #callbacks=[self.checkpointer, self.tensorboard],
                        verbose=self.verbose)
        print("end training")
        
        self.model_trained = True
        print("[+] Model trained")


    def predict(self, audio_path):
        feature = extract_feature(audio_path, **self.audio_config).reshape((1, 1, self.input_length))
        if self.classification:
            prediction = self.model.predict(feature)
            prediction = np.argmax(np.squeeze(prediction))
            return self.int2emotions[prediction]
        else:
            return np.squeeze(self.model.predict(feature))

    def predict_proba(self, audio_path):
        if self.classification:
            feature = extract_feature(audio_path, **self.audio_config).reshape((1, 1, self.input_length))
            proba = self.model.predict(feature)[0][0]
            result = {}
            for prob, emotion in zip(proba, self.emotions):
                result[emotion] = prob
            return result
        else:
            raise NotImplementedError("Probability prediction doesn't make sense for regression")



    def test_score(self):
        y_test = self.y_test[0]
        if self.classification:
            y_pred = self.model.predict(self.X_test)[0]
            y_pred = [np.argmax(y, out=None, axis=None) for y in y_pred]
            y_test = [np.argmax(y, out=None, axis=None) for y in y_test]
            return accuracy_score(y_true=y_test, y_pred=y_pred)
        else:
            y_pred = self.model.predict(self.X_test)[0]
            return mean_absolute_error(y_true=y_test, y_pred=y_pred)

    def train_score(self):
        y_train = self.y_train[0]
        if self.classification:
            y_pred = self.model.predict(self.X_train)[0]
            y_pred = [np.argmax(y, out=None, axis=None) for y in y_pred]
            y_train = [np.argmax(y, out=None, axis=None) for y in y_train]
            return accuracy_score(y_true=y_train, y_pred=y_pred)
        else:
            y_pred = self.model.predict(self.X_train)[0]
            return mean_absolute_error(y_true=y_train, y_pred=y_pred)

    def confusion_matrix(self, percentage=True, labeled=True):
        """Compute confusion matrix to evaluate the test accuracy of the classification"""
        if not self.classification:
            raise NotImplementedError("Confusion matrix works only when it is a classification problem")
        y_pred = self.model.predict(self.X_test)[0]
        y_pred = np.array([ np.argmax(y, axis=None, out=None) for y in y_pred])
        # invert from keras.utils.to_categorical
        y_test = np.array([ np.argmax(y, axis=None, out=None) for y in self.y_test[0] ])
        matrix = confusion_matrix(y_test, y_pred, labels=[self.emotions2int[e] for e in self.emotions]).astype(np.float32)
        if percentage:
            for i in range(len(matrix)):
                matrix[i] = matrix[i] / np.sum(matrix[i])
            # make it percentage
            matrix *= 100
        if labeled:
            matrix = pd.DataFrame(matrix, index=[ f"true_{e}" for e in self.emotions ],
                                    columns=[ f"predicted_{e}" for e in self.emotions ])
        return matrix

    def get_n_samples(self, emotion, partition):
        """Returns number data samples of the `emotion` class in a particular `partition`
        ('test' or 'train')
        """
        if partition == "test":
            if self.classification:
                y_test = np.array([ np.argmax(y, axis=None, out=None)+1 for y in np.squeeze(self.y_test) ]) 
            else:
                y_test = np.squeeze(self.y_test)
            return len([y for y in y_test if y == emotion])
        elif partition == "train":
            if self.classification:
                y_train = np.array([ np.argmax(y, axis=None, out=None)+1 for y in np.squeeze(self.y_train) ])
            else:
                y_train = np.squeeze(self.y_train)
            return len([y for y in y_train if y == emotion])

    def get_samples_by_class(self):
        """
        Returns a dataframe that contains the number of training 
        and testing samples for all emotions
        """
        train_samples = []
        test_samples = []
        total = []
        for emotion in self.emotions:
            n_train = self.get_n_samples(self.emotions2int[emotion]+1, "train")
            n_test = self.get_n_samples(self.emotions2int[emotion]+1, "test")
            train_samples.append(n_train)
            test_samples.append(n_test)
            total.append(n_train + n_test)
        
        # get total
        total.append(sum(train_samples) + sum(test_samples))
        train_samples.append(sum(train_samples))
        test_samples.append(sum(test_samples))
        return pd.DataFrame(data={"train": train_samples, "test": test_samples, "total": total}, index=self.emotions + ["total"])

    def get_random_emotion(self, emotion, partition="train"):
        """
        Returns random `emotion` data sample index on `partition`
        """
        if partition == "train":
            y_train = self.y_train[0]
            index = random.choice(list(range(len(y_train))))
            element = self.int2emotions[np.argmax(y_train[index])]
            while element != emotion:
                index = random.choice(list(range(len(y_train))))
                element = self.int2emotions[np.argmax(y_train[index])]
        elif partition == "test":
            y_test = self.y_test[0]
            index = random.choice(list(range(len(y_test))))
            element = self.int2emotions[np.argmax(y_test[index])]
            while element != emotion:
                index = random.choice(list(range(len(y_test))))
                element = self.int2emotions[np.argmax(y_test[index])]
        else:
            raise TypeError("Unknown partition, only 'train' or 'test' is accepted")

        return index

    def determine_best_model(self):
        # TODO
        # raise TypeError("This method isn't supported yet for deep nn")
        pass


if __name__ == "__main__":
    rec = CNNEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'])

    rec.train(override=False)
    print("Test accuracy score:", rec.test_score() * 100, "%")