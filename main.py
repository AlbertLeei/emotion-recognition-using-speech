from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D, Dense, Dropout
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors
from utils import AVAILABLE_EMOTIONS

from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC
# init a model, let's use SVC
my_model = SVC()
# pass my model to EmotionRecognizer instance
# and balance the dataset
# ['sad', 'neutral', 'happy']
rec = EmotionRecognizer(model=my_model, emotions=AVAILABLE_EMOTIONS, balance=True, verbose=0)

# loads the best estimators from `grid` folder that was searched by GridSearchCV in `grid_search.py`,
# and set the model to the best in terms of test score, and then train it
rec.determine_best_model()
# get the determined sklearn model name
print(rec.model.__class__.__name__, "is the best")
# get the test accuracy score for the best estimator
print("Test score:", rec.test_score())



# train the model
# rec.train()
# # check the test accuracy for that model
# print("Test score:", rec.test_score())
# # check the train accuracy for that model
# print("Train score:", rec.train_score())