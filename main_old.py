#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries and Initiating PySpark

# In[82]:
#import sys
#import tensorflow as tf
#print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import numpy as np
import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import array
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
import sklearn
from sklearn.preprocessing import MinMaxScaler #
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import LSTM, Dense, Activation


# In[83]:


import findspark
findspark.init()


# In[84]:


try:
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
except ImportError as e:
    print('<<<<<!!!!! Please restart your kernel after installing Apache Spark !!!!!>>>>>')


# In[88]:


sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()


# In[89]:


atk_files = ['./data/decimal/decimal_DoS.csv', \
             './data/decimal/decimal_spoofing-GAS.csv', \
             './data/decimal/decimal_spoofing-RPM.csv', \
             './data/decimal/decimal_spoofing-SPEED.csv', \
             './data/decimal/decimal_spoofing-STEERING_WHEEL.csv']

atk_data = pd.concat([pd.read_csv(file) for file in atk_files])
atk_data.to_csv('decimal_attack.csv', index=False)


# ### Data Loading

# In[90]:


attack_df = spark.read.csv('./decimal_attack.csv', header = True)
attack_df.createOrReplaceTempView('attack')
attack_df.show()


# In[91]:


benign_df = spark.read.csv('./data/decimal/decimal_benign.csv', header = True)
benign_df.createOrReplaceTempView('benign')
benign_df.show()


# ### Data Cleaning and Processing

# In[92]:


# renaming an existing label column for downstream feature engineering
attack_df = attack_df.withColumnRenamed("label", "string")
benign_df = benign_df.withColumnRenamed("label", "string")

# dropping irrelevant columns
attack_df = attack_df.drop('ID', 'category', 'specific_class')
benign_df = benign_df.drop('ID', 'category', 'specific_class')

# changing data types of feature columns
cols_to_cast = ['DATA_0','DATA_1','DATA_2','DATA_3','DATA_4','DATA_5','DATA_6','DATA_7']
for col_name in cols_to_cast:
    attack_df = attack_df.withColumn(col_name, col(col_name).cast("int"))
    benign_df = benign_df.withColumn(col_name, col(col_name).cast("int"))


# ### Test and Train Dataset Creation

# In[93]:


# 70% train data, 30% test from both attack and benign datasets
split = [0.7,0.3]
atk_dfs = attack_df.randomSplit(split)
benign_dfs = benign_df.randomSplit(split)

# combine splits from both datasets into new train and test datasets with equal ratios
train_df = benign_dfs[0].union(atk_dfs[0])
test_df = benign_dfs[1].union(atk_dfs[1])

train_df.show()
test_df.show()


# ### Feature Engineering

# In[94]:


indexer = StringIndexer(inputCol = 'string', outputCol = 'label')

train_data = indexer.fit(train_df).transform(train_df)
test_data = indexer.fit(test_df).transform(test_df)


# In[95]:


pandas_df = train_data.toPandas()

# Convert Pandas DataFrame to NumPy array
numpy_array = pandas_df.to_numpy()

X_train = numpy_array[:, :-1]  # Input features (all columns except the last one)
y_train = numpy_array[:, -1]   # Target variable (last column)


# In[96]:


pandas_df = test_data.toPandas()

# Convert Pandas DataFrame to NumPy array
numpy_array = pandas_df.to_numpy()

X_test = numpy_array[:, :-1]  # Input features (all columns except the last one)
y_test = numpy_array[:, -1]   # Target variable (last column)


# In[97]:


X_train = X_train[:, :-1]  # Remove last column
X_test = X_test[:, :-1]

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)


# ### Model Creation

# In[98]:
from keras import Sequential, Input

model = Sequential([
    Input(shape=(X_train.shape[1],)),  # ✅ 권장
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# ### Model Training

# In[99]:


model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)


# ### Model Evaluation

# In[58]:


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


# In[101]:


import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, array
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import LSTM, Dense, Activation
import findspark
import matplotlib.pyplot as plt # For plotting
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from collections import Counter # For class distribution



try:
    y_train_int = y_train.astype(int)
    y_test_int = y_test.astype(int)
except NameError:
    print("Error: y_train or y_test not defined. Please ensure previous cells ran successfully.")
    # Exit the cell or handle the error appropriately
    raise # Re-raise the error to stop execution



print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train_int: {y_train_int.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test_int: {y_test_int.shape}")
print(f"Data type of X_train: {X_train.dtype}")
print(f"Data type of y_train_int: {y_train_int.dtype}")
print(f"Data type of X_test: {X_test.dtype}")
print(f"Data type of y_test_int: {y_test_int.dtype}")



print("\nClass distribution:")
print("Train distribution:", Counter(y_train_int))
print("Test distribution:", Counter(y_test_int))


#Evaluate LSTM Model
if 'model' in globals() and model is not None:
    print("\nEvaluating LSTM model on test set...")

    loss_lstm, accuracy_lstm = model.evaluate(X_test, y_test, verbose=0)
    lstm_preds = (model.predict(X_test) > 0.5).astype(int).flatten()


    precision_lstm = precision_score(y_test_int, lstm_preds, average='weighted')
    recall_lstm = recall_score(y_test_int, lstm_preds, average='weighted')
    f1_lstm = f1_score(y_test_int, lstm_preds, average='weighted')

    print(f'LSTM Test Loss: {loss_lstm:.4f}, Test Accuracy: {accuracy_lstm:.4f}')
    print(f'LSTM Precision: {precision_lstm:.4f}')
    print(f'LSTM Recall: {recall_lstm:.4f}')
    print(f'LSTM F1 Score: {f1_lstm:.4f}')

    # Plot LSTM Confusion Matrix
    cm_lstm = confusion_matrix(y_test_int, lstm_preds)
    disp_lstm = ConfusionMatrixDisplay(confusion_matrix=cm_lstm, display_labels=["Benign", "Attack"])
    disp_lstm.plot(cmap=plt.cm.Oranges)
    plt.title("LSTM Confusion Matrix")
    plt.grid(False)
    plt.show()

else:
     print("\nLSTM model ('model') not found or not trained. Skipping LSTM evaluation.")


#Train and Evaluate Logistic Regression
print("\nTraining and Evaluating Logistic Regression...")
from sklearn.linear_model import LogisticRegression


log_reg_model = LogisticRegression(solver='liblinear', random_state=42)


log_reg_model.fit(X_train, y_train_int)


log_reg_preds = log_reg_model.predict(X_test)


log_reg_accuracy = accuracy_score(y_test_int, log_reg_preds)
log_reg_precision = precision_score(y_test_int, log_reg_preds, average='weighted')
log_reg_recall = recall_score(y_test_int, log_reg_preds, average='weighted')
log_reg_f1 = f1_score(y_test_int, log_reg_preds, average='weighted')

print(f'Logistic Regression Test Accuracy: {log_reg_accuracy:.4f}')
print(f'Logistic Regression Precision: {log_reg_precision:.4f}')
print(f'Logistic Regression Recall: {log_reg_recall:.4f}')
print(f'Logistic Regression F1 Score: {log_reg_f1:.4f}')


cm_log_reg = confusion_matrix(y_test_int, log_reg_preds)
disp_log_reg = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg, display_labels=["Benign", "Attack"])
disp_log_reg.plot(cmap=plt.cm.Greens)
plt.title("Logistic Regression Confusion Matrix")
plt.grid(False)
plt.show()


# Train and Evaluate Support Vector Machine (Linear SVC)
print("\nTraining and Evaluating Linear SVC...")
from sklearn.svm import LinearSVC


linear_svc_model = LinearSVC(random_state=42, max_iter=2000)

try:
    # Fit on the training data. y_train_int now has the correct number of samples.
    linear_svc_model.fit(X_train, y_train_int)


    linear_svc_preds = linear_svc_model.predict(X_test)
    linear_svc_accuracy = accuracy_score(y_test_int, linear_svc_preds)
    linear_svc_precision = precision_score(y_test_int, linear_svc_preds, average='weighted')
    linear_svc_recall = recall_score(y_test_int, linear_svc_preds, average='weighted')
    linear_svc_f1 = f1_score(y_test_int, linear_svc_preds, average='weighted')

    print(f'Linear SVC Test Accuracy: {linear_svc_accuracy:.4f}')
    print(f'Linear SVC Precision: {linear_svc_precision:.4f}')
    print(f'Linear SVC Recall: {linear_svc_recall:.4f}')
    print(f'Linear SVC F1 Score: {linear_svc_f1:.4f}')

    # Plot Linear SVC Confusion Matrix
    cm_linear_svc = confusion_matrix(y_test_int, linear_svc_preds)
    disp_linear_svc = ConfusionMatrixDisplay(confusion_matrix=cm_linear_svc, display_labels=["Benign", "Attack"])
    disp_linear_svc.plot(cmap=plt.cm.Purples)
    plt.title("Linear SVC Confusion Matrix")
    plt.grid(False)
    plt.show()

except Exception as e:
    print(f"Could not train/evaluate Linear SVC. Error: {e}")
    print("This might happen if the model fails to converge. Consider increasing max_iter or scaling features.")


# Train and Evaluate Gaussian Naive Bayes
print("\nTraining and Evaluating Gaussian Naive Bayes...")
from sklearn.naive_bayes import GaussianNB


gnb_model = GaussianNB()

gnb_model.fit(X_train, y_train_int)
gnb_preds = gnb_model.predict(X_test)
gnb_accuracy = accuracy_score(y_test_int, gnb_preds)
gnb_precision = precision_score(y_test_int, gnb_preds, average='weighted')
gnb_recall = recall_score(y_test_int, gnb_preds, average='weighted')
gnb_f1 = f1_score(y_test_int, gnb_preds, average='weighted')

print(f'Gaussian Naive Bayes Test Accuracy: {gnb_accuracy:.4f}')
print(f'Gaussian Naive Bayes Precision: {gnb_precision:.4f}')
print(f'Gaussian Naive Bayes Recall: {gnb_recall:.4f}')
print(f'Gaussian Naive Bayes F1 Score: {gnb_f1:.4f}')

# Plot Gaussian Naive Bayes Confusion Matrix
cm_gnb = confusion_matrix(y_test_int, gnb_preds)
disp_gnb = ConfusionMatrixDisplay(confusion_matrix=cm_gnb, display_labels=["Benign", "Attack"])
disp_gnb.plot(cmap=plt.cm.cividis)
plt.title("Gaussian Naive Bayes Confusion Matrix")
plt.grid(False)
plt.show()




# In[102]:


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


sgd_model = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42)

# Train the model
sgd_model.fit(X_train, y_train)


y_pred_sgd = sgd_model.predict(X_test)

# Print the evaluation metrics
print("SGDClassifier (Linear SVM) Model Performance:")
accuracy_sgd = accuracy_score(y_test, y_pred_sgd)
precision_sgd = precision_score(y_test, y_pred_sgd, average='weighted')
recall_sgd = recall_score(y_test, y_pred_sgd, average='weighted')
f1_sgd = f1_score(y_test, y_pred_sgd, average='weighted')

print(f'Accuracy: {accuracy_sgd:.4f}')
print(f'Precision: {precision_sgd:.4f}')
print(f'Recall: {recall_sgd:.4f}')
print(f'F1 Score: {f1_sgd:.4f}')


# In[103]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt



print("\nEvaluating SGDClassifier...")


y_pred_sgd_int = y_pred_sgd.astype(int)


# Calculate evaluation metrics
accuracy_sgd = accuracy_score(y_test_int, y_pred_sgd_int)
precision_sgd = precision_score(y_test_int, y_pred_sgd_int, average='weighted')
recall_sgd = recall_score(y_test_int, y_pred_sgd_int, average='weighted')
f1_sgd = f1_score(y_test_int, y_pred_sgd_int, average='weighted')

print(f'SGDClassifier Test Accuracy: {accuracy_sgd:.4f}')
print(f'SGDClassifier Precision: {precision_sgd:.4f}')
print(f'SGDClassifier Recall: {recall_sgd:.4f}')
print(f'SGDClassifier F1 Score: {f1_sgd:.4f}')


# Plot SGDClassifier Confusion Matrix
cm_sgd = confusion_matrix(y_test_int, y_pred_sgd_int)
disp_sgd = ConfusionMatrixDisplay(confusion_matrix=cm_sgd, display_labels=["Benign", "Attack"]) # Use your actual class labels
disp_sgd.plot(cmap=plt.cm.Blues) # Choose a different colormap if desired
plt.title("SGDClassifier Confusion Matrix")
plt.grid(False)
plt.show()


# In[106]:


spark.stop()

