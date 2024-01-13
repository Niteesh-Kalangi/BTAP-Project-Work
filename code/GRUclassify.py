import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report
import pickle

print("hello")
data = pd.read_csv(
    '../CognitiveLoad-FeatureGen/out.csv')

#testdata = pd.read_csv('../CognitiveLoad-FeatureGen/test.csv')

sample = data.loc[0, 'freq_010_0':'freq_750_3']

plt.figure(figsize=(16, 10))
plt.plot(range(len(sample)), sample)
plt.title("Features freq_010_0 through freq_750_3")
plt.show()


#  'concentrating':  state = 0. //NEGATIVE
#   netrual :        state = 1 //NEUTRAL
#  'relaxed':       state = 2  //POSITIVE
print(data['Label'].value_counts())


def preprocess_inputs(df):
    df = df.copy()

    y = df['Label'].copy()
    X = df.drop('Label', axis=1).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=123)

    return X_train, X_test, y_train, y_test



X_train, X_test, y_train, y_test = preprocess_inputs(data)
print(X_train)
inputs = tf.keras.Input(shape=(X_train.shape[1],))

expand_dims = tf.expand_dims(inputs, axis=2)

gru = tf.keras.layers.GRU(256, return_sequences=True)(expand_dims)

flatten = tf.keras.layers.Flatten()(gru)

outputs = tf.keras.layers.Dense(3, activation='softmax')(flatten)


model = tf.keras.Model(inputs=inputs, outputs=outputs)
#print(model.summary())


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=5,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print("Test Accuracy: {:.3f}%".format(model_acc * 100))



#test = testdata['Label'].copy()

y_pred = np.array(list(map(lambda x: np.argmax(x), model.predict(X_test))))

label_mapping = {'STRESSED': 0, 'NEUTRAL': 1, 'RELAXED': 2}

cm = confusion_matrix(y_test, y_pred )
clr = classification_report(y_test, y_pred , target_names=label_mapping.keys())

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n----------------------\n", clr)


# save the model to disk
filename = 'GRU_model.sav'
pickle.dump(model, open(filename, 'wb'))