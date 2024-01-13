import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

df = pd.read_csv('../CognitiveLoad-FeatureGen/out.csv')
plt.style.use('ggplot')
params={ 'objective':'multi:softprob',
     'max_depth': 9, 
     'n_estimators':300, # Modify this later
     'colsample_bylevel':0.5,
     'colsample_bytree':0.6,
     'learning_rate':0.05,
     'random_state':20,
     'alpha':10,
     'lambda':8}
xgb_cl = xgb.XGBClassifier(**params)

X = df.drop('Label', axis=1)
y = df['Label'].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1121218, test_size = 0.2
)
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

xgb_cl.fit(X_train, y_train)
print(xgb_cl)
plot_tree(xgb_cl)
preds = xgb_cl.predict(X_test)
print(accuracy_score(y_test, preds))
print(metrics.classification_report(y_test, preds, target_names=["relaxed", "neutral", "concentrated"]))
print(metrics.confusion_matrix(y_test, preds))

plt.show()

# save the model to disk
filename = 'xgb_model.sav'
pickle.dump(xgb_cl, open(filename, 'wb'))