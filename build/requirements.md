STEPS
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from IPython.display import clear_output
Model Evaluation
STEPS
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from IPython.display import clear_output
print(“Libraries Importing…”);
Model Evaluation
STEPS
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score,
roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from ipywidgets import interact, Dropdown
import time
Model Evaluation
STEPS
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score,
roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from ipywidgets import interact, Dropdown
import time
print(“Libraries Imported”);
Model Evaluation
STEPS data = pd.read_csv('Dengue_dataset.csv')
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS data = pd.read_csv('Dengue_dataset.csv’)
Print(“Dataset reading complete”)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS data.head()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS data.head()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
index Age Fever
Days
0
1
61
24
6
7
Platelets
56777
61250
Hematocrit
49.49572857
42.41093608
WBC
3205
3738
Headache
0
1
EyePain
1
0
Muscle
Pain
0
Joint
Pain
0
Rash
0
Nausea
1
Vomiting
1
Abdominal
Pain
0
Data Splitting
Model Training
2
3
4
70
30
33
7
6
88034
55130
49.66143051
50.20159282
3052
2713
1
1
0
1
0
1
0
3
64346
43.46980401
Bleeding Lethargy Dengue
0
0
0
0
4888
1
1
0
0
1
1
0
0
1
1
0
1
0
1
1
1
0
1
0
0
1
0
0
1
0
0
0
0
0
1
1
1
1
1
Model Evaluation
STEPS data.tail()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS data.tail()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
index Age Fever
Days
4995
4996
59
11
2
4
Platelets
123791
111575
Hematocrit
47.18861033
41.84560446
WBC
4239
5989
Headache
1
1
EyePain
0
0
Muscle
Pain
0
1
Joint
Pain
0
Rash
0
Nausea
0
Vomiting
0
Abdominal
Pain
1
Data Splitting
Model Training
4997
4998
4999
19
11
69
2
0
128198
141619
40.2842782
47.3939162
4775
4625
1
1
1
1
1
1
2
90619
46.47148497
Bleeding Lethargy Dengue
0
1
0
1
5982
1
0
0
0
0
0
0
1
1
0
1
0
0
1
1
0
0
1
0
0
0
0
0
0
0
0
1
1
0
0
0
0
0
0
Model Evaluation
STEPS data.describe()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS data.describe()
Importing Libraries
Reading Data
Data Analysis
Age
count
mean
5000
.0
41.8
55
FeverD
ays
5000.0
3.4114
Platelets
141522.48
44
Hematocrit
5000.0
5000.0
41.85748369
248
WBC
5000.
0
6156.
5842
Headache
5000.0
0.5172
Eye 
Pain
5000
.0
0.49
2
Muscle
Pain
5000.0
0.5076
Joint
Pain
5000.
0
0.5
Rash
5000.
0
0.496
2
Nausea
5000.0
0.5006
Vomiting
5000.0
0.5072
Abdominal
Pain
5000.0
Data Encoding
Data Splitting
Model Training
std
min
25%
50%
75%
max
2.0523
626576
449847
18.8
1193
1127
6262
13
10.0
26.0
42.0
58.0
0.0
2.0
3.0
5.0
87116.728
37019156
15060.0
62365.5
130972.0
4.813420018
022687
32.02024405 2002.
0
38.43400540
499999
3848.
75
0.499754
05033094
167
2594.
00605
99771
577
0.0
0.4999
922383
874443
0.49
9985
9970
0337
26
0.0
0.500
05000
75012
502
0.500
03556
58485
003
Bleeding Lethargy Dengue
5000.0
0.496
0.500049
64746510
64
0.499998
15962852
057
0.5000340
056449942
0.2258
0.418149
88923350
07
5000.0
0.5024
0.500044
24689199
3
5000.
0
0.46
0.498
44727
89954
8397
0.0
0.0
42.02366187 6169.
0
1.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
1.0
221014.25
74.0
7.0
299988.0
44.88610313 8302.
5
51.99352817 10993
1.0
1.0
1.0
1.0
1.0
1.0
0.5
1.0
1.0
0.0
1.0
1.0
1.0
1.0
1.0
1.0
1.0
1.0
0.0
1.0
1.0
0.0
0.0
1.0
1.0
1.0
1.0
0.0
1.0
1.0
Model Evaluation
.0
STEPS data.info()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS data.info()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS data.isnull().sum()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
data.isnull().sum()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS data.shape
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
data.shape
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
(5000, 16)
Model Evaluation
STEPS data[‘Dengue'].value_counts()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
data[‘Dengue'].value_counts()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
0    
1    
2700
2300
Name: Dengue, dtype: int64
Model Evaluation
STEPS data[‘Platelets'].value_counts()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
data[‘Platelets'].value_counts()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS data[‘WBC'].value_counts()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
data[‘WBC'].value_counts()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS data = data.replace({‘Dengue':{0:1, 1:0}})
data.head()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
data = data.replace({‘Dengue':{0:1, 1:0}})
data.head()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
index Age Fever
Days
0
1
61
24
6
7
Platelets
56777
61250
Hematocrit
49.49572857
42.41093608
WBC
3205
3738
Headache
0
1
EyePain
1
0
Muscle
Pain
0
Joint
Pain
0
Rash
0
Nausea
1
Vomiting
1
Abdominal
Pain
0
Data Splitting
Model Training
2
3
4
70
30
33
7
6
88034
55130
49.66143051
50.20159282
3052
2713
1
1
0
1
0
1
0
3
64346
43.46980401
Bleeding Lethargy Dengue
0
0
0
0
4888
1
1
0
0
1
1
0
0
1
1
0
1
0
1
1
1
0
1
0
0
1
0
0
1
0
0
0
0
0
0
0
0
0
0
Model Evaluation
STEPS data = data.replace({‘Headache':{0:1, 1:0}})
data.head()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
data = data.replace({‘Headache':{0:1, 1:0}})
data.head()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
index Age Fever
Days
0
1
61
24
6
7
Platelets
56777
61250
Hematocrit
49.49572857
42.41093608
WBC
3205
3738
Headache
1
0
EyePain
1
0
Muscle
Pain
Joint
Pain
0
0
Rash
0
Nausea
1
Vomiting
1
Abdominal
Pain
0
Data Splitting
Model Training
2
3
4
70
30
33
7
6
88034
55130
49.66143051
50.20159282
3052
2713
0
0
0
1
0
1
0
3
64346
43.46980401
Bleeding Lethargy Dengue
0
0
0
0
4888
0
1
0
0
1
1
0
0
1
1
0
1
0
1
1
1
0
1
0
0
1
0
0
1
0
0
0
0
0
0
0
0
0
0
Model Evaluation
STEPS data = data.replace({‘Rash':{0:1, 1:0}})
data.head()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
data = data.replace({‘Rash':{0:1, 1:0}})
data.head()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
index Age Fever
Days
0
1
61
24
6
7
Platelets
56777
61250
Hematocrit
49.49572857
42.41093608
WBC
3205
3738
Headache
1
0
EyePain
1
0
Muscle
Pain
0
Joint
Pain
0
Rash
1
Nausea
1
Vomiting
1
Abdominal
Pain
0
Data Splitting
Model Training
2
3
4
70
30
33
7
6
88034
55130
49.66143051
50.20159282
3052
2713
0
0
0
1
0
0
0
3
64346
43.46980401
Bleeding Lethargy Dengue
0
0
0
0
4888
0
1
0
0
1
1
1
1
0
1
0
1
0
1
1
1
0
1
0
0
1
0
0
1
0
0
0
0
0
0
0
0
0
0
Model Evaluation
STEPS
target_col = "Dengue"
print("Target column set to:", target_col)
X = data.drop(columns=[target_col])
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42, stratify=y
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
print(f" Split: 80% Train ({len(X_train)}) / 20% Test ({len(X_test)}) samples")
Model Evaluation
STEPS
target_col = "Dengue"
print("Target column set to:", target_col)
X = data.drop(columns=[target_col])
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42, stratify=y
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
print(f" Split: 80% Train ({len(X_train)}) / 20% Test ({len(X_test)}) samples")
Target column set to: Dengue 
Split: 80% Train (4000) / 20% Test (1000) samples 
Model Evaluation
STEPS
X
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
X
Importing Libraries
Reading Data
Data Analysis
Data Encoding
index
Age
0
1
61
24
FeverD
ays
6
7
Hematocrit
49.49572857
42.41093608
WBC
3205
3738
Headache
0
1
EyePain
1
0
MuscleP
ain
0
0
JointP
ain
0
0
Rash
0
1
Nausea
1
Vomiting
1
AbdominalPa
in
Data Splitting
Model Training
2
3
4
70
30
33
7
6
3
49.66143051
50.20159282
3052
2713
1
0
Bleeding
Lethargy
0
0
1
1
43.46980401
4888
1
0
1
0
0
0
1
1
0
0
1
1
0
1
0
0
1
1
1
0
1
0
0
1
0
0
1
0
0
0
0
0
Model Evaluation
STEPS
Y.head()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
Y.head()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS model = LogisticRegression(
C=1.0,       
penalty='l2',   
Importing Libraries
# regularization strength
# L2 regularization
solver='liblinear'
)
model.fit(X_train, Y_train)
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# MODEL TRAINING
model = LogisticRegression(
C=1.0,       
# regularization strength
penalty='l2',   
Importing Libraries
# L2 regularization
solver='liblinear'
)
model.fit(X_train, Y_train)
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS # Preparing Feature and Target Data for Logistic Regression Analysis
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
feature_names = numeric_cols
X_num = X[feature_names]
y_arr = np.array(y)
print(f"Feature-target data ready → {len(feature_names)} numeric features selected for modeling")
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Preparing Feature and Target Data for Logistic Regression Analysis
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
feature_names = numeric_cols
X_num = X[feature_names]
y_arr = np.array(y)
print(f"Feature-target data ready → {len(feature_names)} numeric features selected for modeling")
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
def show_1d_sigmoid(f):
X,y = X_num[[f]].values, y_arr; c = LogisticRegression(max_iter=5000).fit(X,y)
a,b = X.min(),X.max(); 
Importing Libraries
a,b = (a-1,b+1) if a==b else (a,b); g= np.linspace(a,b,300)[:,None]; p = c.predict_proba(g)[:,1]
plt.scatter(X[y==0],y[y==0]); plt.scatter(X[y==1],y[y==1]); plt.plot(g,p); plt.xlabel(f); plt.ylabel("Probability 
of Dengue");
plt.ylim(-.1,1.1); plt.grid(1); plt.title(f"Sigmoid using {f}");
plt.show()
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
def show_1d_sigmoid(f):
X,y = X_num[[f]].values, y_arr; c = LogisticRegression(max_iter=5000).fit(X,y)
a,b = X.min(),X.max(); 
Importing Libraries
a,b = (a-1,b+1) if a==b else (a,b); g= np.linspace(a,b,300)[:,None]; p = c.predict_proba(g)[:,1]
plt.scatter(X[y==0],y[y==0]); plt.scatter(X[y==1],y[y==1]); plt.plot(g,p); plt.xlabel(f); plt.ylabel("Probability 
of Dengue");
plt.ylim(-.1,1.1); plt.grid(1); plt.title(f"Sigmoid using {f}");
plt.show()
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Interactive dropdown over all features
interact(
show_1d_sigmoid,
feat=Dropdown(options=feature_names, description="Feature")
)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Training Performance
Y_train_pred = model.predict(X_train)
print(" Training Performance ")
print(f"Accuracy : {accuracy_score(y_train, Y_train_pred):.4f}")
print(f"Precision: {precision_score(y_train, Y_train_pred):.4f}")
print(f"Recall   
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
: {recall_score(y_train, Y_train_pred):.4f}")
print(f"F1 Score : {f1_score(y_train, Y_train_pred):.4f}")
Model Evaluation
STEPS
# Training Performance
Y_train_pred = model.predict(X_train)
print(" Training Performance ")
print(f"Accuracy : {accuracy_score(y_train, Y_train_pred):.4f}")
print(f"Precision: {precision_score(y_train, Y_train_pred):.4f}")
print(f"Recall   
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
: {recall_score(y_train, Y_train_pred):.4f}")
print(f"F1 Score : {f1_score(y_train, Y_train_pred):.4f}")
Accuracy : 0.9732
Precision: 0.9712 
Recall : 0.9707 
F1 Score : 0.9709 
Model Evaluation
STEPS
# Testing Performance
Y_test_pred = model.predict(X_test)
print("Testing Performance")
print(f"Accuracy : {accuracy_score(y_test, Y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, Y_test_pred):.4f}")
print(f"Recall   
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
: {recall_score(y_test, Y_test_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, Y_test_pred):.4f}")
Model Evaluation
STEPS
# Testing Performance
Y_test_pred = model.predict(X_test)
print("Testing Performance")
print(f"Accuracy : {accuracy_score(y_test, Y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, Y_test_pred):.4f}")
print(f"Recall   
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
: {recall_score(y_test, Y_test_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, Y_test_pred):.4f}")
Accuracy : 0.9740 
Precision: 0.9657 
Recall : 0.9783 
F1 Score : 0.9719
Model Evaluation
STEPS
# Confusion Matrix
cm = confusion_matrix(Y_test, Y_test_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
# Confusion Matrix
cm = confusion_matrix(Y_test, Y_test_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS Y_proba = model.predict_proba(X_test)[:,1]
fpr, tpr, th = roc_curve(Y_test, Y_proba)
roc_auc = auc(fpr, tpr)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
Y_proba = model.predict_proba(X_test)[:,1]
fpr, tpr, th = roc_curve(Y_test, Y_proba)
roc_auc = auc(fpr, tpr)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
plt.plot(fpr, tpr, linewidth=3, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
STEPS
plt.plot(fpr, tpr, linewidth=3, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
# Random Test Prediction
sample = X_test.sample(1)
actual = Y_test.loc[sample.index].values[0]
pred = model.predict(sample)[0]
prob = model.predict_proba(sample)[0][1]
STEPS
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
Model Evaluation
Inde
x
Ag
e
Feve
rDay
s
Platelets Hematocrit WBC Headac
he
EyePai
n
Muscl
ePain
Joint
Pain Rash Nausea Vomitin
g
Abdomin
alPain
Bleedin
g
Letharg
y
Deng
ue
1434 69 3 32794 44.363337 4203 1 1 1 0 0 0 1 1 0 0 1
1435 23 6 34881 42.601771 2706 1 1 0 1 0 0 1 0 1 1 1
1436 11 7 64703 48.035335 3961 0 0 1 0 1 0 0 1 0 1 1
1437 26 4 59354 51.648016 4519 0 1 0 1 0 1 0 0 1 0 1
1438 37 3 29390 42.108583 4666 1 1 0 0 0 1 0 0 1 1 1
STEPS
# Random Test Prediction
sample = X_test.sample(1)
Importing Libraries
Reading Data
Data Analysis
Data Encoding
actual = Y_test.loc[sample.index].values[0]
pred = model.predict(sample)[0]
prob = model.predict_proba(sample)[0][1]
Inde
x
Ag
e
Feve
rDay
s
1434 69 3
1435 23 6
Platelets Hematocrit WBC Headac
32794
34881
44.363337
42.601771
he
4203 1
2706 1
EyePai
n
1
1
Muscl
ePain
1
0
Joint
Pain
0
1
Rash Nausea Vomitin
g
0
0
Abdomin
alPain
Data Splitting
Model Training
1436 11 7
1437 26 4
1438 37 3
64703
59354
48.035335
51.648016
3961 0
4519 0
0
29390
Bleedin
g
1
0
1
1
42.108583
4666 1
1
0
0
0
1
0
1
0
0
0
0
1
1
1
0
0
0
1
0
1
0
0
0
1
0
1
1
Letharg
y
0
1
1
0
1
Deng
ue
1
1
1
1
1
Model Evaluation
STEPS
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
print("------ Random Sample Test ------")
print(sample)
print("Actual Dengue:", actual)
print("Predicted Dengue:", pred)
print("Probability:", prob)
Model Evaluation
STEPS
Importing Libraries
Reading Data
Data Analysis
Data Encoding
Data Splitting
Model Training
print("------ Random Sample Test ------")
print(sample)
print("Actual Dengue:", actual)
print("Predicted Dengue:", pred)
print("Probability:", prob)
Model Evaluatio