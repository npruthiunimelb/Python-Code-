#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# In[37]:


prob_threshold = 0.5


# In[38]:


import os


# In[39]:


os.listdir()


# In[40]:


# Load datasets
df_oscc_30 = pd.read_excel('Train NN.xlsx').fillna(0)
df_opmd = pd.read_excel('Test NN 1.xlsx').fillna(0)


# In[41]:


# Prepare the dataset
def prepare_data(data):
    X = data.drop(columns='risk score')
    y = data['risk score']
    return X, y


# In[42]:


df_oscc_30.columns


# In[43]:


df_oscc_30.columns


# In[44]:


df_opmd.columns


# In[45]:


df_oscc_30.head(1)


# In[46]:


# Get features and labels from the 30 OSCC datasheet
X = df_oscc_30.drop("risk score",axis=1)
y = df_oscc_30["risk score"]

# Standardize the features
#scaler = StandardScaler()
#X = scaler.fit_transform(X)

# Standardize the other two datasets as well
#X_opmd = scaler.transform(df_opmd.drop(columns=["risk score"]))
#y_opmd = df_opmd["risk score"]

#X_oscc_51 = scaler.transform(df_oscc_51.drop(columns=["risk score"]))
#y_oscc_51 = df_oscc_51["risk score"]


# In[47]:


import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# In[ ]:





# In[48]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    plot_confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

# Generate a random dataset for demonstration
#X, y = make_classification(n_samples=1000, n_features=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and fit the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Calculate ROC AUC
y_prob = logistic_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print(f'ROC AUC: {roc_auc:.2f}')

# Print coefficients and intercept
coefficients = logistic_model.coef_[0]
intercept = logistic_model.intercept_[0]
print(f'Coefficients: {coefficients}')
print(f'Intercept: {intercept}')

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')

# Calculate and plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.subplot(1, 2, 2)
plot_confusion_matrix(logistic_model, X_test, y_test, cmap=plt.cm.Blues, display_labels=["Class 0", "Class 1"])
plt.title('Confusion Matrix')

plt.tight_layout()
plt.show()

print("Confusion Matrix:")
print(conf_matrix)



# In[49]:


# Define models
models = {
    "Logistic Regression": LogisticRegression(solver="liblinear", max_iter=10000),
    "Lasso": LogisticRegression(solver="liblinear", penalty="l1", max_iter=10000)
}

# Repeated stratified K-Fold cross-validator with 10000 iterations
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)

for name, model in models.items():
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print(f"{name}: ROC AUC = {np.mean(scores)}")
    

    


# # Model 1 70/30 100 times

# In[50]:


models = {
    "Logistic Regression": LogisticRegression(solver="liblinear", max_iter=10000),
    "Lasso": LogisticRegression(solver="liblinear", penalty="l1", max_iter=10000)
}

# Prepare to store the scores
accuracy_scores = {key: [] for key in models.keys()}
roc_auc_scores = {key: [] for key in models.keys()}
precision_scores = {key: [] for key in models.keys()}
recall_scores = {key: [] for key in models.keys()}

coef_storage = {key:[] for key in models}
intercept_storage = {key:[] for key in models}

def precision_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[1,1] / (cm[1,1] + cm[0,1])

def recall_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[1,1] / (cm[1,1] + cm[1,0])

# Split the data into 70/30 %, randomly shuffle the data, and build the models 10000 times
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        coef_storage[name].append(model.coef_[0])
        intercept_storage[name].append(model.intercept_)
        accuracy = model.score(X_test, y_test)
        #y_pred = model.predict(X_test)
        y_pred = model.predict_proba(X_test)
        y_pred = (y_pred[:,1] >= prob_threshold).astype(int)
        roc_auc = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision_scores[name].append(precision)
        recall_scores[name].append(recall)
        


        accuracy_scores[name].append(accuracy)
        roc_auc_scores[name].append(roc_auc)


for name in models.keys():
    coef_storage[name] = np.array(coef_storage[name])
    intercept_storage[name] = np.array(intercept_storage[name])


# Report average accuracy and ROC AUC
for name in models.keys():
    avg_accuracy = np.mean(accuracy_scores[name])
    avg_roc_auc = np.mean(roc_auc_scores[name])
    print(f"{name}: Avg. Accuracy = {avg_accuracy:.3f}, Avg. ROC AUC = {avg_roc_auc:.3f}")


avg_coefs = {key:np.mean(coef_storage[key],axis=0) for key in models.keys()}
avg_intercepts = {key:np.mean(intercept_storage[key],axis=0) for key in models.keys()}

for name, avg_coef in avg_coefs.items():
    print(f"{name} - Average Coefficients")
    print(avg_coef)
    print(f"{name} - Average Intercepts {avg_intercepts[name]}")
    print()
    
precision_scores = {key:np.mean(precision_scores[key]) for key in models.keys()}
recall_scores = {key:np.mean(recall_scores[key]) for key in models.keys()}

for name in models.keys():
    print(f"{name} - Precision = {precision_scores[name]:.3f}, Recall = {recall_scores[name]:.3f}")


# In[51]:


precision_scores = {key:np.mean(precision_scores[key]) for key in models.keys()}
recall_scores = {key:np.mean(recall_scores[key]) for key in models.keys()}

for name in models.keys():
    print(f"{name} - Precision = {precision_scores[name]:.3f}, Recall = {recall_scores[name]:.3f}")
    


# In[52]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

# Set larger font size for better readability
sns.set(font_scale=1.5)

# Histogram of metrics
for name in models.keys():
    plt.figure(figsize=(12, 8))
    sns.distplot(accuracy_scores[name], label="Accuracy")
    sns.distplot(roc_auc_scores[name], label="ROC AUC")
    plt.title(f"Distribution of metrics for {name}")
    plt.legend()
    
    # Turn off the top and right spines (borders) to remove grid lines
    sns.despine()
    
    plt.show()

# Boxplots
metrics = [accuracy_scores, roc_auc_scores, precision_scores, recall_scores]
metric_names = ["Accuracy", "ROC AUC", "Precision", "Recall"]

for metric, metric_name in zip(metrics, metric_names):
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=[metric[name] for name in models.keys()])
    plt.xticks(list(range(len(models.keys()))), models.keys())
    plt.title(f"Boxplot of {metric_name}")
    
    # Turn off the top and right spines (borders) to remove grid lines
    sns.despine()
    
    plt.show()

# ROC Curves overlay
for name, model in models.items():
    plt.figure(figsize=(12, 8))
    
    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        
        # To prevent too many lines from being plotted and making it unreadable,
        # you can plot a subset or average them.
        if i % 200 == 0:  # Plot every 200th ROC curve
            plt.plot(fpr, tpr, alpha=0.1, color='blue')
    
    plt.title(f"ROC Curves for {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
    # Turn off the top and right spines (borders) to remove grid lines
    sns.despine()
    
    plt.show()


# In[53]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# Histogram of metrics
#This code combines the distribution plots for both accuracy and ROC AUC scores in a single set 
#of plots for each model. It uses sns.distplot to overlay the histograms of both accuracy and ROC AUC scores
#in the same plot for each model. 

sns.set_style("white")

#This results in a single set of plots for each model that shows the distribution of both metrics together.
# Set a larger font size for better readability
sns.set(font_scale=1.5)
for name in models.keys():
    plt.figure(figsize=(12, 8))
    sns.distplot(accuracy_scores[name], label="Accuracy")
    sns.distplot(roc_auc_scores[name], label="ROC AUC")
    plt.title(f"Distribution of metrics for {name}")
    plt.legend()
    plt.show()

# Boxplots
metrics = [accuracy_scores, roc_auc_scores, precision_scores, recall_scores]
metric_names = ["Accuracy", "ROC AUC", "Precision", "Recall"]

for metric, metric_name in zip(metrics, metric_names):
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=[metric[name] for name in models.keys()])
    plt.xticks(list(range(len(models.keys()))), models.keys())
    plt.title(f"Boxplot of {metric_name}")
    plt.show()

# ROC Curves overlay
for name, model in models.items():
    plt.figure(figsize=(12, 8))
    
    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        
        # To prevent too many lines from being plotted and making it unreadable,
        # you can plot a subset or average them.
        if i % 200 == 0:  # Plot every 200th ROC curve
            plt.plot(fpr, tpr, alpha=0.1, color='blue')
        


            
    
    plt.title(f"ROC Curves for {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


# In[54]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# Histogram of metrics
#This code combines the distribution plots for both accuracy and ROC AUC scores in a single set 
#of plots for each model. It uses sns.distplot to overlay the histograms of both accuracy and ROC AUC scores
#in the same plot for each model. 
#This results in a single set of plots for each model that shows the distribution of both metrics together.
sns.set(font_scale=1.5)
for name in models.keys():
    plt.figure(figsize=(10, 6))
    sns.distplot(accuracy_scores[name], label="Accuracy")
    sns.distplot(roc_auc_scores[name], label="ROC AUC")
    plt.title(f"Distribution of metrics for {name}")
    plt.legend()
    plt.show()

# Boxplots
metrics = [accuracy_scores, roc_auc_scores, precision_scores, recall_scores]
metric_names = ["Accuracy", "ROC AUC", "Precision", "Recall"]

for metric, metric_name in zip(metrics, metric_names):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[metric[name] for name in models.keys()])
    plt.xticks(list(range(len(models.keys()))), models.keys())
    plt.title(f"Boxplot of {metric_name}")
    plt.show()

# ROC Curves overlay
for name, model in models.items():
    plt.figure(figsize=(10, 6))
    
    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        
        # To prevent too many lines from being plotted and making it unreadable,
        # you can plot a subset or average them.
        if i % 200 == 0:  # Plot every 200th ROC curve
            plt.plot(fpr, tpr, alpha=0.1, color='blue')
        


            
    
    plt.title(f"ROC Curves for {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


# In[55]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

# Set a larger font size for better readability


# Histogram of metrics
for name in models.keys():
    plt.figure(figsize=(10, 6))
    sns.distplot(accuracy_scores[name], label="Accuracy")
    sns.distplot(roc_auc_scores[name], label="ROC AUC")
    plt.title(f"Distribution of metrics for {name}")
    plt.xlabel("Metric Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# Boxplots
metrics = [accuracy_scores, roc_auc_scores, precision_scores, recall_scores]
metric_names = ["Accuracy", "ROC AUC", "Precision", "Recall"]

for metric, metric_name in zip(metrics, metric_names):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[metric[name] for name in models.keys()])
    plt.xticks(list(range(len(models.keys()))), models.keys())
    plt.title(f"Boxplot of {metric_name}")
    plt.xlabel("Models")
    plt.ylabel(metric_name)
    plt.show()

# ROC Curves overlay
for name, model in models.items():
    plt.figure(figsize=(10, 6))
    
    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        
        if i % 200 == 0:  # Plot every 200th ROC curve
            plt.plot(fpr, tpr, alpha=0.1, color='blue')
            
    plt.title(f"ROC Curves for {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


# In[56]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

# Set larger font sizes for better readability
sns.set(font_scale=1.5)
plt.rcParams.update({'axes.labelsize': 16, 'axes.titlesize': 18})

# Histogram of metrics
for name in models.keys():
    plt.figure(figsize=(12, 8))
    sns.distplot(accuracy_scores[name], label="Accuracy")
    sns.distplot(roc_auc_scores[name], label="ROC AUC")
    plt.title(f"Distribution of metrics for {name}")
    plt.xlabel("Metric Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# Boxplots
metrics = [accuracy_scores, roc_auc_scores, precision_scores, recall_scores]
metric_names = ["Accuracy", "ROC AUC", "Precision", "Recall"]

for metric, metric_name in zip(metrics, metric_names):
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=[metric[name] for name in models.keys()])
    plt.xticks(list(range(len(models.keys()))), models.keys())
    plt.title(f"Boxplot of {metric_name}")
    plt.xlabel("Models")
    plt.ylabel(metric_name)
    plt.show()

# ROC Curves overlay
for name, model in models.items():
    plt.figure(figsize=(12, 8))
    
    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        
        if i % 200 == 0:  # Plot every 200th ROC curve
            plt.plot(fpr, tpr, alpha=0.1, color='blue')
            
    plt.title(f"ROC Curves for {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


# In[57]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

# Set larger font sizes for better readability
sns.set(font_scale=1.5)
plt.rcParams.update({'axes.labelsize': 16, 'axes.titlesize': 18})

# Histogram of metrics
for name in models.keys():
    plt.figure(figsize=(10, 6))
    sns.distplot(accuracy_scores[name], label="Accuracy")
    sns.distplot(roc_auc_scores[name], label="ROC AUC")
    plt.title(f"Distribution of metrics for {name}")
    plt.xlabel("Metric Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# Boxplots
metrics = [accuracy_scores, roc_auc_scores, precision_scores, recall_scores]
metric_names = ["Accuracy", "ROC AUC", "Precision", "Recall"]

for metric, metric_name in zip(metrics, metric_names):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[metric[name] for name in models.keys()])
    plt.xticks(list(range(len(models.keys()))), models.keys())
    plt.title(f"Boxplot of {metric_name}")
    plt.xlabel("Models")
    plt.ylabel(metric_name)
    plt.show()

# ROC Curves overlay
for name, model in models.items():
    plt.figure(figsize=(10, 6))
    
    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        
        if i % 200 == 0:  # Plot every 200th ROC curve
            plt.plot(fpr, tpr, alpha=0.1, color='blue')
            
    plt.title(f"ROC Curves for {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


# In[58]:


for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    
    for name, model in models.items():
        model.fit(X_train, y_train)

        if i % 200 == 0:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=["Predicted 0", "Predicted 1"],
                        yticklabels=["Actual 0", "Actual 1"])
            plt.title(f"Confusion Matrix: {name} (Iteration {i})")
            plt.show()


# In[59]:


from sklearn.metrics import precision_recall_curve


# In[60]:


for name, model in models.items():
    # Compute predictions and predicted probabilities for the test set
    y_pred_proba = model.predict_proba(X_test)[:,1]
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
plt.show()


# In[61]:


print(precision_scores["Lasso"])


# In[62]:


coef_storage['Logistic Regression']


# In[63]:


len(coef_storage['Logistic Regression'])


# In[64]:


# intercept_storage['Logistic Regression']


# In[65]:


coef_storage['Lasso']


# In[66]:


len(coef_storage['Lasso'])


# In[67]:


# intercept_storage['Lasso']


# In[68]:


# roc_auc_scores['Logistic Regression']


# In[69]:


# roc_auc_scores['Lasso']


# In[70]:


#ROC AND AUC Graphs seperately 


# In[71]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# Distribution of ROC AUC Scores
for name in models.keys():
    plt.figure(figsize=(10, 6))
    sns.distplot(roc_auc_scores[name], label="ROC AUC")
    plt.title(f"Distribution of ROC AUC scores for {name}")
    plt.legend()
    plt.show()

# Distribution of Accuracy Scores
for name in models.keys():
    plt.figure(figsize=(10, 6))
    sns.distplot(accuracy_scores[name], label="Accuracy")
    plt.title(f"Distribution of Accuracy scores for {name}")
    plt.legend()
    plt.show()


# # Hyper Parameter Tuning

# In[72]:


import numpy as np
np.random.seed(0)


# In[73]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# In[74]:


logistic_regression_params = {
    "C": np.logspace(-4, 4, 20),
    "solver": ["liblinear"],
}

lasso_params = {
    "C": np.logspace(-4, 4, 20),
    "solver": ["liblinear"],
    "penalty": ["l1"],
}


# In[75]:


import numpy as np

# Set global random seed
np.random.seed(0)

def tune_hyperparameters(model, params, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_

models = {
    "Logistic Regression": (LogisticRegression(max_iter=100000, random_state=0), logistic_regression_params),
    "Lasso": (LogisticRegression(max_iter=100000, random_state=0), lasso_params),
}

best_params = {}

for model_name, (model, params) in models.items():
    best_param, best_score = tune_hyperparameters(model, params, X_train, y_train)
    best_params[model_name] = (best_param, best_score)
    print(f"{model_name}: Best Params: {best_param}, Best Score: {best_score}")


# In[76]:


best_params_models = {}

for model_name, (params, score) in best_params.items():
    if model_name == "Logistic Regression":
        best_params_models[model_name] = LogisticRegression(**params)
    elif model_name == "Lasso":
        best_params_models[model_name] = LogisticRegression(**params)

print(best_params_models)


# In[77]:


print(best_params_models)


# # 70/30 Hyper param model

# # Model 2

# In[78]:


models = {
    "Logistic Regression": LogisticRegression(solver="liblinear", max_iter=1, C=545.5594781168514),
    "Lasso": LogisticRegression(solver="liblinear", penalty="l1", max_iter=1, C=11.288378916846883)
}

# Prepare to store the scores
accuracy_scores = {key: [] for key in models.keys()}
roc_auc_scores = {key: [] for key in models.keys()}
precision_scores = {key: [] for key in models.keys()}
recall_scores = {key: [] for key in models.keys()}

coef_storage = {key:[] for key in models}
intercept_storage = {key:[] for key in models}

def precision_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[1,1] / (cm[1,1] + cm[0,1])

def recall_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[1,1] / (cm[1,1] + cm[1,0])

# Split the data into 70/30 %, randomly shuffle the data, and build the models 10000 times
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        coef_storage[name].append(model.coef_[0])
        intercept_storage[name].append(model.intercept_)
        accuracy = model.score(X_test, y_test)
        #y_pred = model.predict(X_test)
        y_pred = model.predict_proba(X_test)
        y_pred = (y_pred[:,1] >= prob_threshold).astype(int)
        roc_auc = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision_scores[name].append(precision)
        recall_scores[name].append(recall)
        


        accuracy_scores[name].append(accuracy)
        roc_auc_scores[name].append(roc_auc)


for name in models.keys():
    coef_storage[name] = np.array(coef_storage[name])
    intercept_storage[name] = np.array(intercept_storage[name])


# Report average accuracy and ROC AUC
for name in models.keys():
    avg_accuracy = np.mean(accuracy_scores[name])
    avg_roc_auc = np.mean(roc_auc_scores[name])
    print(f"{name}: Avg. Accuracy = {avg_accuracy:.3f}, Avg. ROC AUC = {avg_roc_auc:.3f}")


avg_coefs = {key:np.mean(coef_storage[key],axis=0) for key in models.keys()}
avg_intercepts = {key:np.mean(intercept_storage[key],axis=0) for key in models.keys()}

for name, avg_coef in avg_coefs.items():
    print(f"{name} - Average Coefficients")
    print(avg_coef)
    print(f"{name} - Average Intercepts {avg_intercepts[name]}")
    print()
    
precision_scores = {key:np.mean(precision_scores[key]) for key in models.keys()}
recall_scores = {key:np.mean(recall_scores[key]) for key in models.keys()}

for name in models.keys():
    print(f"{name} - Precision = {precision_scores[name]:.3f}, Recall = {recall_scores[name]:.3f}")


# In[79]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# Histogram of metrics
for name in models.keys():
    plt.figure(figsize=(10, 6))
    sns.distplot(accuracy_scores[name], label="Accuracy")
    sns.distplot(roc_auc_scores[name], label="ROC AUC")
    plt.title(f"Distribution of metrics for {name}")
    plt.legend()
    plt.show()

# Boxplots
metrics = [accuracy_scores, roc_auc_scores, precision_scores, recall_scores]
metric_names = ["Accuracy", "ROC AUC", "Precision", "Recall"]

for metric, metric_name in zip(metrics, metric_names):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[metric[name] for name in models.keys()])
    plt.xticks(list(range(len(models.keys()))), models.keys())
    plt.title(f"Boxplot of {metric_name}")
    plt.show()

# ROC Curves overlay
for name, model in models.items():
    plt.figure(figsize=(10, 6))
    
    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        
        # To prevent too many lines from being plotted and making it unreadable,
        # you can plot a subset or average them.
        if i % 250 == 0:  # Plot every 250th ROC curve
            plt.plot(fpr, tpr, alpha=0.1, color='blue')
    
    plt.title(f"ROC Curves for {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


# In[80]:


for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    
    for name, model in best_params_models.items():
        model.fit(X_train, y_train)

        if i % 200 == 0:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=["Predicted 0", "Predicted 1"],
                        yticklabels=["Actual 0", "Actual 1"])
            plt.title(f"Confusion Matrix: {name} (Iteration {i})")
            plt.show()


# In[ ]:





# In[ ]:





# In[81]:


for name in models.keys():
    # Distribution of Accuracy Scores
    plt.figure(figsize=(10, 6))
    sns.distplot(accuracy_scores[name], label="Accuracy")
    plt.title(f"Distribution of Accuracy scores for {name}")
    plt.legend()
    plt.show()

    # Distribution of ROC AUC Scores
    plt.figure(figsize=(10, 6))
    sns.distplot(roc_auc_scores[name], label="ROC AUC")
    plt.title(f"Distribution of ROC AUC scores for {name}")
    plt.legend()
    plt.show()


# In[82]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    plot_confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

# Generate a random dataset for demonstration
#X, y = make_classification(n_samples=1000, n_features=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, C=545.5594781168514)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and fit the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Calculate ROC AUC
y_prob = logistic_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print(f'ROC AUC: {roc_auc:.2f}')

# Print coefficients and intercept
coefficients = logistic_model.coef_[0]
intercept = logistic_model.intercept_[0]
print(f'Coefficients: {coefficients}')
print(f'Intercept: {intercept}')

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')

# Calculate and plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.subplot(1, 2, 2)
plot_confusion_matrix(logistic_model, X_test, y_test, cmap=plt.cm.Blues, display_labels=["Class 0", "Class 1"])
plt.title('Confusion Matrix')

plt.tight_layout()
plt.show()

print("Confusion Matrix:")
print(conf_matrix)


# # Model 3: Build a model using all the data and use it for predictions

# In[ ]:





# In[ ]:


#MODEL3 #model bulid on 60 oscc dataset and tested on opmd with best parameters


# In[ ]:


# df_opmd = pd.read_excel('Test NN 1.xlsx').fillna(0)

X_opmd = df_opmd.drop("risk score",axis=1)
y_opmd = df_opmd["risk score"]


# In[ ]:


df_opmd.columns


# In[ ]:


df_oscc_30.columns


# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[ ]:


# prob_threshold =0.7


# In[ ]:


from sklearn.metrics import roc_curve, precision_recall_curve, auc


# In[ ]:


#model in best parameters 
for name, model in best_params_models.items():
    # Train the model on the whole 30 OSCC dataset

    # min max scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_opmd = scaler.transform(X_opmd)
    #X_oscc_51 = scaler.transform(X_oscc_51)


    model.fit(X, y)
    print(model.get_params())
    
    # Predict on the OPMD dataset
    # y_pred_opmd = model.predict(X_opmd)
    y_pred_proba_opmd = model.predict_proba(X_opmd)
    #print(y_pred_proba_opmd)
    y_pred_opmd = (y_pred_proba_opmd[:,1] >= prob_threshold).astype(int)
    #print(y_pred_opmd)
    cm_opmd = confusion_matrix(y_opmd, y_pred_opmd)
    roc_auc_opmd = roc_auc_score(y_opmd, y_pred_opmd)
    report_opmd = classification_report(y_opmd, y_pred_opmd)

    print(f"Model: {name}\n")
    print("OPMD dataset:")
    print("Confusion Matrix:")
    print(cm_opmd)
    print(f"ROC AUC Score: {roc_auc_opmd}")
    print("Classification Report:")
    print(report_opmd)
    print("\n51 OSCC dataset:")
    print("Confusion Matrix:")
    print("\n" + "=" * 80 + "\n")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_opmd, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"])
    plt.title(f"Confusion Matrix: {name}")
    plt.show()
    
    # ROC-AUC Curve for OPMD dataset
    y_pred_proba_opmd_1 = y_pred_proba_opmd[:, 1]  # Probabilities for class 1
    fpr, tpr, _ = roc_curve(y_opmd, y_pred_proba_opmd_1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC-AUC Curve: {name}")
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall Curve for OPMD dataset
    precision, recall, _ = precision_recall_curve(y_opmd, y_pred_proba_opmd_1)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"Precision-Recall Curve: {name}")
    plt.legend(loc="lower left")
    plt.show()


# In[ ]:


# Print coefficients using scikit-learn
for name, model in best_params_models.items():
    model.fit(X, y)
    print(f"{name} Coefficients (scikit-learn):")
    print(model.coef_[0])
    print()



# In[ ]:


# Print coefficients and intercepts using scikit-learn
for name, model in best_params_models.items():
    model.fit(X, y)
    print(f"{name} Coefficients (scikit-learn):")
    print("Coefficients:", model.coef_[0])
    print("Intercept:", model.intercept_)
    print()


# In[ ]:




for name, model in best_params_models.items():
    # Train the model on the whole 30 OSCC dataset
    model.fit(X, y)

    # Predict probabilities on the OPMD dataset
    y_pred_proba_opmd = model.predict_proba(X_opmd)

    # Print the predicted probabilities for both classes
    print(f"Predicted Probabilities for {name} Model:")
    print(y_pred_proba_opmd)

    # To get probabilities for class 0 and class 1 separately:
    y_pred_proba_class_0 = y_pred_proba_opmd[:, 0]  # Probabilities for class 0
    y_pred_proba_class_1 = y_pred_proba_opmd[:, 1]  # Probabilities for class 1

    # Example: Print class 0 and class 1 probabilities
    print(f"Probabilities for Class 0: {y_pred_proba_class_0}")
    print(f"Probabilities for Class 1: {y_pred_proba_class_1}")


# In[ ]:


for name, model in best_params_models.items():
    # Train the model on the whole 30 OSCC dataset
    model.fit(X, y)

    # Predict probabilities on the OPMD dataset
    y_pred_proba_opmd = model.predict_proba(X_opmd)

    # Apply a threshold to get binary predictions (0 or 1)
    threshold = 0.5  # You can adjust the threshold if needed

    # Convert predicted probabilities to binary predictions
    y_pred_opmd = (y_pred_proba_opmd[:, 1] >= threshold).astype(int)

    # Print the binary predictions
    print(f"Binary Predictions for {name} Model:")
    print(y_pred_opmd)


# # Model 4

# In[ ]:


#model 4 is model with 60 oscc tersted on opmd with no hyperparameters 


# In[ ]:


models = {
    "Logistic Regression": LogisticRegression(solver="liblinear", max_iter=10000 ),
    "Lasso": LogisticRegression(solver="liblinear", penalty="l1", max_iter=10000)
}


# In[ ]:


from sklearn.metrics import roc_curve, auc
for name, model in models.items():
    # Train the model on the whole 30 OSCC dataset

    # min max scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_opmd = scaler.transform(X_opmd)
    #X_oscc_51 = scaler.transform(X_oscc_51)


    model.fit(X, y)
    print(model.get_params())
    
    # Predict on the OPMD dataset
    # y_pred_opmd = model.predict(X_opmd)
    y_pred_proba_opmd = model.predict_proba(X_opmd)
    #print(y_pred_proba_opmd)
    y_pred_opmd = (y_pred_proba_opmd[:,1] >= prob_threshold).astype(int)
    #print(y_pred_opmd)
    cm_opmd = confusion_matrix(y_opmd, y_pred_opmd)
    roc_auc_opmd = roc_auc_score(y_opmd, y_pred_opmd)
    report_opmd = classification_report(y_opmd, y_pred_opmd)

    print(f"Model: {name}\n")
    print("OPMD dataset:")
    print("Confusion Matrix:")
    print(cm_opmd)
    print(f"ROC AUC Score: {roc_auc_opmd}")
    print("Classification Report:")
    print(report_opmd)
    print("\n51 OSCC dataset:")
    print("Confusion Matrix:")
    print("\n" + "=" * 80 + "\n")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_opmd, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"])
    plt.title(f"Confusion Matrix: {name}")
    plt.show()
    
    # ROC-AUC Curve for OPMD dataset
    y_pred_proba_opmd_1 = y_pred_proba_opmd[:, 1]  # Probabilities for class 1
    fpr, tpr, _ = roc_curve(y_opmd, y_pred_proba_opmd_1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC-AUC Curve: {name}")
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall Curve for OPMD dataset
    precision, recall, _ = precision_recall_curve(y_opmd, y_pred_proba_opmd_1)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"Precision-Recall Curve: {name}")
    plt.legend(loc="lower left")
    plt.show()


# In[ ]:


# Print coefficients using scikit-learn
for name, model in models.items():
    model.fit(X, y)
    print(f"{name} Coefficients (scikit-learn):")
    print(model.coef_[0])
    print()


# In[ ]:


# Print coefficients and intercepts using scikit-learn
for name, model in models.items():
    model.fit(X, y)
    print(f"{name} Coefficients (scikit-learn):")
    print("Coefficients:", model.coef_[0])
    print("Intercept:", model.intercept_)
    print()


# In[ ]:


from sklearn.metrics import roc_curve, auc

# Assuming you have already scaled X and X_opmd using StandardScaler

for name, model in models.items():
    # Train the model on the whole 30 OSCC dataset
    model.fit(X, y)

    # Predict probabilities on the OPMD dataset
    y_pred_proba_opmd = model.predict_proba(X_opmd)

    # Print the predicted probabilities for both classes
    print(f"Predicted Probabilities for {name} Model:")
    print(y_pred_proba_opmd)

    # To get probabilities for class 0 and class 1 separately:
    y_pred_proba_class_0 = y_pred_proba_opmd[:, 0]  # Probabilities for class 0
    y_pred_proba_class_1 = y_pred_proba_opmd[:, 1]  # Probabilities for class 1

    # Example: Print class 0 and class 1 probabilities separately
    print(f"Probabilities for Class 0: {y_pred_proba_class_0}")
    print(f"Probabilities for Class 1: {y_pred_proba_class_1}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




