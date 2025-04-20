#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


train = pd.read_csv("C:/Users/ongmtp/Documents/Aly6020/M3/Training_Data-1.csv")
test = pd.read_csv("C:/Users/ongmtp/Documents/Aly6020/M3/Test_Data.csv")


# In[3]:


train


# In[4]:


test


# In[5]:


train.isnull().sum()


# In[6]:


test.isnull().sum()


# In[7]:


train = train.drop('POINTS SCORED\nBY EITHER TEAM', axis=1)
test = test.drop('POINTS SCORED\nBY EITHER TEAM', axis=1)


# In[8]:


import matplotlib.pyplot as plt

# Create a bar chart for 'YARDS GAINED' including NaN values
plt.figure(figsize=(10, 6))
train['YARDS GAINED'].value_counts(dropna=False).sort_index().plot(kind='bar')

# Add title and labels
plt.title('Distribution of Yards Gained (Including NaN)', fontsize=14)
plt.xlabel('Yards Gained', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=90)

# Display the bar chart
plt.tight_layout()
plt.show()


# In[9]:


# Fill NaN values in 'YARDS GAINED' with the median
train['YARDS GAINED'] = train['YARDS GAINED'].fillna(train['YARDS GAINED'].median())

test['YARDS GAINED'] = test['YARDS GAINED'].fillna(test['YARDS GAINED'].median())


# In[10]:


train.isnull().sum()


# In[11]:


# Create a bar chart for 'YARDS GAINED' including NaN values
plt.figure(figsize=(10, 6))
train['DOWN'].value_counts(dropna=False).sort_index().plot(kind='bar')

# Add title and labels
plt.title('Distribution of DOWN(Including NaN)', fontsize=14)
plt.xlabel('DOWN', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=90)

# Display the bar chart
plt.tight_layout()
plt.show()


# In[12]:


train["DOWN"] = train["DOWN"].fillna(train["DOWN"].mode()[0])


# In[13]:


train.isnull().sum()


# In[14]:


train


# In[15]:


print(train.dtypes)


# # Feature Engineering

# In[16]:


# Convert 'hh:mm:ss' to total seconds assuming it represents minutes
train['Remain_Time_in_Seconds'] = train['REMAINING TIME IN THE QUARTER (mm:ss)'].apply(
    lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1])
)

# Verify the result
print(train[['REMAINING TIME IN THE QUARTER (mm:ss)', 'Remain_Time_in_Seconds']].head())


# In[17]:


train


# In[18]:


train = train.drop('REMAINING TIME IN THE QUARTER (mm:ss)', axis=1)


# In[19]:


train


# In[20]:


train['game_id'] = train['DATE'] + '-' + train['OFFENSIVE TEAM'] + '-' + train['DEFENSIVE TEAM']
test['game_id'] = test['DATE'] + '-' + test['OFFENSIVE TEAM'] + '-' + test['DEFENSIVE TEAM']


# In[21]:


selected = ['game_id', 'QUARTER', 'DOWN', 'TO GO', 'YARD LINE 0-100', 'Play_Type', 'Remain_Time_in_Seconds', 'OFFENSIVE TEAM']


# In[22]:


train = train[selected]


# In[23]:


train


# In[24]:


# Convert 'hh:mm:ss' to total seconds assuming it represents minutes
test['Remain_Time_in_Seconds'] = test['REMAINING TIME IN THE QUARTER (mm:ss)'].apply(
    lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1])
)

# Verify the result
print(test[['REMAINING TIME IN THE QUARTER (mm:ss)', 'Remain_Time_in_Seconds']].head())


# In[25]:


test = test[selected]


# In[26]:


test


# In[27]:


# Convert 'DOWN' column to integer type
train['DOWN'] = train['DOWN'].astype(int)

# Verify the changes
print(train['DOWN'].head())


# In[28]:


test.isnull().sum()


# In[29]:


test = test.dropna(subset=['DOWN'])


# In[30]:


print(test['DOWN'].isnull().sum())


# In[31]:


# Convert 'DOWN' column to integer type
test['DOWN'] = test['DOWN'].astype(int)

# Verify the changes
print(test['DOWN'].head())


# In[32]:


train


# In[33]:


test


# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create boxplots for numeric features
for column in train.select_dtypes(include=['number']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=train[column])
    plt.title(f"Boxplot for {column}")
    plt.show()


# In[35]:


# Create boxplots for numeric features
for column in test.select_dtypes(include=['number']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=train[column])
    plt.title(f"Boxplot for {column}")
    plt.show()


# In[36]:


# Calculate Q1, Q3, and IQR for 'TO GO'
Q1 = train['TO GO'].quantile(0.25)
Q3 = train['TO GO'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out rows where 'TO GO' is an outlier
train = train[(train['TO GO'] >= lower_bound) & (train['TO GO'] <= upper_bound)]


# In[37]:


# Calculate Q1, Q3, and IQR for 'TO GO'
Q1 = test['TO GO'].quantile(0.25)
Q3 = test['TO GO'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out rows where 'TO GO' is an outlier
test = test[(test['TO GO'] >= lower_bound) & (test['TO GO'] <= upper_bound)]


# In[38]:


train


# In[39]:


test


# In[40]:


train['red_zone'] = train['YARD LINE 0-100'].apply(lambda x: 1 if x <= 20 else 0)
test['red_zone'] = test['YARD LINE 0-100'].apply(lambda x: 1 if x <= 20 else 0)


# In[41]:


# Group by 'red_zone' and 'Play_Type' to count occurrences
redzone_counts = train.groupby(['red_zone', 'Play_Type']).size().unstack()

# Plot the bar chart
redzone_counts.plot(kind='bar', figsize=(10, 6))

# Add titles and labels
plt.title('Amount of Run or Pass in Red Zone and Non-Red Zone', fontsize=16)
plt.xlabel('Red Zone (1 = In Red Zone, 0 = Not in Red Zone)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Play Type', fontsize=10)

# Display the plot
plt.tight_layout()
plt.show()


# In[42]:


# Last two minutes of the quarter
train['late quarter'] = train['Remain_Time_in_Seconds'].apply(
    lambda x: 1 if int(x) <= 120 else 0 # Check if remaining time in seconds is within the last two minutes
)

# Last two minutes of the quarter
test['late quarter'] = test['Remain_Time_in_Seconds'].apply(
    lambda x: 1 if int(x) <= 120 else 0 # Check if remaining time in seconds is within the last two minutes
)


# In[43]:


train['TO_GO_DOWN_INTERACTION'] = train['TO GO'] * train['DOWN']
test['TO_GO_DOWN_INTERACTION'] = test['TO GO'] * test['DOWN']


# In[44]:


print(train.dtypes)
print(test.dtypes)


# In[45]:


train['pass_attempts'] = train.groupby(['game_id', 'OFFENSIVE TEAM'])['Play_Type'].transform(
    lambda x: (x == 'Pass').cumsum()
)
train['run_attempts'] = train.groupby(['game_id', 'OFFENSIVE TEAM'])['Play_Type'].transform(
    lambda x: (x == 'Run').cumsum()
)
test['pass_attempts'] = test.groupby(['game_id', 'OFFENSIVE TEAM'])['Play_Type'].transform(
    lambda x: (x == 'Pass').cumsum()
)
test['run_attempts'] = test.groupby(['game_id', 'OFFENSIVE TEAM'])['Play_Type'].transform(
    lambda x: (x == 'Run').cumsum()
)


# In[46]:


# Calculate previous pass attempts
train['previous_pass_attempts'] = train.groupby(['game_id', 'OFFENSIVE TEAM'])['pass_attempts'].shift(1)
test['previous_pass_attempts'] = test.groupby(['game_id', 'OFFENSIVE TEAM'])['pass_attempts'].shift(1)

train['previous_pass_attempts'] = train['previous_pass_attempts'].fillna(0)
test['previous_pass_attempts'] = test['previous_pass_attempts'].fillna(0)


# In[47]:


#Run attempt
train['previous_run_attempts'] = train.groupby(['game_id', 'OFFENSIVE TEAM'])['run_attempts'].shift(1)
test['previous_run_attempts'] = test.groupby(['game_id', 'OFFENSIVE TEAM'])['run_attempts'].shift(1)

train['previous_run_attempts'] = train['previous_run_attempts'].fillna(0)
test['previous_run_attempts'] = test['previous_run_attempts'].fillna(0)


# In[48]:


train


# # Model

# In[49]:


train['Play_Type'] = train['Play_Type'].map({'Pass': 0, 'Run': 1})


# In[50]:


test['Play_Type'] = test['Play_Type'].map({'Pass': 0, 'Run': 1})


# In[51]:


train


# In[52]:


test


# In[53]:


X_train = train.drop(columns=['game_id', 'Play_Type', 'OFFENSIVE TEAM', 'pass_attempts', 'run_attempts'])
y_train = train['Play_Type']

X_test = test.drop(columns=['game_id', 'Play_Type', 'OFFENSIVE TEAM', 'pass_attempts', 'run_attempts'])
y_test = test['Play_Type']


# In[54]:


# Check for missing values in y_train
print(y_train.isnull().sum())


# In[55]:


train['previous_pass_attempts'] = train['previous_pass_attempts'].astype(int)
test['previous_pass_attempts'] = test['previous_pass_attempts'].astype(int)
train['previous_run_attempts'] = train['previous_run_attempts'].astype(int)
test['previous_run_attempts'] = test['previous_run_attempts'].astype(int)


# # XGB

# In[56]:


from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[78]:


xgb_clf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)


# In[79]:


# Check for missing values in y_train
print(y_train.isnull().sum())


# In[80]:


xgb_clf.fit(X_train, y_train)


# In[81]:


print(X_train.head())
print(X_train.columns)


# In[82]:


print(train.dtypes)
print(test.dtypes)


# In[83]:


# 4. Make predictions
y_pred = xgb_clf.predict(X_test)


# In[84]:


# 5. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[64]:


from xgboost import plot_importance
import matplotlib.pyplot as plt

# Assuming `model` is your trained XGBClassifier
# Plot feature importance
plt.figure(figsize=(10, 6))
plot_importance(xgb_clf, importance_type='weight')  # Use 'weight', 'gain', or 'cover'
plt.title("Feature Importance (XGBoost)")
plt.show()


# In[67]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
conf_matrix = confusion_matrix(y_test, y_pred)

# Create and plot the confusion matrix display
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Run", "Pass"])  # Update labels as needed
disp.plot(cmap=plt.cm.Blues)

# Add title and show the plot
plt.title("Confusion Matrix")
plt.show()


# # Random Forest

# In[68]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Create the Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=500,      # Number of trees in the forest
    max_depth=None,        # Maximum depth of the tree (default: no limit)
    random_state=42,       # Random seed for reproducibility
    bootstrap=True,        # Use bootstrapping (default: True)
    class_weight='balanced' # Handle imbalanced classes (optional)
)

# Train the classifier
rf_clf.fit(X_train, y_train)


# In[69]:


# Predict on the test set
y_pred = rf_clf.predict(X_test)


# In[70]:


# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[86]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Add an intercept column (if necessary) for VIF calculation
X_train['Intercept'] = 1

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X_train.columns
vif_data['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

# Drop the intercept row
vif_data = vif_data[vif_data['Feature'] != 'Intercept']

# Display the VIF values
print(vif_data)


# In[ ]:




