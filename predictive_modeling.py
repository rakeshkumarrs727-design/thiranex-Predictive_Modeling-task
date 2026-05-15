import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. LOAD DATA
# Ensure train.csv is uploaded directly to your GitHub main page
df = pd.read_csv('train.csv')

# 2. DATA CLEANING (Prevents errors during training)
# Filling missing Age with the median and Embarked with the most frequent value
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna('S')

# 3. ENCODING CATEGORICAL DATA
# Machines need numbers, so we change 'male/female' to 0/1
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Selecting features (the variables used to predict)
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# 4. TRAIN/TEST SPLIT
# 80% of data is for training, 20% is for testing accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. MODELING (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. EVALUATION: Confusion Matrix (Requirement)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'])
plt.title('Confusion Matrix: Survival Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 7. EVALUATION: ROC Curve (Requirement)
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('Model Performance: ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

print(f"Final Prediction Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")
