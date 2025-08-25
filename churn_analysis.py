# Import the pandas library, which is essential for working with data in Python
import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # type: ignore

# Load the CSV file into a pandas DataFrame
# A DataFrame is like a spreadsheet or a table within your Python code
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 1. Preview the first 5 rows of your data
print("---------- First 5 Rows of the Data ----------")
print(df.head())

# 2. Get a summary of the dataset's structure
print("\n---------- Dataset Information ----------")
df.info()

# 3. Get a statistical summary of the numerical columns
print("\n---------- Statistical Summary ----------")
print(df.describe())

# --- Data Cleaning and Preprocessing ---

# 1. Convert 'TotalCharges' from text to a number.
# 'errors='coerce'' will turn any values that can't be converted (like empty spaces) into 'NaN' (Not a Number).
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


# 2. Handle the Missing Values created in the step above.
# First, we check how many missing values were created.
print("---------- Missing Values Before Cleaning ----------")
print(df.isnull().sum())

# Since it's a very small number of rows, we can safely remove them.
df.dropna(inplace=True)

print("\n---------- Missing Values After Cleaning ----------")
print(df.isnull().sum())


# 3. Drop the 'customerID' column because it's just a unique identifier and has no predictive value.
df.drop(columns=['customerID'], inplace=True)


# 4. Standardize several 'Yes'/'No' columns into numbers (1/0). This is essential for our model later.
for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']:
    if col in df.columns:
        df[col] = df[col].replace({'Yes': 1, 'No': 0})

# --- Final Check ---
# Let's look at the .info() output again to confirm our changes.
print("\n---------- Cleaned Data Info ----------")
df.info()

print("\n---------- Cleaned Data Preview ----------")
print(df.head())


# --- Exploratory Data Analysis (EDA) ---

# Set a professional style for our plots
sns.set_style("whitegrid")

# 1. First, let's see our overall churn rate.
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution (0 = Stayed, 1 = Churned)')
plt.ylabel('Number of Customers')
plt.show()

# 2. Now, let's investigate how key services and contracts relate to churn.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Churn by Key Customer Segments', fontsize=16)

# Churn by Contract Type
sns.countplot(x='Contract', hue='Churn', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Churn is Highest in Month-to-Month Contracts')

# Churn by Internet Service
sns.countplot(x='InternetService', hue='Churn', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Churn is Higher for Fiber Optic Users')

# Churn by Tech Support
sns.countplot(x='TechSupport', hue='Churn', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Customers Without Tech Support Churn More')

# Churn by Payment Method
sns.countplot(x='PaymentMethod', hue='Churn', data=df, ax=axes[1, 1])
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha="right") # Rotate labels to prevent overlap
axes[1, 1].set_title('Electronic Check Users Have High Churn')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
plt.show()

# 3. Finally, let's look at the financial and loyalty aspects.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Financial and Loyalty Factors by Churn', fontsize=16)

# Monthly Charges
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False)
axes[0].set_title('Higher Monthly Charges Lead to More Churn')

# Total Charges
sns.kdeplot(data=df, x='TotalCharges', hue='Churn', fill=True, common_norm=False)
axes[1].set_title('Low Total Charges Indicate Churn (New Customers)')

# Tenure (Customer Loyalty)
sns.kdeplot(data=df, x='tenure', hue='Churn', fill=True, common_norm=False)
axes[2].set_title('New Customers (Low Tenure) Churn Most')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



# --- Feature Preprocessing ---

# First, let's separate our target variable (what we want to predict) from our features (the inputs).
X = df.drop('Churn', axis=1) # All columns except 'Churn'
y = df['Churn']              # Only the 'Churn' column

# Now, we'll convert all the categorical columns in our features (X) into dummy/indicator variables.
# The `drop_first=True` argument helps to avoid redundancy in our data.
X = pd.get_dummies(X, drop_first=True)

# Let's look at our processed data to see the new columns.
print("---------- Processed Features (X) Preview ----------")
print(X.head())

print("\n---------- Shape of our new features dataset ----------")
print(X.shape)


# --- Split the Data into Training and Testing Sets ---

# We'll split the data, using 80% for training and 20% for testing.
# The 'random_state=42' ensures that we get the same split every time we run the code, which is good for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Let's check the size of our new datasets
print("---------- Data Split Summary ----------")
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)



# --- Build, Train, and Evaluate the Model ---

# 1. Create an instance of the Logistic Regression model
model = LogisticRegression(max_iter=1000) # max_iter is increased to ensure the model converges

# 2. Train the model on your training data
print("---------- Training the model... ----------")
model.fit(X_train, y_train)
print("Model training complete!")

# 3. Make predictions on the test data (the data the model has never seen)
print("\n---------- Making predictions on the test data... ----------")
y_pred = model.predict(X_test)
print("Predictions made!")

# 4. Evaluate the model's performance
print("\n---------- Model Evaluation ----------")

# Accuracy Score: What percentage of predictions were correct?
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix: A table to see where the model got confused.
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report: A detailed report on the model's performance.
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

df.to_csv('cleaned_churn_data.csv', index=False)