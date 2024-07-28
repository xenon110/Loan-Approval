import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load and preprocess the data
location = r"C:\Users\mayan\Downloads\01Exercise1.csv"
loandata = pd.read_csv(location)

loanprep = loandata.copy()

# Identify and drop null values
loanprep = loanprep.dropna()

# Drop the irrelevant column from the data
loanprep = loanprep.drop(['gender'], axis=1)

# Create dummy variables for the categorical variable
loanprep = pd.get_dummies(loanprep, drop_first=True)

# Normalize the data using StandardScaler
scaler = StandardScaler()
loanprep[['income', 'loanamt']] = scaler.fit_transform(loanprep[['income', 'loanamt']])

# Create the variable X and Y
Y = loanprep[['status_Y']]
X = loanprep.drop(['status_Y'], axis=1)

# Split the X and Y datasets into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)

# Build the logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train.values.ravel())
Y_predict = model.predict(X_test)
# print(Y_predict)

# Build the confusion matrix and get the accuracy/score
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)
result = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
print(result)
score = model.score(X_test, Y_test)
print(score)

# User input for prediction
user_married = int(input("Enter 1 if you are married and 0 if not: "))
user_income = float(input("Enter your income: "))
user_loanamt = float(input("Enter your loan amount: "))

# Create a DataFrame for user input
user_data = pd.DataFrame({
    'married': [user_married],
    'income': [user_income],
    'loanamt': [user_loanamt]
})

# Normalize the user input using the same scaler
user_data[['income', 'loanamt']] = scaler.transform(user_data[['income', 'loanamt']])

# Add dummy variables for the categorical feature
user_data = pd.get_dummies(user_data, drop_first=True)

# Ensure the user_data DataFrame has the same columns as the training data
for col in X.columns:
    if col not in user_data.columns:
        user_data[col] = 0

# Reorder columns to match the training data
user_data = user_data[X.columns]

# Predict the loan approval status for the user input
user_prediction = model.predict(user_data)

# Output the result
if user_prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Not Approved")
