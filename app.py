import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# load data
data = pd.read_csv('creditcard.csv')

# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

st.title("Deceit - Detector")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

input_df1 = st.text_input('Time')
input_df2 = st.text_input('Value 1')
input_df3 = st.text_input('Value 2')
input_df4 = st.text_input('Value 3')
input_df5 = st.text_input('Value 4')
input_df6 = st.text_input('Value 5')
input_df7 = st.text_input('Value 6')
input_df8 = st.text_input('Value 7')
input_df9 = st.text_input('Value 8')
input_df10 = st.text_input('Value 9')
input_df11 = st.text_input('Value 10')
input_df12 = st.text_input('Value 11')
input_df13 = st.text_input('Value 12')
input_df14 = st.text_input('Value 13')
input_df15 = st.text_input('Value 14')
input_df16 = st.text_input('Value 15')
input_df17 = st.text_input('Value 16')
input_df18 = st.text_input('Value 17')
input_df19 = st.text_input('Value 18')
input_df20 = st.text_input('Value 19')
input_df21 = st.text_input('Value 20')
input_df22 = st.text_input('Value 21')
input_df23 = st.text_input('Value 22')
input_df24 = st.text_input('Value 23')
input_df25 = st.text_input('Value 24')
input_df26 = st.text_input('Value 25')
input_df27 = st.text_input('Value 26')
input_df28 = st.text_input('Value 27')
input_df29 = st.text_input('Value 28')
input_df30 = st.text_input('Amount')
input_df_lst = [input_df1, input_df2, input_df3, input_df4, input_df5, input_df6, input_df7, input_df8, input_df9, input_df10, input_df11, input_df12, input_df13, input_df14, input_df15, input_df16, input_df17, input_df18, input_df19, input_df20, input_df21, input_df22, input_df23, input_df24, input_df25, input_df26, input_df27, input_df28, input_df29, input_df30]
submit = st.button("Submit")

if submit:
    features = np.array(input_df_lst, dtype=np.float64)
    prediction = model.predict(features.reshape(1,-1))
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")
