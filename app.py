import streamlit as st
import pandas as pd
import joblib


# ---------------------------
# Load Saved Files
# ---------------------------

model = joblib.load(
    "models/fraud_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

encoder = joblib.load(
    "models/type_encoder.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)



# ---------------------------
# Page Config
# ---------------------------

st.set_page_config(

    page_title="AI Fraud Detector",

    page_icon="💳",

    layout="centered"

)



# ---------------------------
# Title
# ---------------------------

st.title(
    "💳 AI Powered Fraud Transaction Detector"
)

st.write(
    "Machine Learning based transaction fraud prediction system"
)



# ---------------------------
# Inputs
# ---------------------------

step = st.number_input(

    "Transaction Time Step",

    min_value=0

)



transaction_type = st.selectbox(

    "Transaction Type",

    [

        "PAYMENT",

        "TRANSFER",

        "CASH_OUT",

        "DEBIT",

        "CASH_IN"

    ]

)



amount = st.number_input(

    "Transaction Amount",

    min_value=0.0

)



old_balance_org = st.number_input(

    "Sender Old Balance",

    min_value=0.0

)



new_balance_org = st.number_input(

    "Sender New Balance",

    min_value=0.0

)



old_balance_dest = st.number_input(

    "Receiver Old Balance",

    min_value=0.0

)



new_balance_dest = st.number_input(

    "Receiver New Balance",

    min_value=0.0

)



# ---------------------------
# Prediction
# ---------------------------


if st.button("Predict Transaction"):


    # Encode transaction type

    encoded_type = encoder.transform(

        [transaction_type]

    )[0]



    # Feature Engineering

    balance_change_org = (

        old_balance_org -

        new_balance_org

    )


    balance_change_dest = (

        old_balance_dest -

        new_balance_dest

    )



    # Create dataframe

    input_data = pd.DataFrame({

        "step":[step],

        "type":[encoded_type],

        "amount":[amount],

        "oldbalanceOrg":[old_balance_org],

        "newbalanceOrig":[new_balance_org],

        "oldbalanceDest":[old_balance_dest],

        "newbalanceDest":[new_balance_dest],

        "balance_change_org":[balance_change_org],

        "balance_change_dest":[balance_change_dest]

    })


    # Same order as training

    input_data = input_data[feature_columns]



    # Scale amount only

    input_data["amount"] = scaler.transform(

        input_data[["amount"]]

    )



    # Fraud probability

    probability = model.predict_proba(

        input_data

    )[0][1]



    # Custom threshold

    if probability > 0.30:

        prediction = 1

    else:

        prediction = 0



    st.subheader(
        "Prediction Result"
    )



    if prediction == 1:


        st.error(

            "⚠️ Fraud Transaction Detected"

        )


        st.write(

            f"Fraud Probability: {probability:.2%}"

        )


    else:


        st.success(

            "✅ Legitimate Transaction"

        )


        st.write(

            f"Fraud Probability: {probability:.2%}"

        )
