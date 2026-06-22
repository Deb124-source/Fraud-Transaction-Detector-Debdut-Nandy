import streamlit as st
import pandas as pd
import joblib


# -------------------------------
# Load Models
# -------------------------------

model = joblib.load(
    "models/fraud_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

encoder = joblib.load(
    "models/type_encoder.pkl"
)



# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(

    page_title="Fraud Transaction Detector",

    page_icon="💳",

    layout="centered"

)



# -------------------------------
# Title
# -------------------------------

st.title(
    "💳 AI Powered Fraud Transaction Detector"
)


st.write(
    "Machine Learning based transaction fraud prediction system"
)



# -------------------------------
# User Inputs
# -------------------------------


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



# -------------------------------
# Prediction
# -------------------------------


if st.button("Predict Transaction"):


    encoded_type = encoder.transform(
        [transaction_type]
    )[0]


    balance_change_org = (
        old_balance_org -
        new_balance_org
    )


    balance_change_dest = (
        old_balance_dest -
        new_balance_dest
    )


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



    # Scale amount

    input_data["amount"] = scaler.transform(

        input_data[["amount"]]

    )



    prediction = model.predict(

        input_data

    )


    probability = model.predict_proba(

        input_data

    )[0][1]



    st.subheader(
        "Prediction Result"
    )


    if prediction[0] == 1:


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
