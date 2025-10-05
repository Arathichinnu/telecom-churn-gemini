# coding: utf-8
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
import pickle
import google.generativeai as genai

app = Flask(__name__)

# Load your dataset
df_1 = pd.read_csv("first_telc.csv")

# Load trained churn model
model = pickle.load(open("model.sav", "rb"))
model_features = model.feature_names_in_

# Configure Gemini API from environment variable
load_dotenv()  
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDpeKymyNapCJmfcz8V-aIDbPn-EL_KXz4")  # Replace if testing
genai.configure(api_key=GEMINI_KEY)

generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_output_tokens": 500
}

gemini_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config
)

def chat_with_gemini(prompt):
    try:
        response = gemini_model.generate_content([prompt])
        return response.text if response.text else "No suggestion generated."
    except Exception as e:
        print(f"Error contacting Gemini API: {e}")
        return "Chatbot temporarily unavailable due to API issues."

@app.route("/")
def loadPage():
    return render_template('home.html',
                           **{f'query{i}': "" for i in range(1, 20)},
                           output1="", output2="", chat_response="")

@app.route("/", methods=['POST'])
def predict():
    try:
        # Collect all 19 inputs
        inputs = [request.form.get(f'query{i}', '') for i in range(1, 20)]

        # Map inputs to DataFrame
        new_df = pd.DataFrame([inputs], columns=[
            'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
            'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'tenure'
        ])

        # Merge with existing dataset
        df_2 = pd.concat([df_1, new_df], ignore_index=True)

        # Handle tenure safely
        df_2['tenure'] = pd.to_numeric(df_2['tenure'], errors='coerce').fillna(0).astype(int)
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        df_2['tenure_group'] = pd.cut(df_2.tenure, range(1, 80, 12), right=False, labels=labels)
        df_2.drop(columns=['tenure'], inplace=True)

        # One-hot encoding
        df_2 = pd.get_dummies(df_2, columns=[
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'tenure_group'
        ])

        # Align columns with model features
        for col in model_features:
            if col not in df_2:
                df_2[col] = 0
        df_2 = df_2[model_features]

        # Prediction
        single = model.predict(df_2.tail(1))
        probability = model.predict_proba(df_2.tail(1))[:, 1]

        if single == 1:
            o1 = "This customer is likely to churn!"
        else:
            o1 = "This customer is likely to continue!"
        o2 = f"Confidence: {probability[0]*100:.2f}%"

        chat_prompt = (
    "Use the customer inputs to suggest simple, clear, and practical ways "
    "to improve telecom customer satisfaction. "
    "Give the suggestions as short bullet points, one per line, "
    "so that normal people can easily understand."
    "\n\nCustomer Inputs: " + ", ".join(inputs))
                                                                                                            
        chat_response = chat_with_gemini(chat_prompt)


    except Exception as e:
        o1, o2, chat_response = "Error processing request", str(e), "Chatbot unavailable."

    return render_template('home.html',
                           output1=o1, output2=o2, chat_response=chat_response,
                           **{f'query{i}': inputs[i-1] for i in range(1, 20)}
                           )

if __name__ == '__main__':
    app.run(debug=True)
