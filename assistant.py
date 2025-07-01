import streamlit as st
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os
from openai import OpenAI

# ✅ Initialize OpenAI client (secure: reads key from env)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- DATABASE SETUP ---
conn = sqlite3.connect('customer_data.db')
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation TEXT,
    churned INTEGER,
    summary TEXT
)
''')
conn.commit()

# --- LOAD CSV AND STORE IN DB (only first time) ---
if 'data_loaded' not in st.session_state:
    df = pd.read_csv('data.csv')
    for _, row in df.iterrows():
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize this customer message politely in one sentence."},
                {"role": "user", "content": row['conversation']}
            ]
        )
        summary = response.choices[0].message.content
        c.execute("INSERT OR IGNORE INTO customers (customer_id, conversation, churned, summary) VALUES (?, ?, ?, ?)",
                  (int(row['customer_id']), row['conversation'], int(row['churned']), summary))
    conn.commit()
    st.session_state['data_loaded'] = True

# --- DASHBOARD UI ---
st.title("📊 Vodafone AI-powered Customer Assistant")

# Load data from DB
df = pd.read_sql_query("SELECT * FROM customers", conn)
st.subheader("📄 Customer Data")
st.write(df)

# --- Churn prediction model ---
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["conversation"])
y = df["churned"]
model = LogisticRegression()
model.fit(X, y)

# --- Visualize churn distribution ---
st.subheader("📊 Churn Distribution")
fig, ax = plt.subplots()
df["churned"].value_counts().plot(kind="bar", ax=ax)
ax.set_xticklabels(["Not churned", "Churned"], rotation=0)
st.pyplot(fig)

# --- Test on new customer message ---
st.subheader("✏️ Test on new customer message")
new_text = st.text_input("Enter message:")
if new_text:
    # Predict churn
    new_vec = vectorizer.transform([new_text])
    pred = model.predict(new_vec)[0]
    st.write("✅ Prediction:", "Likely to churn" if pred == 1 else "Not likely to churn")

    # Summarize with GPT
    summary_resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Summarize this message politely in one sentence."},
            {"role": "user", "content": new_text}
        ]
    )
    summary_text = summary_resp.choices[0].message.content
    st.write("📄 AI-generated Summary:", summary_text)

    # Store new entry in DB
    c.execute("INSERT INTO customers (conversation, churned, summary) VALUES (?, ?, ?)",
              (new_text, int(pred), summary_text))
    conn.commit()
