import streamlit as st
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os
from openai import OpenAI
import time

# Initialize OpenAI client securely
client = OpenAI(api_key=os.environ.get("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"))

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

# --- LOAD CSV AND STORE IN DB IF NOT ALREADY THERE ---
if 'data_loaded' not in st.session_state:
    df = pd.read_csv('data.csv')

    for _, row in df.iterrows():
        # Check if already exists in DB
        c.execute("SELECT summary FROM customers WHERE customer_id=?", (int(row['customer_id']),))
        result = c.fetchone()
        if result:
            continue  # skip if summary already exists

        # Call OpenAI once per new row
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",

