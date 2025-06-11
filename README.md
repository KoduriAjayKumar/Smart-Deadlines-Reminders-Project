# 📧 Smart Deadline Reminders

A Flask-based web app that monitors your Gmail inbox for urgent or time-sensitive emails and alerts you automatically using a Machine Learning model (LSTM). Helps you avoid missing important deadlines by showing all urgent emails in a simple dashboard.

---

## 🧠 Problem Statement

In daily life, we receive many emails — and it's easy to miss important ones with deadlines. Manually checking every mail is time-consuming. This project solves that problem by:

- Automatically reading unread Gmail messages
- Using an ML model to detect urgent messages
- Alerting the user and displaying these emails in a dashboard

No manual checking is required. Just log in once, and let the app handle it.

---

## 🛠 Tech Stack Used

- **Frontend:** HTML, CSS (for login page)
- **Backend:** Python, Flask
- **Database:** SQLite
- **Email Access:** IMAP (reading), SMTP (alerts)
- **Machine Learning:** Keras LSTM Model
- **Libraries:** `imaplib`, `smtplib`, `keras`, `flask`, `pickle`, `sqlite3`, `email`, `threading`, `re`, `numpy`

---

## 🚀 Features

- Gmail login using App Password
- Auto-fetch unread emails from last 30 days
- Classify emails as "Time-sensitive" or "Normal"
- Show urgent emails in dashboard
- Email alert for urgent mails
- Mark emails as complete or delete them

---

## ⚙️ How to Run the Project

### 1. Clone the Repo


git clone https://github.com/yourusername/smart-deadline-reminders.git
cd smart-deadline-reminders

###2. Install Required Libraries
Using requirements.txt:


pip install -r requirements.txt
Or install manually:


pip install flask keras tensorflow numpy
###3. Set Up Your Gmail App Password
Enable 2-Step Verification in your Gmail account.

Go to Google Account → Security → App Passwords.

Generate an app password for "Mail".

Use that 16-character password in the app login.

###4. Run the App

python app.py
Open your browser and go to: http://localhost:5000

Log in with your Gmail and app password.

The dashboard will display your urgent emails.

###5. Training the ML Model (Optional)
If you want to retrain the model using your own dataset:


python model_training.py
This will generate:

time_sensitive_model.h5 → the trained LSTM model

tokenizer.pkl → the tokenizer for preprocessing
