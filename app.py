# EMAIL DEADLINE MANAGEMENT SYSTEM:

# A Flask-based web application that monitors Gmail inboxes
# for time-sensitive emails and sends alerts when urgent
# messages are detected using machine learning classification.


# IMPORTS - All required libraries and modules:

import imaplib          # For connecting to Gmail IMAP server
import email           # For parsing email messages
import smtplib         # For sending email alerts
import pickle          # For loading the trained tokenizer
import numpy as np     # For numerical operations
import sqlite3         # For database operations
import threading       # For background email monitoring
import time           # For delays and timing
from email.mime.text import MIMEText  # For creating email messages
from email import header              # For decoding email headers
from keras.models import load_model   # For loading ML model
from keras.preprocessing.sequence import pad_sequences  # For text preprocessing
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash  # Flask web framework
from werkzeug.security import generate_password_hash, check_password_hash  # For password security
from datetime import datetime, timedelta  # For date/time operations
import re             # For regular expressions
import os             # For file system operations
import json           # For JSON operations

# FLASK APPLICATION SETUP:
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production!


# DATABASE INITIALIZATION:

def init_db():
    """
    Initialize SQLite database with required tables.
    Creates three main tables:
    1. users - Store user credentials
    2. deadline_emails - Store detected urgent emails
    3. processed_emails - Track processed emails to avoid duplicates
    """
    conn = sqlite3.connect('email_deadlines.db')
    cursor = conn.cursor()
    
    # Users table - stores Gmail credentials for each user
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            app_password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Deadline emails table - stores detected urgent emails
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS deadline_emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            sender TEXT NOT NULL,
            subject TEXT NOT NULL,
            body TEXT,
            email_date TIMESTAMP,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_completed BOOLEAN DEFAULT FALSE,
            is_deleted BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Processed emails table - prevents duplicate processing
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            email_id TEXT UNIQUE,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# MACHINE LEARNING MODEL LOADING:

# Define paths for the trained model and tokenizer:
MODEL_PATH = "time_sensitive_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

# Try to load the pre-trained LSTM model and tokenizer
# If loading fails, the system will fall back to keyword-based classification
try:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)
    print("✅ LSTM model and tokenizer loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Error loading model/tokenizer: {e}")
    model = None
    tokenizer = None
    MODEL_LOADED = False


# EMAIL PROCESSING CLASS:

class EmailProcessor:
    """
    Main class for processing emails, classification, and monitoring.
    Handles email fetching, ML-based classification, and alert sending.
    """
    
    def __init__(self):
        """Initialize the email processor with default settings."""
        self.running = False
        self.check_interval = 300  # Check every 5 minutes (300 seconds)
    
    def decode_subject(self, encoded_subject):
        """
        Decode email subject lines that may be encoded (e.g., UTF-8, base64).
        Gmail subjects can be encoded in various formats, this handles them properly.
        
        Args:
            encoded_subject (str): Raw encoded subject line
            
        Returns:
            str: Decoded, readable subject line
        """
        if not encoded_subject:
            return ""
        
        # Decode each part of the subject (some may be encoded differently)
        decoded_parts = header.decode_header(encoded_subject)
        subject = ""
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                # Decode bytes to string using detected encoding
                subject += part.decode(encoding or "utf-8", errors="ignore")
            else:
                # Already a string, just add it
                subject += part
        return subject.strip()
    
    def preprocess_text(self, text, max_len=100):
        """
        Preprocess text for machine learning model input.
        Converts text to sequences that the LSTM model can understand.
        
        Args:
            text (str): Raw text to preprocess
            max_len (int): Maximum sequence length for padding
            
        Returns:
            numpy.ndarray: Preprocessed text sequence ready for ML model
        """
        if not MODEL_LOADED:
            return None
        
        # Clean text: remove non-word characters and convert to lowercase
        text = re.sub(r"\W", " ", text)
        text = text.lower().strip()
        
        # Convert text to sequences using the trained tokenizer
        sequences = tokenizer.texts_to_sequences([text])
        
        # Pad sequences to ensure consistent input size
        padded_seq = pad_sequences(sequences, maxlen=max_len, padding="post")
        return padded_seq
    
    def classify_email(self, subject, body):
        """
        Classify an email as 'Time-sensitive' or 'Normal' using ML model.
        Falls back to keyword-based classification if ML model isn't available.
        
        Args:
            subject (str): Email subject line
            body (str): Email body content
            
        Returns:
            str: Classification result ('Time-sensitive' or 'Normal')
        """
        if not MODEL_LOADED:
            # Fallback: Simple keyword-based classification
            urgent_keywords = ['urgent', 'deadline', 'asap', 'immediate', 'due', 'expires', 'final notice']
            text = (subject + " " + body).lower()
            return "Time-sensitive" if any(keyword in text for keyword in urgent_keywords) else "Normal"
        
        # ML-based classification using LSTM model
        # Preprocess both subject and body
        subject_seq = self.preprocess_text(subject)
        body_seq = self.preprocess_text(body)
        
        # Create numerical features (length of subject, body, exclamation marks)
        num_features = np.array([[len(subject), len(body), subject.count("!") + body.count("!")]])
        
        # Get prediction from the trained model
        prediction = model.predict([subject_seq, num_features])[0][0]
        
        # Classify based on prediction threshold (0.5)
        return "Time-sensitive" if prediction > 0.5 else "Normal"
    
    def is_alert_email(self, subject, sender, user_email):
        """
        Check if an email is an alert we sent to avoid infinite processing loop.
        This prevents the system from processing its own alert emails.
        
        Args:
            subject (str): Email subject
            sender (str): Email sender
            user_email (str): User's email address
            
        Returns:
            bool: True if this is our own alert email
        """
        if sender and user_email.lower() in sender.lower():
            if "🚨 Urgent Email Alert:" in subject:
                return True
        return False
    
    def send_email_alert(self, user_email, app_password, subject, sender):
        """
        Send an email alert to the user when an urgent email is detected.
        This creates a notification email and sends it to the user's inbox.
        
        Args:
            user_email (str): User's Gmail address
            app_password (str): User's Gmail app password
            subject (str): Original email subject
            sender (str): Original email sender
            
        Returns:
            bool: True if alert was sent successfully
        """
        # Create alert email body with urgent styling
        email_body = f"""
        🚨 Urgent Email Received! 🚨

        📩 From: {sender}
        📌 Subject: {subject}

        Please check your dashboard for details.
        """
        
        # Create email message
        msg = MIMEText(email_body)
        msg["Subject"] = f"🚨 Urgent Email Alert: {subject}"
        msg["From"] = user_email
        msg["To"] = user_email

        try:
            # Send email using Gmail SMTP server
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(user_email, app_password)
                server.sendmail(user_email, user_email, msg.as_string())
            print("📧 Email alert sent successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to send email alert: {e}")
            return False
    
    def fetch_and_process_emails(self, user_id, user_email, app_password):
        """
        Fetch unread emails from Gmail and process them for urgency detection.
        This is the main email processing function that:
        1. Connects to Gmail IMAP
        2. Fetches unread emails from last 30 days
        3. Classifies each email
        4. Stores urgent emails in database
        5. Sends alert emails for urgent messages
        
        Args:
            user_id (int): Database user ID
            user_email (str): User's Gmail address
            app_password (str): User's Gmail app password
        """
        try:
            # Connect to Gmail IMAP server
            mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
            mail.login(user_email, app_password)
            mail.select("inbox")

            # Search for unread emails from last 30 days
            since_date = (datetime.now() - timedelta(days=30)).strftime('%d-%b-%Y')
            search_criteria = f'(UNSEEN SINCE {since_date})'

            status, email_ids = mail.search(None, search_criteria)
            if status != 'OK':
                return
                
            email_ids = email_ids[0].split()
            
            # Connect to database for storing results
            conn = sqlite3.connect('email_deadlines.db')
            cursor = conn.cursor()

            # Process each unread email
            for e_id in email_ids:
                try:
                    # Fetch email content
                    status, msg_data = mail.fetch(e_id, "(RFC822)")
                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email)

                    # Extract email metadata
                    subject = self.decode_subject(msg.get("Subject", ""))
                    sender = msg.get("From", "")
                    email_date = email.utils.parsedate_to_datetime(msg.get("Date", ""))
                    
                    # Skip if this is our own alert email (prevent infinite loop)
                    if self.is_alert_email(subject, sender, user_email):
                        continue
                    
                    # Check if email was already processed (prevent duplicates)
                    email_message_id = msg.get("Message-ID", f"no-id-{e_id}")
                    cursor.execute("SELECT id FROM processed_emails WHERE email_id = ? AND user_id = ?", 
                                 (email_message_id, user_id))
                    if cursor.fetchone():
                        continue

                    # Extract email body content
                    body = ""
                    if msg.is_multipart():
                        # Handle multipart emails (HTML + text)
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                try:
                                    body = part.get_payload(decode=True).decode(errors="ignore")
                                except:
                                    body = ""
                                break
                    else:
                        # Handle simple text emails
                        try:
                            body = msg.get_payload(decode=True).decode(errors="ignore")
                        except:
                            body = ""

                    # Classify email urgency using ML model
                    urgency = self.classify_email(subject, body)
                    
                    # If email is urgent, store it and send alert
                    if urgency == "Time-sensitive":
                        # Store urgent email in database
                        cursor.execute('''
                            INSERT INTO deadline_emails (user_id, sender, subject, body, email_date)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (user_id, sender, subject, body, email_date))
                        
                        # Send alert email to user
                        self.send_email_alert(user_email, app_password, subject, sender)
                        
                        print(f"🚨 Urgent email detected and stored: {subject}")
                    
                    # Mark email as processed to avoid reprocessing
                    cursor.execute('''
                        INSERT INTO processed_emails (user_id, email_id)
                        VALUES (?, ?)
                    ''', (user_id, email_message_id))
                    
                except Exception as e:
                    print(f"Error processing email {e_id}: {e}")
                    continue

            # Save all database changes
            conn.commit()
            conn.close()
            mail.logout()

        except Exception as e:
            print(f"❌ Error fetching emails: {e}")
    
    def start_monitoring(self):
        """
        Start the continuous email monitoring loop.
        This runs in a separate thread and checks all users' emails periodically.
        """
        self.running = True
        while self.running:
            try:
                # Get all registered users from database
                conn = sqlite3.connect('email_deadlines.db')
                cursor = conn.cursor()
                cursor.execute("SELECT id, email, app_password FROM users")
                users = cursor.fetchall()
                conn.close()
                
                # Process emails for each user
                for user_id, user_email, app_password in users:
                    self.fetch_and_process_emails(user_id, user_email, app_password)
                
                # Wait before next check cycle
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def stop_monitoring(self):
        """Stop the email monitoring loop."""
        self.running = False


# Create a single instance of the email processor for the entire application
email_processor = EmailProcessor()


# FLASK ROUTES - Web Interface

@app.route('/')
def index():
    """
    Home page route - redirects to dashboard if logged in, otherwise to login.
    """
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Login page route - handles user authentication and credential storage.
    
    GET: Display login form
    POST: Process login credentials, test Gmail connection, store in database
    """
    if request.method == 'POST':
        # Get form data
        email_addr = request.form['email']
        app_password = request.form['app_password']
        
        # Test Gmail connection to validate credentials
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
            mail.login(email_addr, app_password)
            mail.logout()
        except Exception as e:
            flash(f'Invalid email credentials: {e}')
            return render_template('login.html')
        
        # Store or update user credentials in database
        conn = sqlite3.connect('email_deadlines.db')
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute("SELECT id FROM users WHERE email = ?", (email_addr,))
        user = cursor.fetchone()
        
        if user:
            # Update existing user's password
            cursor.execute("UPDATE users SET app_password = ? WHERE email = ?", 
                         (app_password, email_addr))
            user_id = user[0]
        else:
            # Create new user record
            cursor.execute("INSERT INTO users (email, app_password) VALUES (?, ?)", 
                         (email_addr, app_password))
            user_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        # Set session variables
        session['user_id'] = user_id
        session['user_email'] = email_addr
        
        flash('Login successful!')
        return redirect(url_for('dashboard'))
    
    # GET request: show login form
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    """
    Dashboard route - displays all detected urgent emails for the logged-in user.
    Shows email details, completion status, and action buttons.
    """
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get user's urgent emails from database
    conn = sqlite3.connect('email_deadlines.db')
    cursor = conn.cursor()
    
    # Fetch all non-deleted urgent emails for the user
    cursor.execute('''
        SELECT id, sender, subject, body, email_date, detected_at, is_completed
        FROM deadline_emails 
        WHERE user_id = ? AND is_deleted = FALSE
        ORDER BY detected_at DESC
    ''', (session['user_id'],))
    
    emails = cursor.fetchall()
    conn.close()
    
    return render_template('dashboard.html', emails=emails)

@app.route('/toggle_complete/<int:email_id>')
def toggle_complete(email_id):
    """
    Toggle completion status of an urgent email.
    Allows users to mark emails as completed or pending.
    
    Args:
        email_id (int): Database ID of the email to toggle
    """
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Update completion status in database
    conn = sqlite3.connect('email_deadlines.db')
    cursor = conn.cursor()
    
    # Toggle the is_completed field
    cursor.execute('''
        UPDATE deadline_emails 
        SET is_completed = NOT is_completed 
        WHERE id = ? AND user_id = ?
    ''', (email_id, session['user_id']))
    
    conn.commit()
    conn.close()
    
    return redirect(url_for('dashboard'))

@app.route('/delete_email/<int:email_id>')
def delete_email(email_id):
    """
    Soft delete an urgent email from the dashboard.
    Marks email as deleted rather than removing from database.
    
    Args:
        email_id (int): Database ID of the email to delete
    """
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Mark email as deleted in database
    conn = sqlite3.connect('email_deadlines.db')
    cursor = conn.cursor()
    
    # Set is_deleted flag to TRUE
    cursor.execute('''
        UPDATE deadline_emails 
        SET is_deleted = TRUE 
        WHERE id = ? AND user_id = ?
    ''', (email_id, session['user_id']))
    
    conn.commit()
    conn.close()
    
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    """
    Logout route - clears user session and redirects to login.
    """
    session.clear()
    flash('Logged out successfully!')
    return redirect(url_for('login'))


# MAIN APPLICATION ENTRY POINT

if __name__ == '__main__':
    # Initialize the database (create tables if they don't exist)
    init_db()
    
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Start email monitoring in a background thread
    # This allows the web server and email monitoring to run simultaneously
    monitoring_thread = threading.Thread(target=email_processor.start_monitoring, daemon=True)
    monitoring_thread.start()
    
    # Print startup information
    print("🚀 Starting Email Deadline Management System...")
    print("📧 Email monitoring will start automatically")
    print("🌐 Web interface will be available at http://localhost:5000")
    
    try:
        # Start the Flask web server
        # debug=True enables auto-reload during development
        # host='0.0.0.0' makes it accessible from other devices on the network
        # threaded=True allows multiple simultaneous requests
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        # Gracefully handle Ctrl+C shutdown
        print("\n⏹️ Stopping email monitoring...")
        email_processor.stop_monitoring()