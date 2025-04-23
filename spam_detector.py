import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import random

# Function to generate spam and ham data
def generate_spam_ham_data(num_samples):
    spam_templates = [
        "Congratulations! You've won a ${amount} gift card. Click {link} to claim your prize.",
        "Earn ${amount} per month from home! No experience required. Visit {link} now.",
        "You've been selected for a free {item}. Claim it fast at {link}.",
        "Limited time offer: Buy one {item}, get one free! Check it out at {link}.",
        "You're a lucky winner of our {type} lottery! Claim your prize today.",
        "Don't miss out on exclusive deals! Visit {link} for more details.",
        "Your account shows suspicious activity. Verify now at {link}.",
        "Please provide your OTP for the following",
        "New message from {sender}. Open it immediately at {link}.",
        "Act quickly! Your chance to win {item} ends soon.",
        "Work from home and earn ${amount} quickly! Details at {link}."
    ]

    ham_templates = [
        "Hi {name}, I hope this email finds you well. Let's catch up soon.",
        "Dear {name}, can we schedule a call to discuss the {topic} project?",
        "Hi {name}, just wanted to confirm our meeting on {date}. Let me know if the time still works.",
        "Come home by 9 pm",
        "I am really sorry, I missed you today",
        "I love you so much",
        "Please come to class on time from now on",
        "Please find the attached document related to our {topic} discussion.",
        "Thank you for your response, {name}. I appreciate your support.",
        "Hi {name}, looking forward to discussing project updates during our next meeting.",
        "Good morning {name}, hope you are having a productive day.",
        "Hello {name}, can you let me know your availability for a quick chat?",
        "Dear {name}, please share your thoughts on the attached proposal.",
        "Hi {name}, here's the update regarding the {topic} you requested."
    ]

    spam_messages = [
        template.format(
            amount=random.randint(100, 1000),
            item=random.choice(["iPhone", "TV", "Laptop", "Headphones"]),
            type=random.choice(["mega", "monthly", "holiday"]),
            sender=random.choice(["unknown sender", "support team"]),
            link="www.example.com"
        ) for template in spam_templates for _ in range(10)
    ]

    ham_messages = [
        template.format(
            name=random.choice(["Alice", "Bob", "Charlie", "David"]),
            topic=random.choice(["budget", "marketing", "sales"]),
            date=random.choice(["Monday", "Wednesday", "Friday"])
        ) for template in ham_templates for _ in range(10)
    ]

    data = {
        'message': [],
        'label': []
    }

    all_messages = [(msg, 'spam') for msg in spam_messages] + [(msg, 'ham') for msg in ham_messages]
    random.shuffle(all_messages)

    for message, label in all_messages[:num_samples]:
        data['message'].append(message)
        data['label'].append(label)

    return pd.DataFrame(data)

# Generate dataset with 1000 samples
df = generate_spam_ham_data(1000)
df = df.sample(frac=1).reset_index(drop=True)

# Initialize TF-IDF Vectorizer and fit-transform the data
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['message'])
model = MultinomialNB()
model.fit(X_tfidf, df['label'])

# Global dictionary to track message occurrences
message_frequency = {}

def classify_email_with_repetition(email_text):
    """
    Classifies an email as 'spam' or 'ham', considering repetition.
    """
    # Update the frequency of the message
    if email_text in message_frequency:
        message_frequency[email_text] += 1
    else:
        message_frequency[email_text] = 1

    # Check if the message is repeated
    if message_frequency[email_text] > 1:
        return 'spam'  # Mark repeated messages as spam

    # Otherwise, use the ML model's prediction
    email_tfidf = vectorizer.transform([email_text])
    prediction = model.predict(email_tfidf)
    return prediction[0]

def classify_emails_with_repetition():
    """
    Classifies emails and identifies repetitive messages.
    """
    input_text = input_box.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showerror("Error", "Please enter email messages to classify.")
        return

    emails = input_text.split("\n")
    results = [(email.strip(), classify_email_with_repetition(email.strip())) for email in emails if email.strip()]
    ham_emails = [email for email, label in results if label == 'ham']
    spam_emails = [email for email, label in results if label == 'spam']

    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, "Ham Emails (Top):\n")
    output_box.insert(tk.END, "\n".join(ham_emails) + "\n\n")
    output_box.insert(tk.END, "Spam Emails (Bottom):\n")
    output_box.insert(tk.END, "\n".join(spam_emails))

# Main Tkinter window
root = tk.Tk()
root.title("Email Spam Detection")
root.geometry("800x600")  # Set the window size

# Create a frame for input area
input_frame = ttk.Frame(root)
input_frame.pack(pady=10, padx=20, fill=tk.X)

# Input box label
ttk.Label(input_frame, text="Enter Emails (one per line):").pack(pady=5, anchor='w')

# Text box for email input
input_box = scrolledtext.ScrolledText(input_frame, width=80, height=10)
input_box.pack(pady=5)

# Button to classify emails
classify_button = ttk.Button(input_frame, text="Classify Emails", command=classify_emails_with_repetition)
classify_button.pack(pady=10)

# Output area frame
output_frame = ttk.Frame(root)
output_frame.pack(pady=10, padx=20, fill=tk.X)

# Output box label
ttk.Label(output_frame, text="Classification Results:").pack(pady=5, anchor='w')

# Text box for displaying the results
output_box = scrolledtext.ScrolledText(output_frame, width=80, height=10, state='normal')
output_box.pack(pady=5)

# Run the application
root.mainloop()
