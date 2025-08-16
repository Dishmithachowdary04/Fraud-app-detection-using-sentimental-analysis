import tkinter as tk
from tkinter import messagebox
from google_play_scraper import app as playstore_app, reviews, Sort
from textblob import TextBlob
from PIL import Image, ImageTk
import requests
from io import BytesIO
import numpy as np

# ✅ Set your SerpAPI key here
SERPAPI_KEY = "YOUR_SERPAPI_KEY"

def analyze_sentiment(reviews):
    return [TextBlob(review).sentiment.polarity for review in reviews]

def classify_app(sentiments):
    avg = np.mean(sentiments)
    return ("Fraud 🚫", "The app has mostly negative reviews. Avoid using it.") if avg < 0 else \
           ("Safe ✅", "The app has mostly positive reviews. It is safe to use.")

def extract_app_id(link):
    try:
        return link.strip().split("id=")[1].split("&")[0]
    except:
        return None

class FraudAppDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fraud App Detection")
        self.root.state("zoomed")

        # Background Image
        self.bg_image = Image.open("C:/Users/91939/OneDrive/Documents/Desktop/PCL/pic 1.jpg")
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)
        self.bg_label = tk.Label(root, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.label = tk.Label(root, text="Fraud App Detection Using Sentiment Analysis",
                              font=("times", 22, "bold"), fg="white", bg="black")
        self.label.pack(pady=10)

        self.top_frame = tk.Frame(root, bg="black")
        self.top_frame.pack(pady=10)

        self.logo_label = tk.Label(self.top_frame, bg="black")
        self.logo_label.pack(side=tk.LEFT, padx=10)

        self.app_name_label = tk.Label(self.top_frame, text="", font=("times", 20, "bold"), fg="yellow", bg="black")
        self.app_name_label.pack(side=tk.LEFT, padx=10)

        self.link_label = tk.Label(root, text="Enter Google Play Store App Link:",
                                   font=("times", 18), fg="white", bg="black")
        self.link_label.pack(pady=5)

        self.link_entry = tk.Entry(root, width=55, font=("times", 16))
        self.link_entry.pack(pady=10)

        self.analyze_button = tk.Button(root, text="Analyze App", command=self.analyze_app,
                                        font=("times", 14, "bold"), bg="#28B463", fg="white", padx=15, pady=8)
        self.analyze_button.pack(pady=20)

        self.result_label = tk.Label(root, text="", font=("times", 20, "bold"), fg="white", bg="black")
        self.result_label.pack(pady=15)

        self.reason_label = tk.Label(root, text="", font=("times", 18), fg="white", bg="black")
        self.reason_label.pack(pady=10)

        self.positive_label = tk.Label(root, text="", font=("times", 18, "bold"), fg="#008000", bg="black")
        self.positive_label.pack(pady=10)

        self.negative_label = tk.Label(root, text="", font=("times", 18, "bold"), fg="#FF0000", bg="black")
        self.negative_label.pack(pady=10)

        self.similar_label = tk.Label(root, text="", font=("times", 16, "bold"), fg="cyan", bg="black", justify="left")
        self.similar_label.pack(pady=15)

        self.root.bind("<Configure>", self.resize_background)

    def analyze_app(self):
        app_link = self.link_entry.get().strip()
        app_id = extract_app_id(app_link)

        if not app_id:
            messagebox.showerror("Error", "Invalid Google Play Store link! Must contain 'id=com.example.app'.")
            return

        try:
            app_details = playstore_app(app_id)
            app_name = app_details["title"]
            app_logo_url = app_details["icon"]

            self.app_name_label.config(text=app_name)

            app_reviews, _ = reviews(app_id, count=100, sort=Sort.NEWEST)
            review_texts = [r["content"] for r in app_reviews]

            sentiments = analyze_sentiment(review_texts)
            status, reason = classify_app(sentiments)

            pos = sum(1 for s in sentiments if s >= 0)
            neg = len(sentiments) - pos

            self.result_label.config(text=f"App Classification: {status}")
            self.reason_label.config(text=f"Reason: {reason}")
            self.positive_label.config(text=f"Positive Reviews: {pos} 😊")
            self.negative_label.config(text=f"Negative Reviews: {neg} 😠")

            self.load_app_logo(app_logo_url)
            self.show_similar_apps(app_name)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze app: {e}")

    def load_app_logo(self, url):
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).resize((150, 150), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.logo_label.config(image=photo)
            self.logo_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load logo: {e}")

    def show_similar_apps(self, app_name):
        try:
            url = f"https://serpapi.com/search.json"
            params = {
                "engine": "google_play",
                "q": f"apps similar to {app_name}",
                "hl": "en",
                "api_key": SERPAPI_KEY
            }
            response = requests.get(url, params=params)
            data = response.json()

            apps = data.get("organic_results", [])[:5]
            if not apps:
                self.similar_label.config(text="No recommendations found.")
                return

            formatted = "Recommended Similar Apps:\n" + "\n".join(f"• {app['title']}" for app in apps)
            self.similar_label.config(text=formatted)
        except Exception as e:
            self.similar_label.config(text="Could not fetch similar apps.")

    def resize_background(self, event):
        new_width, new_height = event.width, event.height
        self.bg_resized = self.bg_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.bg_resized)
        self.bg_label.config(image=self.bg_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = FraudAppDetectionApp(root)
    root.mainloop()
