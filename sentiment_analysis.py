import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import os

# Create images folder
os.makedirs("images", exist_ok=True)

# Load dataset
df = pd.read_csv("data/IMDB Dataset.csv")

print("First 5 Reviews:")
print(df.head())

# Function to classify sentiment
def get_sentiment(review):
    polarity = TextBlob(review).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Predicted Sentiment"] = df["review"].apply(get_sentiment)

# Display sentiment counts
print("\nSentiment Counts:")
print(df["Predicted Sentiment"].value_counts())

# ----------------------------
# Bar Chart
# ----------------------------
plt.figure(figsize=(6,4))
df["Predicted Sentiment"].value_counts().plot(kind="bar")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("images/sentiment_distribution.png")
plt.show()

# ----------------------------
# Word Cloud
# ----------------------------
text = " ".join(df["review"].astype(str))

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud")
plt.tight_layout()
plt.savefig("images/wordcloud.png")
plt.show()

print("\nSentiment Analysis Completed Successfully!")