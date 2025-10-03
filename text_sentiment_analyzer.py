from textblob import TextBlob

# --- Sample Text Data ---
sample_texts = [
    "I absolutely love this new phone; it's the best product I've bought all year!",
    "The customer service was terrible, and the delivery was late. Very disappointing.",
    "The movie was okay, nothing special, but not bad either.",
    "This code snippet is perfect and exactly what I needed."
]

def analyze_sentiment_textblob(text):
    """
    Analyzes the sentiment of a given text using TextBlob.

    TextBlob returns two properties:
    1. Polarity: A float in the range [-1.0, 1.0]. -1.0 is negative, 1.0 is positive.
    2. Subjectivity: A float in the range [0.0, 1.0]. 0.0 is objective, 1.0 is subjective.
    """
    
    # Create a TextBlob object
    analysis = TextBlob(text)
    
    # Get the polarity score
    polarity_score = analysis.sentiment.polarity
    
    # Classify sentiment based on polarity
    if polarity_score > 0.1:
        sentiment = "POSITIVE 😊"
    elif polarity_score < -0.1:
        sentiment = "NEGATIVE 😠"
    else:
        sentiment = "NEUTRAL 😐"
        
    print("-" * 50)
    print(f"Text: '{text}'")
    print(f"  -> Sentiment: {sentiment}")
    print(f"  -> Polarity Score: {polarity_score:.2f}")

# Run the analysis on the sample texts
for text in sample_texts:
    analyze_sentiment_textblob(text)
