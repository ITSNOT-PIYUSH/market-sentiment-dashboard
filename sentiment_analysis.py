import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

class StockSentimentAnalyzer:
    def __init__(self):
        # Load FinBERT model and tokenizer - pretrained model for financial sentiment analysis
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.labels = ["negative", "neutral", "positive"]
    
    def analyze_sentiment(self, texts):
        """
        Analyze sentiment of a list of texts related to stocks
        
        Args:
            texts (list): List of text strings to analyze
            
        Returns:
            dict: Contains sentiment scores and overall sentiment
        """
        if not texts:
            return {"sentiment": "neutral", "scores": {"negative": 0.33, "neutral": 0.34, "positive": 0.33}}
            
        # Tokenize texts
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            
        # Calculate average sentiment scores across all texts
        avg_scores = scores.mean(dim=0)
        avg_scores_dict = {self.labels[i]: float(avg_scores[i]) for i in range(len(self.labels))}
        
        # Get overall sentiment
        overall_sentiment = self.labels[torch.argmax(avg_scores).item()]
        
        return {
            "sentiment": overall_sentiment,
            "scores": avg_scores_dict
        }
        
    def get_sentiment_color(self, sentiment):
        """Return appropriate color for each sentiment category"""
        color_map = {
            "positive": "green",
            "neutral": "grey",
            "negative": "red"
        }
        return color_map.get(sentiment, "grey")