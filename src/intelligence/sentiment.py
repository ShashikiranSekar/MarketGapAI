import pandas as pd

def classify_review_score(score):
    if pd.isna(score): return "Unknown"
    score=float(score)
    if score<=2:return "Negative"
    if score==3:return "Neutral"
    if score<=5:return "Positive"
    return "Unknown"

def analyze_sentiment(reviews):
    result=reviews.copy()
    if "review_score" not in result: raise ValueError("order_reviews must contain review_score.")
    result["review_score"]=pd.to_numeric(result["review_score"],errors="coerce")
    result["sentiment"]=result["review_score"].apply(classify_review_score)
    cols=[c for c in ["review_comment_title","review_comment_message"] if c in result]
    result["review_text"]=result[cols].fillna("").astype(str).agg(" ".join,axis=1).str.replace(r"\s+"," ",regex=True).str.strip() if cols else ""
    result["has_comment"]=result["review_text"].ne("")
    return result
