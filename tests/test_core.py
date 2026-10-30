import pandas as pd
from src.intelligence.sentiment import analyze_sentiment
from src.intelligence.pain_points import detect_pain_points, summarize_pain_points
from src.intelligence.market_research import build_competitor_comparison
from src.intelligence.market_gap_analysis import summarize_category_opportunities

def test_rating_sentiment_and_pain_points():
    df=pd.DataFrame({"review_score":[1,3,5],"review_comment_title":["Late","Okay","Great"],"review_comment_message":["late delivery","average","excellent"]})
    result=detect_pain_points(analyze_sentiment(df))
    assert result["sentiment"].tolist()==["Negative","Neutral","Positive"]
    assert result.iloc[0]["has_pain_point"]

def test_competitor_comparison():
    df=pd.DataFrame([{"Competitor":"A","Feature":"Returns","Status":"1"},{"Competitor":"B","Feature":"Returns","Status":""}])
    out=build_competitor_comparison(df)
    assert out.loc[0,"A"]=="1"
    assert out.loc[0,"B"]=="Unknown"

def test_opportunity_score_bounds():
    df=pd.DataFrame({"category":["x","x","y","y"],"review_score":[1,5,2,2],"is_negative":[True,False,True,True]})
    out=summarize_category_opportunities(df,min_associations=1)
    assert out["opportunity_signal_score"].between(0,100).all()
