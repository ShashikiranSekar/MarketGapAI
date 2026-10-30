import pandas as pd

def build_market_snapshot(reviews,category_summary=None,focus=""):
    result={"focus":focus,"review_count":len(reviews)}
    if "sentiment" in reviews:
        result["negative_reviews"]=int(reviews["sentiment"].eq("Negative").sum())
    if "has_comment" in reviews:
        result["reviews_with_text"]=int(reviews["has_comment"].sum())
    if category_summary is not None and not category_summary.empty:
        result["top_categories"]=category_summary.head(10).to_dict("records")
    return result


def build_competitor_comparison(table):
    required = {"Competitor", "Feature", "Status"}

    if not required.issubset(table.columns):
        raise ValueError(
            "Need Competitor, Feature, Status columns."
        )

    data = table.copy()

    for column in required:
        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    data = data[
        (data["Competitor"] != "") &
        (data["Feature"] != "")
    ]

    data["Status"] = data["Status"].replace(
        "", "Unknown"
    )

    data = data.drop_duplicates(
        ["Competitor", "Feature"],
        keep="last"
    )

    if data.empty:
        return pd.DataFrame(columns=["Feature"])

    result = data.pivot(
        index="Feature",
        columns="Competitor",
        values="Status"
    ).reset_index()

    result = result.fillna("Unknown")

    return result

def score_opportunities(category_summary):
    if category_summary is None or category_summary.empty:return pd.DataFrame()
    out=category_summary.copy()
    out["rank"]=range(1,len(out)+1)
    out["evidence_strength"]="Exploratory"
    out["interpretation"]="Investigate; not validated demand"
    return out

def make_opportunity_briefs(category_summary,cluster_summary,focus=""):
    briefs=[]
    if category_summary is None or category_summary.empty:return briefs
    for _,row in category_summary.head(10).iterrows():
        briefs.append({
            "Opportunity hypothesis":f"Investigate unmet customer needs in {row['category']}",
            "Focus":focus or "Customer experience and product fit",
            "Evidence":f"{int(row['review_product_associations'])} review-product associations; negative rate {row['negative_rate_pct']}%; average rating {row['average_review_score']}",
            "Signal score":row["opportunity_signal_score"],
            "Suggested validation":"Review a sample of complaints, interview customers, and test a small product/service improvement.",
            "Confidence":"Low to exploratory",
        })
    return briefs
