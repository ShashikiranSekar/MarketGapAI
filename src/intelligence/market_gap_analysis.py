import pandas as pd

def attach_product_categories(clustered_reviews,orders,order_items,products,translations):
    specs=[("reviews",clustered_reviews,{"order_id","review_score","cluster_id"}),("items",order_items,{"order_id","product_id"}),("products",products,{"product_id","product_category_name"}),("translations",translations,{"product_category_name","product_category_name_english"})]
    for name,df,required in specs:
        missing=required-set(df.columns)
        if missing:raise ValueError(f"{name} missing columns: {sorted(missing)}")
    cols=[c for c in ["order_id","review_score","cluster_id","review_text","is_negative"] if c in clustered_reviews]
    out=clustered_reviews[cols].copy()
    out=out.merge(order_items[["order_id","product_id"]].drop_duplicates(),on="order_id",how="left")
    out=out.merge(products[["product_id","product_category_name"]].drop_duplicates("product_id"),on="product_id",how="left")
    out=out.merge(translations[["product_category_name","product_category_name_english"]].drop_duplicates("product_category_name"),on="product_category_name",how="left")
    out["category"]=out["product_category_name_english"].fillna(out["product_category_name"]).fillna("Unknown")
    out["is_negative"]=out["review_score"]<=2
    return out

def summarize_category_opportunities(mapped_reviews,min_associations=20):
    if mapped_reviews.empty:return pd.DataFrame()
    out=mapped_reviews.groupby("category").agg(review_product_associations=("review_score","size"),negative_associations=("is_negative","sum"),average_review_score=("review_score","mean")).reset_index()
    out["negative_rate_pct"]=(out["negative_associations"]/out["review_product_associations"]*100).round(2)
    out["average_review_score"]=out["average_review_score"].round(2)
    out=out[out["review_product_associations"]>=min_associations].copy()
    if out.empty:return out
    nmax=max(float(out["negative_associations"].max()),1)
    rmax=max(float(out["negative_rate_pct"].max()),1)
    out["opportunity_signal_score"]=(.5*out["negative_associations"]/nmax+.5*out["negative_rate_pct"]/rmax).mul(100).round(2)
    return out.sort_values("opportunity_signal_score",ascending=False).reset_index(drop=True)
