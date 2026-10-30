
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_review_text(reviews):
    result = reviews.copy()

    if "review_score" not in result:
        raise ValueError("Reviews need review_score.")

    if "review_text" not in result:
        cols = [
            c for c in [
                "review_comment_title",
                "review_comment_message"
            ] if c in result
        ]

        if not cols:
            raise ValueError("No review text columns found.")

        result["review_text"] = (
            result[cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )

    result["review_score"] = pd.to_numeric(
        result["review_score"], errors="coerce"
    )

    result["review_text"] = (
        result["review_text"].fillna("").astype(str)
    )

    return result[
        result["review_text"].str.len().ge(8)
    ].dropna(subset=["review_score"]).reset_index(drop=True)


def cluster_reviews(
    reviews,
    max_reviews=5000,
    n_clusters=8,
    batch_size=64,
    random_state=42
):
    result = prepare_review_text(reviews)

    if result.empty:
        raise ValueError("No usable review text found.")

    max_reviews = min(int(max_reviews), len(result))

    negative = result[result.review_score <= 2]
    other = result[result.review_score > 2]

    if len(negative) >= max_reviews:
        selected = negative.sample(
            n=max_reviews, random_state=random_state
        )
    else:
        remaining = max_reviews - len(negative)

        selected = pd.concat([
            negative,
            other.sample(
                n=min(remaining, len(other)),
                random_state=random_state
            )
        ], ignore_index=True)

    selected = selected.reset_index(drop=True)

    if len(selected) < 3:
        raise ValueError("At least three reviews are required.")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words=None,
        sublinear_tf=True
    )

    vectors = vectorizer.fit_transform(
        selected["review_text"]
    )

    if vectors.shape[1] == 0:
        raise ValueError("No usable text vocabulary found.")

    k = min(int(n_clusters), len(selected) - 1)

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=max(256, batch_size * 4),
        n_init=3
    )

    selected["cluster_id"] = model.fit_predict(vectors)
    selected["is_negative"] = selected.review_score <= 2

    terms = vectorizer.get_feature_names_out()
    summaries = []

    for cid, group in selected.groupby("cluster_id"):

        center = model.cluster_centers_[int(cid)]
        top_indices = center.argsort()[-5:][::-1]

        keywords = [
            terms[i] for i in top_indices
        ]

        summaries.append({
            "cluster_id": int(cid),
            "review_count": len(group),
            "negative_count": int(group.is_negative.sum()),
            "negative_rate_pct": round(
                group.is_negative.mean() * 100, 2
            ),
            "average_score": round(
                group.review_score.mean(), 2
            ),
            "top_keywords": ", ".join(keywords),
            "example_reviews": " | ".join(
                group.review_text.drop_duplicates()
                .head(3).tolist()
            )
        })

    summary = pd.DataFrame(summaries)

    summary = summary.sort_values(
        ["negative_count", "review_count"],
        ascending=False
    ).reset_index(drop=True)

    return selected, summary
