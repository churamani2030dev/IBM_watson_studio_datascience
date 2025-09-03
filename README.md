# IBM_watson_studio_datascience

IBM Watson Studio Community Recommendation Engine

## Project Description

This project builds a full recommendation engine for the IBM Watson Studio community by mining historical user–article interactions and producing suggestions three ways—popularity (rank-based), user–user collaborative filtering, content-based from article text—and then a matrix-factorization model with SVD; you start with EDA to understand the dataset and compute the rubric metrics (median_val, user_article_interactions, max_views_by_user, max_views, most_viewed_article_id, unique_articles, unique_users, total_articles) to verify you’re reading and aggregating correctly, then implement a rank-based recommender that simply sorts articles by total interactions and returns the top IDs and names (this is your cold-start fallback), next create a user–item matrix with users as rows, articles as columns, and 1/0 flags for interactions and write helpers to find similar users (e.g., cosine similarity on rows), union their interacted articles, drop the current user’s history, and rank recommendations by count (tie-break by global popularity); improve this by biasing toward neighbors with more interactions and by re-ranking within the candidate set by article frequency so heavy users get better lists, add a pathway to handle new users by falling back to the rank-based top-N; for content-based recommendations, vectorize article text with TF-IDF, use KMeans to cluster articles (pick k via elbow or silhouette), and recommend items that share a cluster or maximize cosine similarity to a query article; finally, perform SVD on the user–item matrix (after centering/normalizing as needed), explain how many latent features you keep (using an explained-variance or performance curve), and use the reduced U, Σ, Vᵀ to compute article–article similarity in latent space to produce predictions for users and direct item-to-item suggestions, then discuss results and how you’d evaluate in practice (e.g., hit-rate@k, MAP@k, offline splits); package all of this in a clean notebook or script with necessary documentation.

## Getting Started

These instructions will help you set up and run the project on your local machine or a platform like Google Colab.

### Prerequisites

*   Python 3.6+
*   Jupyter Notebook or JupyterLab (if running locally)
*   Required Python libraries (listed in `requirements.txt` or below)

### Installation

1.  Clone the repository:
