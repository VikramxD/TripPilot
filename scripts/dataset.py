import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import wandb
wandb.require('core')
from datasets import load_dataset
from pydantic_settings import BaseSettings


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class DatasetSettings(BaseSettings):
    topic_modelling_dataset_name: str = "argilla/tripadvisor-hotel-reviews"
    processed_dataset_path: str = "processed_dataset.csv"
    wandb_project: str = "intense_topic_modelling_eda"

class DatasetUtilities:
    def __init__(self, name: str):
        self.name = name
        self.dataset = load_dataset(self.name)

    def process_dataset(self):
        text = self.dataset["train"]["text"]
        label = self.dataset["train"]["prediction"]
        reviews = []
        ratings = []
        for i in range(len(text)):
            t = text[i]
            l = label[i][0]["label"]
            reviews.append(t)
            ratings.append(l)
        df = pd.DataFrame({"review": reviews, "rating": ratings})
        df = df.reset_index(drop=True)
        # Convert 'rating' to numeric type
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        return df

    def export_to_csv(self, path: str):
        df = self.process_dataset()
        return df.to_csv(path, index=False)

    def log_to_wandb(self, df):
        wandb.init(project=DatasetSettings().wandb_project)
        
        wandb.log({"dataset_sample": wandb.Table(dataframe=df.head())})

        self.log_basic_stats(df)
        self.log_rating_distribution(df)
        self.log_review_length_distribution(df)
        self.log_word_cloud(df)
        self.log_ngram_analysis(df)
        self.log_sentiment_analysis(df)
        self.log_topic_modeling(df)
        self.log_rating_correlation(df)

        wandb.finish()
    
    def log_basic_stats(self, df):
        stats = df.describe(include='all').transpose()
        table_data = []
        for col, row in stats.iterrows():
            table_data.append([col] + row.tolist())
        columns = ['column'] + stats.columns.tolist()
        wandb.log({"basic_stats": wandb.Table(columns=columns, data=table_data)})
        wandb.log({
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_types": wandb.Table(columns=["Column", "Type"], 
                                        data=[[col, str(dtype)] for col, dtype in df.dtypes.items()])
        })
        for col in df.select_dtypes(include=['object']):
            unique_values = df[col].nunique()
            top_values = df[col].value_counts().head(5).to_dict()
            wandb.log({
                f"{col}_unique_values": unique_values,
                f"{col}_top_values": wandb.Table(columns=["Value", "Count"], 
                                                 data=[[k, v] for k, v in top_values.items()])
            })

    def log_rating_distribution(self, df):
        plt.figure(figsize=(12, 6))
        sns.countplot(x='rating', data=df, palette='viridis')
        plt.title('Distribution of Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        wandb.log({"rating_distribution": wandb.Image(plt)})
        plt.close()

    def log_review_length_distribution(self, df):
        df['review_length'] = df['review'].str.len()
        plt.figure(figsize=(12, 6))
        sns.histplot(df['review_length'], bins=50, kde=True, color='purple')
        plt.title('Distribution of Review Lengths')
        plt.xlabel('Review Length (characters)')
        plt.ylabel('Count')
        wandb.log({"review_length_distribution": wandb.Image(plt)})
        plt.close()

    def log_word_cloud(self, df):
        stop_words = set(stopwords.words('english'))
        text = ' '.join(df['review'])
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Reviews (Excluding Stop Words)')
        wandb.log({"word_cloud": wandb.Image(plt)})
        plt.close()

    def log_ngram_analysis(self, df, n=2):
        def get_top_ngrams(corpus, n=2, top_k=20):
            vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
            bag_of_words = vec.transform(corpus)
            sum_words = bag_of_words.sum(axis=0) 
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            return words_freq[:top_k]

        top_ngrams = get_top_ngrams(df['review'], n=n)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=[x[1] for x in top_ngrams], y=[x[0] for x in top_ngrams], palette='viridis')
        plt.title(f'Top {len(top_ngrams)} {n}-grams in Reviews')
        plt.xlabel('Frequency')
        plt.ylabel(f'{n}-gram')
        plt.tight_layout()
        wandb.log({f"top_{n}grams": wandb.Image(plt)})
        plt.close()

    def log_sentiment_analysis(self, df):
        df['sentiment'] = df['rating'].apply(lambda x: 'Positive' if x > 3 else ('Negative' if x < 3 else 'Neutral'))
        plt.figure(figsize=(10, 6))
        sns.countplot(x='sentiment', data=df.dropna(subset=['sentiment']), palette='RdYlGn')
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        wandb.log({"sentiment_distribution": wandb.Image(plt)})
        plt.close()

    def log_topic_modeling(self, df, num_topics=5):
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(df['review'])
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(doc_term_matrix)
        feature_names = vectorizer.get_feature_names_out()
        topic_summaries = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            topic_summaries.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
        wandb.log({"topic_modeling": wandb.Table(columns=["Topic", "Top Words"], 
                                                 data=[[f"Topic {i+1}", summary] for i, summary in enumerate(topic_summaries)])})

    def log_rating_correlation(self, df):
        df['review_length'] = df['review'].str.len()
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='rating', y='review_length', data=df)
        plt.title('Review Length vs Rating')
        plt.xlabel('Rating')
        plt.ylabel('Review Length')
        wandb.log({"review_length_vs_rating": wandb.Image(plt)})
        plt.close()

def main():
    settings = DatasetSettings()
    dataset_utils = DatasetUtilities(settings.topic_modelling_dataset_name)
    df = dataset_utils.process_dataset()
    dataset_utils.export_to_csv(path=settings.processed_dataset_path)
    dataset_utils.log_to_wandb(df)

if __name__ == "__main__":
    main()