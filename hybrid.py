import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class HybridRecommender:
    """混合推荐系统（内容+协同过滤）
    新增:
        * save_model(model_dir): 将 TF‑IDF、特征矩阵、SVD 权重保存到 model_dir
        * load_model(model_dir): 从本地加载已训练模型
    """

    def __init__(self):
        # Load datasets
        self.books_df = pd.read_csv('dataset/Books.csv')
        self.ratings_df = pd.read_csv('dataset/Ratings.csv')
        self.users_df = pd.read_csv('dataset/Users.csv')

        # Initialize content-based features
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.book_features = None

        # Initialize collaborative filtering model
        self.cf_model = None

    # ------------------------------------------------------------------ #
    #                   内容过滤: 构建 TF‑IDF 特征                        #
    # ------------------------------------------------------------------ #
    def prepare_content_based_features(self):
        # Fill missing values with an empty string
        self.books_df['Book-Title'] = self.books_df['Book-Title'].fillna('')
        self.books_df['Book-Author'] = self.books_df['Book-Author'].fillna('')
        self.books_df['Publisher'] = self.books_df['Publisher'].fillna('')

        # Combine book features
        self.books_df['content'] = (
            self.books_df['Book-Title'] + ' ' +
            self.books_df['Book-Author'] + ' ' +
            self.books_df['Publisher']
        )

        # Create TF-IDF matrix
        self.book_features = self.tfidf.fit_transform(self.books_df['content'])

    def get_content_based_recommendations(self, book_title, n_recommendations=5):
        # Validate
        if book_title not in self.books_df['Book-Title'].values:
            print(f"Book Title '{book_title}' not found in the dataset.")
            return []

        # Book index
        book_idx = self.books_df[self.books_df['Book-Title'] == book_title].index[0]

        # Fit NearestNeighbors on the fly (lightweight)
        nn_model = NearestNeighbors(n_neighbors=n_recommendations+1,
                                    metric='cosine', algorithm='brute')
        nn_model.fit(self.book_features)

        distances, indices = nn_model.kneighbors(
            self.book_features[book_idx], n_neighbors=n_recommendations+1
        )
        similar_books_indices = indices.flatten()[1:]
        return [self.books_df.iloc[i]['Book-Title'] for i in similar_books_indices]

    # ------------------------------------------------------------------ #
    #                   协同过滤: Surprise SVD                            #
    # ------------------------------------------------------------------ #
    def train_collaborative_filtering(self):
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(
            self.ratings_df[['User-ID', 'ISBN', 'Book-Rating']], reader
        )
        trainset, _ = train_test_split(data, test_size=0.2, random_state=42)

        self.cf_model = SVD(random_state=42)
        self.cf_model.fit(trainset)

    def get_collaborative_recommendations(self, user_id, n_recommendations=5):
        # Predict for every book
        predictions = [
            (isbn, self.cf_model.predict(user_id, isbn).est)
            for isbn in self.books_df['ISBN'].unique()
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_books_isbn = [isbn for isbn, _ in predictions[:n_recommendations]]
        return [
            self.books_df[self.books_df['ISBN'] == isbn]['Book-Title'].values[0]
            for isbn in top_books_isbn
        ]

    # ------------------------------------------------------------------ #
    #                           Hybrid 合并                               #
    # ------------------------------------------------------------------ #
    def get_hybrid_recommendations(self, user_id, book_title,
                                   n_recommendations=5, alpha=0.5):
        content_recs = self.get_content_based_recommendations(
            book_title, n_recommendations*2
        )
        cf_recs = self.get_collaborative_recommendations(
            user_id, n_recommendations*2
        )

        combined = {}
        for r in content_recs:
            combined[r] = alpha
        for r in cf_recs:
            combined[r] = combined.get(r, 0) + (1-alpha)
        sorted_recs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [r for r, _ in sorted_recs[:n_recommendations]]

    # ------------------------------------------------------------------ #
    #                       训练、保存、加载                               #
    # ------------------------------------------------------------------ #
    def train(self):
        print('Preparing content-based features...')
        self.prepare_content_based_features()
        print('Training collaborative filtering model...')
        self.train_collaborative_filtering()
        print('Training complete!')

    def save_model(self, model_dir='models'):
        """保存权重到指定目录"""
        if self.book_features is None or self.cf_model is None:
            raise ValueError('请先调用 train() 再保存模型。')

        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.tfidf, os.path.join(model_dir, 'tfidf.pkl'))
        joblib.dump(self.book_features, os.path.join(model_dir, 'book_features.pkl'))
        with open(os.path.join(model_dir, 'cf_model.pkl'), 'wb') as f:
            pickle.dump(self.cf_model, f)
        print(f'Model saved to {model_dir}/')

    @classmethod
    def load_model(cls, model_dir='model'):
        """从本地目录恢复已训练模型"""
        inst = cls.__new__(cls)  # bypass __init__
        # Need datasets for metadata
        inst.books_df = pd.read_csv('dataset/Books.csv')
        inst.ratings_df = pd.read_csv('dataset/Ratings.csv')
        inst.users_df = pd.read_csv('dataset/Users.csv')

        inst.tfidf = joblib.load(os.path.join(model_dir, 'tfidf.pkl'))
        inst.book_features = joblib.load(os.path.join(model_dir, 'book_features.pkl'))
        with open(os.path.join(model_dir, 'cf_model.pkl'), 'rb') as f:
            inst.cf_model = pickle.load(f)
        return inst

# ------------------------------------------------------------- #
# Example execution                                             #
# ------------------------------------------------------------- #
if __name__ == '__main__':
    rec = HybridRecommender()
    rec.train()
    rec.save_model()  # 保存到 ./model
    print('示例: 推荐结果 ->', rec.get_hybrid_recommendations(1, 'Brave New World'))
