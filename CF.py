import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

class BookRecommender:
    def __init__(self):
        self.books_df = None
        self.ratings_df = None
        self.popular_books = None
        self.user_item_matrix = None
        self.item_features = None
        self.knn_model = None
        self.valid_users = None
        self.scaler = MinMaxScaler()
        self.is_full_data_loaded = False
        self.item_user_matrix = None
        
    def load_initial_data(self, ratings_path, books_path):
        """加载初始数据，包括图书信息和评分数据"""
        try:
            # 加载图书数据
            self.books_df = pd.read_csv(books_path, low_memory=False)
            
            # 加载评分数据（只读取需要的列）
            ratings = pd.read_csv(ratings_path, usecols=['ISBN', 'User-ID', 'Book-Rating'])
            
            # 过滤掉评分为0的记录
            ratings = ratings[ratings['Book-Rating'] > 0]
            
            # 计算每本书的评分统计
            rating_stats = ratings.groupby('ISBN').agg({
                'Book-Rating': ['count', 'mean']
            }).reset_index()
            rating_stats.columns = ['ISBN', 'rating_count', 'avg_rating']
            
            # 计算流行度分数
            rating_stats['popularity_score'] = (
                rating_stats['rating_count'] * 0.7 + 
                rating_stats['avg_rating'] * 0.3
            )
            
            # 合并图书信息和评分统计
            self.books_df = pd.merge(
                self.books_df, 
                rating_stats,
                on='ISBN',
                how='left'
            )
            
            # 填充缺失值
            self.books_df['rating_count'] = self.books_df['rating_count'].fillna(0)
            self.books_df['avg_rating'] = self.books_df['avg_rating'].fillna(0)
            self.books_df['popularity_score'] = self.books_df['popularity_score'].fillna(0)
            
            # 预计算热门图书
            self.popular_books = self.books_df.nlargest(50, 'popularity_score')
            
            return True
            
        except Exception as e:
            print(f"Error loading initial data: {str(e)}")
            return False
            
    def load_full_data(self, ratings_path):
        """加载完整评分数据并构建用户-物品矩阵"""
        try:
            # 加载完整评分数据
            self.ratings_df = pd.read_csv(ratings_path)
            
            # 构建用户-物品矩阵
            self.user_item_matrix = self.ratings_df.pivot(
                index='User-ID',
                columns='ISBN',
                values='Book-Rating'
            ).fillna(0)
            
            # 构建物品-用户矩阵（转置）
            self.item_user_matrix = self.user_item_matrix.T
            
            self.is_full_data_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading full data: {str(e)}")
            return False
            
    def _build_item_features(self):
        """构建图书特征矩阵"""
        try:
            # 1. 提取年份特征
            self.books_df['year'] = pd.to_numeric(
                self.books_df['Year-Of-Publication'],
                errors='coerce'
            )
            year_mean = self.books_df['year'].mean()
            self.books_df['year'] = self.books_df['year'].fillna(year_mean)
            
            # 2. 标准化特征
            features = ['year', 'avg_rating', 'rating_count']
            self.item_features = self.books_df[features].values
            self.item_features = self.scaler.fit_transform(self.item_features)
            
            # 3. 初始化KNN模型
            self.knn_model = NearestNeighbors(
                n_neighbors=10,
                metric='cosine',
                algorithm='brute'
            )
            self.knn_model.fit(self.item_features)
            
        except Exception as e:
            print(f"特征构建错误: {str(e)}")
            self.item_features = None
            self.knn_model = None
            
    def search_by_title(self, title, n=12):
        """按书名搜索图书"""
        try:
            # 不区分大小写的模糊搜索
            mask = self.books_df['Book-Title'].str.lower().str.contains(title.lower())
            results = self.books_df[mask].nlargest(n, 'popularity_score')
            results['method'] = 'Title Search'
            return results
        except Exception as e:
            print(f"Title search error: {str(e)}")
            return pd.DataFrame()
    
    def search_by_author(self, author, n=12):
        """按作者搜索图书"""
        try:
            # 不区分大小写的模糊搜索，并处理NaN值
            # 首先过滤掉Book-Author为NaN的行
            valid_authors = self.books_df.dropna(subset=['Book-Author'])
            
            # 然后在有效的作者数据中搜索
            mask = valid_authors['Book-Author'].str.lower().str.contains(str(author).lower())
            results = valid_authors[mask].nlargest(n, 'popularity_score').copy()  # 使用copy()避免警告
            results['method'] = 'Author Search'
            return results
        except Exception as e:
            print(f"Author search error: {str(e)}")
            return pd.DataFrame()
        
    def recommend_by_title(self, book_title, n=6):
        """
        基于书名推荐相似图书，返回DataFrame（含书名、作者、图片等）
        """
        try:
            if self.user_item_matrix is None:
                print('user_item_matrix未初始化')
                return pd.DataFrame()
            pt = self.user_item_matrix
            if book_title not in pt.index:
                print(f'书名 {book_title} 不在user_item_matrix中')
                return pd.DataFrame()
            # 计算相似度
            similarity_score = cosine_similarity(pt)
            index = np.where(pt.index == book_title)[0][0]
            similar_items = sorted(list(enumerate(similarity_score[index])),
                                    key=lambda x: x[1], reverse=True)[1:n+1]
            rec_titles = [pt.index[i[0]] for i in similar_items]
            # 查找详细信息
            books = self.books_df[self.books_df['Book-Title'].isin(rec_titles)].copy()
            books['method'] = 'Title-Similar'
            return books
        except Exception as e:
            print(f'基于书名推荐出错: {e}')
            return pd.DataFrame()
        
    def get_similar_books(self, book_id, n=8):
        """获取相似图书推荐"""
        try:
            if not self.is_full_data_loaded:
                return self.get_popular_books(n)
            
            # 获取目标图书的用户评分向量
            target_vector = self.item_user_matrix.loc[book_id]
            
            # 计算与其他图书的相似度
            similarities = cosine_similarity(
                target_vector.values.reshape(1, -1),
                self.item_user_matrix.values
            )[0]
            
            # 获取相似度最高的图书
            similar_indices = np.argsort(similarities)[-n-1:-1][::-1]
            similar_isbns = self.item_user_matrix.index[similar_indices]
            
            # 获取相似图书的详细信息
            similar_books = self.books_df[self.books_df['ISBN'].isin(similar_isbns)]
            similar_books['method'] = 'Similar Books'
            
            return similar_books
            
        except Exception as e:
            print(f"Similar books error: {str(e)}")
            return self.get_popular_books(n)
    
    def get_popular_books(self, n=12):
        """获取热门图书"""
        try:
            if self.popular_books is None:
                return pd.DataFrame()
            
            result = self.popular_books.head(n).copy()
            result['method'] = 'Popular'
            return result
            
        except Exception as e:
            print(f"Popular books error: {str(e)}")
            return pd.DataFrame()
            
    def recommend(self, user_id=None, book_id=None, n=5):
        """主推荐方法"""
        try:
            # 1. 如果没有提供任何ID，返回热门图书
            if user_id is None and book_id is None:
                return self.get_popular_books(n)
                
            # 2. 如果需要个性化推荐，确保完整数据已加载
            if not self.is_full_data_loaded:
                return self.get_popular_books(n)
                
            # 3. 基于用户的推荐
            if user_id is not None:
                if user_id in self.valid_users:
                    return self._user_based_recommend(user_id, n)
                else:
                    # 新用户，读取行为数据
                    import sys, os
                    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                    from app import get_user_behavior_csv, get_cold_start_books
                    behavior = get_user_behavior_csv(user_id)
                    likes = behavior['like']
                    if likes:
                        recs = []
                        import random
                        for like in random.sample(likes, min(2, len(likes))):
                            similar = self.get_similar_books(like, n=n//2)
                            if similar is not None and not similar.empty:
                                recs.append(similar)
                        if recs:
                            combined = pd.concat(recs).drop_duplicates(subset=['ISBN'])
                            combined['method'] = 'Similar-to-Like'
                            return combined.head(n)
                    # 没有收藏，冷启动
                    cold_books = get_cold_start_books(n)
                    cold_books['method'] = 'Cold-Start'
                    return cold_books.head(n)
                
            # 4. 基于内容的推荐
            if book_id is not None and book_id in self.books_df['ISBN'].values:
                return self._content_based_recommend(book_id, n)
                
            # 5. 默认返回热门图书
            return self.get_popular_books(n)
            
        except Exception as e:
            print(f"推荐错误: {str(e)}")
            return self.get_popular_books(n)
            
    def _user_based_recommend(self, user_id, n=5):
        """基于用户的协同过滤推荐"""
        try:
            # 获取用户评分向量
            user_ratings = self.user_item_matrix.loc[user_id]
            
            # 计算用户相似度
            user_similarities = self.user_item_matrix.dot(user_ratings) / (
                np.linalg.norm(self.user_item_matrix, axis=1) * 
                np.linalg.norm(user_ratings)
            )
            
            # 获取最相似的用户
            similar_users = user_similarities.nlargest(6).index[1:]
            similar_ratings = self.user_item_matrix.loc[similar_users]
            
            # 计算预测评分
            pred_ratings = similar_ratings.mean()
            pred_ratings[user_ratings > 0] = 0
            
            # 获取推荐图书
            recommended_books = self.books_df[
                self.books_df['ISBN'].isin(pred_ratings.nlargest(n).index)
            ].copy()
            recommended_books['method'] = 'User-CF'
            
            return recommended_books
            
        except Exception as e:
            print(f"用户推荐错误: {str(e)}")
            return self.get_popular_books(n)
            
    def _content_based_recommend(self, book_id, n=5):
        """基于内容的推荐"""
        try:
            if self.knn_model is None:
                return self.get_popular_books(n)
                
            # 获取图书索引
            book_idx = self.books_df[self.books_df['ISBN'] == book_id].index
            if len(book_idx) == 0:
                return self.get_popular_books(n)
                
            # 获取相似图书
            book_features = self.item_features[book_idx[0]].reshape(1, -1)
            distances, indices = self.knn_model.kneighbors(book_features)
            
            # 返回推荐结果
            similar_books = self.books_df.iloc[indices[0][1:n+1]].copy()
            similar_books['similarity'] = 1 - distances[0][1:n+1]
            similar_books['method'] = 'Content'
            
            return similar_books
            
        except Exception as e:
            print(f"内容推荐错误: {str(e)}")
            return self.get_popular_books(n)
            
    def get_user_history(self, user_id):
        """获取用户阅读历史"""
        try:
            if not self.is_full_data_loaded:
                return pd.DataFrame()
            
            # 获取用户评分记录
            user_ratings = self.ratings_df[self.ratings_df['User-ID'] == user_id]
            
            # 合并图书信息
            history = pd.merge(
                user_ratings,
                self.books_df,
                on='ISBN',
                how='left'
            )
            
            return history.sort_values('Book-Rating', ascending=False)
            
        except Exception as e:
            print(f"User history error: {str(e)}")
            return pd.DataFrame()

    def get_books_by_isbn(self, isbns):
        """通过ISBN列表获取图书信息"""
        try:
            if self.books_df is None or isbns is None or len(isbns) == 0:
                return pd.DataFrame()
            
            # 确保isbns是列表形式
            if not isinstance(isbns, list):
                isbns = [isbns]
                
            # 查找匹配ISBN的图书
            books = self.books_df[self.books_df['ISBN'].isin(isbns)].copy()  # 使用copy()避免SettingWithCopyWarning
            
            # 添加分类信息（如果没有，使用空值）
            if 'Category' not in books.columns:
                books['Category'] = books['Book-Author']  # 临时使用作者名作为分类
                
            return books
        except Exception as e:
            print(f"获取图书信息出错: {str(e)}")
            return pd.DataFrame()
