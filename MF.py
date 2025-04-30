import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

class MatrixFactorizationRecommender:
    def __init__(self, n_factors=20, learning_rate=0.01, regularization=0.02, n_epochs=20):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
        self.user_map = None
        self.item_map = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.model_path = 'models/mf_model.pkl'
        self.surprise_model = None
        
    def load_data(self, ratings_path):
        """加载评分数据并准备训练"""
        try:
            # 加载评分数据
            ratings = pd.read_csv(ratings_path)
            
            # 过滤掉评分为0的记录
            ratings = ratings[ratings['Book-Rating'] > 0]
            
            # 创建用户和物品的映射
            self.user_map = {user: idx for idx, user in enumerate(ratings['User-ID'].unique())}
            self.item_map = {item: idx for idx, item in enumerate(ratings['ISBN'].unique())}
            
            # 初始化模型参数
            n_users = len(self.user_map)
            n_items = len(self.item_map)
            
            self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
            self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
            self.user_bias = np.zeros(n_users)
            self.item_bias = np.zeros(n_items)
            self.global_bias = ratings['Book-Rating'].mean()
            
            # 准备训练数据
            self.train_data = np.array([
                [self.user_map[row['User-ID']], 
                 self.item_map[row['ISBN']], 
                 row['Book-Rating']]
                for _, row in ratings.iterrows()
            ])
            
            # 准备Surprise格式的数据
            reader = Reader(rating_scale=(1, 10))
            self.surprise_data = Dataset.load_from_df(
                ratings[['User-ID', 'ISBN', 'Book-Rating']], 
                reader
            )
            
            return True
            
        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            return False
            
    def train(self, use_surprise=True):
        """训练矩阵分解模型"""
        try:
            if not hasattr(self, 'train_data'):
                print("请先加载数据")
                return False
                
            if use_surprise:
                # 使用Surprise库训练
                trainset, testset = train_test_split(self.surprise_data, test_size=0.2)
                self.surprise_model = SVD(
                    n_factors=self.n_factors,
                    n_epochs=self.n_epochs,
                    lr_all=self.learning_rate,
                    reg_all=self.regularization
                )
                self.surprise_model.fit(trainset)
                self.is_trained = True
                
                # 保存模型
                self._save_model()
                return True
            else:
                # 使用自定义实现训练
                for epoch in range(self.n_epochs):
                    np.random.shuffle(self.train_data)
                    
                    for user_idx, item_idx, rating in self.train_data:
                        # 计算预测误差
                        prediction = self.predict_rating(user_idx, item_idx)
                        error = rating - prediction
                        
                        # 更新参数
                        self.user_bias[user_idx] += self.learning_rate * (error - self.regularization * self.user_bias[user_idx])
                        self.item_bias[item_idx] += self.learning_rate * (error - self.regularization * self.item_bias[item_idx])
                        
                        self.user_factors[user_idx] += self.learning_rate * (
                            error * self.item_factors[item_idx] - 
                            self.regularization * self.user_factors[user_idx]
                        )
                        self.item_factors[item_idx] += self.learning_rate * (
                            error * self.user_factors[user_idx] - 
                            self.regularization * self.item_factors[item_idx]
                        )
                    
                    # 计算训练误差
                    train_rmse = self._calculate_rmse()
                    print(f"Epoch {epoch+1}/{self.n_epochs}, RMSE: {train_rmse:.4f}")
                
                self.is_trained = True
                # 保存模型
                self._save_model()
                return True
                
        except Exception as e:
            print(f"训练错误: {str(e)}")
            return False
            
    def _save_model(self):
        """保存模型参数"""
        try:
            # 确保模型目录存在
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # 保存模型参数
            model_data = {
                'user_factors': self.user_factors,
                'item_factors': self.item_factors,
                'user_bias': self.user_bias,
                'item_bias': self.item_bias,
                'global_bias': self.global_bias,
                'user_map': self.user_map,
                'item_map': self.item_map,
                'surprise_model': self.surprise_model
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print("模型保存成功")
            return True
        except Exception as e:
            print(f"模型保存错误: {str(e)}")
            return False
            
    def load_model(self):
        """加载已保存的模型"""
        try:
            if not os.path.exists(self.model_path):
                print("模型文件不存在")
                return False
                
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.user_factors = model_data['user_factors']
            self.item_factors = model_data['item_factors']
            self.user_bias = model_data['user_bias']
            self.item_bias = model_data['item_bias']
            self.global_bias = model_data['global_bias']
            self.user_map = model_data['user_map']
            self.item_map = model_data['item_map']
            self.surprise_model = model_data['surprise_model']
            self.is_trained = True
            
            print("模型加载成功")
            return True
        except Exception as e:
            print(f"模型加载错误: {str(e)}")
            return False
            
    def predict_rating(self, user_idx, item_idx):
        """预测用户对物品的评分"""
        if self.surprise_model is not None:
            # 使用Surprise模型预测
            user_id = list(self.user_map.keys())[list(self.user_map.values()).index(user_idx)]
            item_id = list(self.item_map.keys())[list(self.item_map.values()).index(item_idx)]
            return self.surprise_model.predict(user_id, item_id).est
        else:
            # 使用自定义实现预测
            return (self.global_bias + 
                    self.user_bias[user_idx] + 
                    self.item_bias[item_idx] + 
                    np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
                
    def _calculate_rmse(self):
        """计算均方根误差"""
        predictions = np.array([
            self.predict_rating(user_idx, item_idx)
            for user_idx, item_idx, _ in self.train_data
        ])
        actual = self.train_data[:, 2]
        return np.sqrt(mean_squared_error(actual, predictions))
        
    def recommend_for_user(self, user_id, n=10):
        """为用户推荐物品"""
        try:
            if not self.is_trained:
                print("模型尚未训练")
                return pd.DataFrame()
            
            # 处理用户不在模型中的情况
            if user_id not in self.user_map:
                print(f"用户 {user_id} 不存在于MF模型，尝试用行为数据推荐")
                # 延迟导入，避免循环依赖
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from app import get_user_behavior_csv, cf_recommender, get_cold_start_books
                behavior = get_user_behavior_csv(user_id)
                likes = behavior['like']
                if likes:
                    recs = []
                    import random
                    for like in random.sample(likes, min(2, len(likes))):
                        similar = cf_recommender.get_similar_books(like, n=n//2)
                        if similar is not None and not similar.empty:
                            recs.append(similar)
                    if recs:
                        combined = pd.concat(recs).drop_duplicates(subset=['ISBN'])
                        return combined.head(n)[['ISBN']]
                # 没有收藏，冷启动
                cold_books = get_cold_start_books(n)
                return cold_books[['ISBN']]
            
            user_idx = self.user_map[user_id]
            predictions = []
            for item_id, item_idx in self.item_map.items():
                pred_rating = self.predict_rating(user_idx, item_idx)
                predictions.append((item_id, pred_rating))
            top_items = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
            return pd.DataFrame(top_items, columns=['ISBN', 'Predicted_Rating'])
        except Exception as e:
            print(f"推荐错误: {str(e)}")
            return pd.DataFrame(columns=['ISBN', 'Predicted_Rating'])
            
    def get_similar_items(self, item_id, n=10):
        """获取相似物品"""
        try:
            if not self.is_trained:
                print("模型尚未训练")
                return pd.DataFrame()
                
            # 处理物品不在模型中的情况
            if item_id not in self.item_map:
                print(f"物品 {item_id} 不存在")
                # 对于不存在的物品，返回热门物品
                if hasattr(self, 'item_bias') and self.item_bias is not None:
                    popular_items = []
                    for item_id, item_idx in self.item_map.items():
                        popular_items.append((item_id, self.item_bias[item_idx]))
                    top_popular = sorted(popular_items, key=lambda x: x[1], reverse=True)[:n]
                    return pd.DataFrame(top_popular, columns=['ISBN', 'Similarity'])
                else:
                    # 如果没有物品偏置项，返回空DataFrame
                    return pd.DataFrame(columns=['ISBN', 'Similarity'])
                
            item_idx = self.item_map[item_id]
            target_vector = self.item_factors[item_idx]
            
            # 计算与其他物品的相似度
            similarities = []
            for other_id, other_idx in self.item_map.items():
                if other_id != item_id:
                    similarity = np.dot(target_vector, self.item_factors[other_idx]) / (
                        np.linalg.norm(target_vector) * np.linalg.norm(self.item_factors[other_idx])
                    )
                    similarities.append((other_id, similarity))
            
            # 获取最相似的n个物品
            similar_items = sorted(similarities, key=lambda x: x[1], reverse=True)[:n]
            
            return pd.DataFrame(similar_items, columns=['ISBN', 'Similarity'])
            
        except Exception as e:
            print(f"获取相似物品错误: {str(e)}")
            return pd.DataFrame(columns=['ISBN', 'Similarity']) 