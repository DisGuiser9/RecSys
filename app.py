from flask import Flask, render_template, request, jsonify, session, abort
import pandas as pd
import numpy as np
from CF import BookRecommender
from MF import MatrixFactorizationRecommender
from hybrid import HybridRecommender
from datetime import datetime, timedelta
import json
import os
import csv
import random
import re
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'book_recommendation_system_secret_key'  # 添加secret_key，用于session

# Initialize recommender systems
cf_recommender = BookRecommender()
mf_recommender = MatrixFactorizationRecommender()
hybrid_recommender = HybridRecommender()

# Data file paths
RATINGS_PATH = 'dataset/Ratings.csv'
BOOKS_PATH = 'dataset/Books.csv'
BOOKS_CSV_PATH = os.path.join('dataset', 'Books.csv')
USER_DATA_DIR = 'data/users'

# Load initial data
def load_initial_data():
    try:
        if cf_recommender.load_initial_data(RATINGS_PATH, BOOKS_PATH):
            print("Initial data loaded successfully")
            return True
        return False
    except Exception as e:
        print(f"Error loading initial data: {str(e)}")
        return False

# Load full data
def load_full_data():
    try:
        if cf_recommender.load_full_data(RATINGS_PATH):
            print("Full data loaded successfully")
            return True
        return False
    except Exception as e:
        print(f"Error loading full data: {str(e)}")
        return False

# Load and train MF model
def load_and_train_mf():
    try:
        if mf_recommender.load_model():
            print("MF model loaded successfully")
            return True
            
        if mf_recommender.load_data(RATINGS_PATH):
            print("MF data loaded successfully")
            if mf_recommender.train(use_surprise=True):
                print("MF model trained and saved successfully")
                return True
        return False
    except Exception as e:
        print(f"Error loading/training MF model: {str(e)}")
        return False

# Load Hybrid model
def load_hybrid_model():
    """加载Hybrid混合推荐模型"""
    try:
        global hybrid_recommender
        print("正在加载Hybrid模型...")
        loaded_hybrid = HybridRecommender.load_model('models')
        hybrid_recommender = loaded_hybrid
        print("Hybrid模型加载成功")
        return True
    except Exception as e:
        print(f"加载Hybrid模型出错: {str(e)}")
        return False

# Get user profile
def get_user_profile(user_id):
    try:
        behavior = get_user_behavior_csv(user_id)
        likes = behavior['like']
        searches = behavior['search']
        
        # Get book info for likes
        favorite_books_info = cf_recommender.get_books_by_isbn(likes) if likes else pd.DataFrame()
        
        profile = {
            'favorite_books': likes,
            'search_history': searches,
            'favorite_authors': favorite_books_info['Book-Author'].value_counts().head(3).to_dict() if not favorite_books_info.empty else {},
            'favorite_categories': favorite_books_info['Category'].value_counts().head(3).to_dict() if not favorite_books_info.empty else {},
        }
        return profile
    except Exception as e:
        print(f"Error getting user profile: {str(e)}")
        return None

# Hybrid recommendation strategy
def hybrid_recommend(user_id, n=10):
    try:
        recommendations = []
        profile = get_user_profile(user_id)
        likes = profile['favorite_books'] if profile else []
        # 1. Recommend based on likes (explicit)
        if likes:
            favorite_books_info = cf_recommender.get_books_by_isbn(likes)
            # Recommend by favorite authors
            if profile['favorite_authors']:
                author_recs = cf_recommender.search_by_author(list(profile['favorite_authors'].keys())[0])
                if not author_recs.empty:
                    author_recs_copy = author_recs.copy()
                    author_recs_copy['method'] = 'Author-Based'
                    recommendations.append(author_recs_copy)
            # Recommend by favorite authors as categories (since we don't have real categories)
            if profile['favorite_categories']:
                # 使用作者作为分类，搜索分类对应的作者名
                author_name = list(profile['favorite_categories'].keys())[0]
                category_recs = cf_recommender.search_by_author(author_name)
                if not category_recs.empty:
                    category_recs_copy = category_recs.copy()
                    category_recs_copy['method'] = 'Category-Based'
                    recommendations.append(category_recs_copy)
            # Recommend similar to last liked books
            for book_id in likes[-3:]:
                similar_books = mf_recommender.get_similar_items(book_id, n=2)
                if not similar_books.empty:
                    similar_books_data = cf_recommender.get_books_by_isbn(similar_books['ISBN'].tolist())
                    if not similar_books_data.empty:
                        similar_books_copy = similar_books_data.copy()
                        similar_books_copy['method'] = 'Similar-to-Favorite'
                        recommendations.append(similar_books_copy)
        # 2. Personalized (implicit)
        if mf_recommender.is_trained:
            mf_recs = mf_recommender.recommend_for_user(user_id, n=n)
            if not mf_recs.empty:
                mf_books = cf_recommender.get_books_by_isbn(mf_recs['ISBN'].tolist())
                if not mf_books.empty:
                    mf_books_copy = mf_books.copy()
                    mf_books_copy['method'] = 'MF-Personalized'
                    recommendations.append(mf_books_copy)
        # 3. Popular
        popular_recs = cf_recommender.get_popular_books(n)
        if not popular_recs.empty:
            popular_copy = popular_recs.copy()
            popular_copy['method'] = 'Popular'
            recommendations.append(popular_copy)
        
        if recommendations:
            final_recs = pd.concat(recommendations, ignore_index=True)
            final_recs = final_recs.drop_duplicates(subset=['ISBN'], keep='first')
            return final_recs.head(n)
        return pd.DataFrame()
    except Exception as e:
        print(f"Error in hybrid recommendation: {str(e)}")
        return pd.DataFrame()

def is_valid_user_id(user_id):
    return isinstance(user_id, str) and re.fullmatch(r'user([1-9][0-9]?|100)', user_id)

@app.route('/')
def index():
    try:
        user_id = request.args.get('user_id') or session.get('user_id')
        if not is_valid_user_id(user_id):
            user_id = None
        if user_id:
            session['user_id'] = user_id
            print(f"使用用户ID: {user_id}")
        else:
            print("未提供用户ID，仅显示热门图书")
        recommendation_mode = request.args.get('mode', 'quick')
        print(f"推荐模式: {recommendation_mode}")
        # 全局热门图书
        popular_books = cf_recommender.get_popular_books(12)
        if popular_books.empty:
            print("警告: 热门图书列表为空")
        else:
            print(f"成功获取热门图书: {len(popular_books)} 本")
        popular_books['method'] = 'Popular'
        # 个性化推荐
        recommended_books = pd.DataFrame()
        favorite_isbns = set()
        if user_id:
            try:
                behavior = get_user_behavior_csv(user_id)
                favorite_isbns = set(behavior['like'])
                print(f"用户 {user_id} 有 {len(favorite_isbns)} 本收藏图书")
                if recommendation_mode == 'accurate':
                    print(f"为用户 {user_id} 使用精准推荐模式")
                    recommended_books = get_accurate_recommendations(user_id, favorite_isbns, n=12)
                else:
                    print(f"为用户 {user_id} 使用快速推荐模式")
                    recommended_books = get_quick_recommendations(user_id, n=12)
                if recommended_books.empty:
                    print("获取推荐失败，将使用热门图书")
            except Exception as e:
                print(f"获取推荐时出错: {e}")
        formatted_popular = format_books(popular_books, favorite_isbns)
        if recommended_books.empty:
            formatted_recommended = []
        else:
            formatted_recommended = format_books(recommended_books, favorite_isbns)
        return render_template('index.html', 
                              popular_books=formatted_popular[:6],
                              guess_books=formatted_recommended[:6],
                              show_stats=True, 
                              recommendation_mode=recommendation_mode)
    except Exception as e:
        print(f"Index error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return render_template('index.html', 
                              popular_books=[], 
                              guess_books=[], 
                              error=str(e), 
                              recommendation_mode=recommendation_mode)

@app.route('/recommend', methods=['POST'])
def recommend():
    """处理各种推荐请求"""
    data = request.get_json()
    
    # 获取当前用户ID（如果已登录）
    user_id = data.get('user_id') or session.get('user_id')
    
    if not is_valid_user_id(user_id):
        return jsonify({'error': 'Invalid user_id', 'books': []}), 400
    
    session['user_id'] = user_id
    
    # 获取推荐模式，默认为快速模式
    recommendation_mode = data.get('recommendation_mode', 'quick')  # 'quick' 或 'accurate'
    
    # 检查是否有必要的数据
    if not data:
        return jsonify({'error': '没有提供数据'}), 400
    
    request_type = data.get('type')
    if not request_type:
        return jsonify({'error': '没有指定请求类型'}), 400
    
    try:
        # 根据请求类型进行不同处理
        if request_type == 'title':
            title = data.get('title')
            if not title:
                return jsonify({'error': '没有提供书名'}), 400
            
            # 记录搜索行为
            if user_id:
                record_user_behavior_csv(user_id, 'search', extra_data=title)
                print(f"记录用户 {user_id} 搜索: {title}")
            
            books = cf_recommender.search_by_title(title, n=6)
            if books is None or books.empty:
                return jsonify({'books': [], 'message': '未找到相关书籍'}), 200
            
            books['method'] = 'Title-Search'
            formatted_books = format_books(books, get_user_behavior_csv(user_id)['like'] if user_id else [])
            return jsonify({
                'books': formatted_books,
                'message': '基于书名推荐成功'
            }), 200
            
        elif request_type == 'subject':
            subject = data.get('subject')
            if not subject:
                return jsonify({'error': '没有提供学科'}), 400
            
            # 记录搜索行为
            if user_id:
                record_user_behavior_csv(user_id, 'search', extra_data=f"subject:{subject}")
            
            books = cf_recommender.search_by_title(subject, n=8)  # 使用title搜索代替不存在的subject方法
            if books is None or books.empty:
                return jsonify({'books': [], 'message': '未找到相关书籍'}), 200
            
            books['method'] = 'Subject-Search'
            formatted_books = format_books(books, get_user_behavior_csv(user_id)['like'] if user_id else [])
            return jsonify({
                'books': formatted_books,
                'message': '基于学科推荐成功'
            }), 200
            
        elif request_type == 'genre':
            genre = data.get('genre')
            if not genre:
                return jsonify({'error': '没有提供分类'}), 400
            
            # 记录搜索行为
            if user_id:
                record_user_behavior_csv(user_id, 'search', extra_data=f"genre:{genre}")
            
            books = cf_recommender.search_by_author(genre, n=6)  # 使用author搜索代替不存在的genre方法
            if books is None or books.empty:
                return jsonify({'books': [], 'message': '未找到相关书籍'}), 200
                
            books['method'] = 'Genre-Search'
            formatted_books = format_books(books, get_user_behavior_csv(user_id)['like'] if user_id else [])
            return jsonify({
                'books': formatted_books,
                'message': '基于分类推荐成功'
            }), 200
            
        elif request_type == 'personalized':
            if not user_id:
                # 用户未登录，返回热门推荐
                books = cf_recommender.get_popular_books(6)
                books['method'] = 'Popular'
                formatted_books = format_books(books)
                return jsonify({
                    'books': formatted_books,
                    'message': '用户未登录，返回热门推荐'
                }), 200
            
            # 获取用户收藏
            likes = get_user_behavior_csv(user_id)['like']
            
            # 根据推荐模式选择不同的推荐策略
            if recommendation_mode == 'quick':
                books = get_quick_recommendations(user_id, n=6)
            else:  # accurate模式
                books = get_accurate_recommendations(user_id, likes, n=6)
            
            if books is None or books.empty:
                # 如果没有个性化推荐，返回热门推荐
                books = cf_recommender.get_popular_books(6)
                books['method'] = 'Popular'
                formatted_books = format_books(books, likes)
                return jsonify({
                    'books': formatted_books,
                    'message': '无个性化推荐，返回热门推荐'
                }), 200
            
            formatted_books = format_books(books, likes)
            return jsonify({
                'books': formatted_books,
                'message': f'个性化推荐成功 ({recommendation_mode}模式)'
            }), 200
        
        else:
            return jsonify({'error': '不支持的请求类型'}), 400
            
    except Exception as e:
        print(f"推荐错误: {str(e)}")
        return jsonify({'error': f'推荐过程中发生错误: {str(e)}'}), 500

# 冷启动推荐函数
def get_cold_start_books(n=6):
    """
    冷启动推荐：从全体图书中排除热门图书，随机推荐n本
    """
    try:
        # 获取前20本热门图书ISBN
        popular_isbns = set(cf_recommender.get_popular_books(20)['ISBN'])
        # 获取所有图书
        all_books = cf_recommender.books_df
        # 排除热门图书
        cold_books = all_books[~all_books['ISBN'].isin(popular_isbns)]
        # 随机采样n本
        if len(cold_books) >= n:
            return cold_books.sample(n)
        else:
            return cold_books
    except Exception as e:
        print(f"冷启动推荐出错: {e}")
        return pd.DataFrame()

# 快速推荐策略 - 使用矩阵分解(MF)方法
def get_quick_recommendations(user_id, n=6):
    """
    快速推荐策略 - 使用矩阵分解(MF)方法
    """
    try:
        likes = get_user_behavior_csv(user_id)['like']
        print(f"用户收藏的ISBN: {likes}")
        
        if not likes or len(likes) == 0:
            print("用户没有收藏，返回热门书籍")
            popular_books = cf_recommender.get_popular_books(n).copy()
            popular_books['method'] = 'Popular'
            return popular_books
        
        # # 首先尝试使用MF模型
        # if mf_recommender.is_trained:
        #     # 如果用户在MF模型中
        #     if user_id in mf_recommender.user_map:
        #         print(f"使用MF模型为用户 {user_id} 推荐图书")
        #         mf_recs = mf_recommender.recommend_for_user(user_id, n*2)
        #         if mf_recs is not None and not mf_recs.empty:
        #             # 排除已收藏图书
        #             mf_recs = mf_recs[~mf_recs['ISBN'].isin(likes)]
        #             if len(mf_recs) > 0:
        #                 mf_books = cf_recommender.get_books_by_isbn(mf_recs['ISBN'].tolist()[:n])
        #                 if not mf_books.empty:
        #                     mf_books['method'] = 'MF-Personalized'
        #                     print(f"MF成功推荐 {len(mf_books)} 本图书")
        #                     return mf_books.head(n)
        
        # # 如果MF方法失败，尝试基于收藏的相似图书
        # print("MF方法失败，尝试基于内容的相似图书推荐")

        all_similar_books = []
        
        for isbn in likes:
            try:
                # 获取最近10本相似图书
                similar_books = mf_recommender.get_similar_items(isbn, n=5)
                if similar_books is not None and not similar_books.empty:
                    # 确保similar_books包含similarity列
                    if 'similarity' not in similar_books.columns:
                        similar_books['similarity'] = 0.8  # 默认相似度
                    similar_books_copy = similar_books.copy()
                    similar_books_copy['source_isbn'] = isbn
                    all_similar_books.append(similar_books_copy)
            except Exception as e:
                print(f"获取ISBN={isbn}的相似图书失败: {e}")
                continue
        
        if all_similar_books:
            # 合并所有相似图书
            combined = pd.concat(all_similar_books, ignore_index=True)
            # 排除已收藏
            combined = combined[~combined['ISBN'].isin(likes)]
            # 确保存在similarity列
            if 'similarity' not in combined.columns:
                combined['similarity'] = 0.5  # 默认相似度
            # 根据相似度排序
            combined = combined.sort_values('similarity', ascending=False)
            # 去重
            combined = combined.drop_duplicates('ISBN', keep='first')
            
            if not combined.empty:
                result_books = cf_recommender.get_books_by_isbn(combined['ISBN'].tolist()[:n])
                if not result_books.empty:
                    # 添加相似度和来源信息
                    similarity_dict = dict(zip(combined['ISBN'], combined['similarity']))
                    source_dict = dict(zip(combined['ISBN'], combined['source_isbn']))
                    
                    result_books['similarity'] = result_books['ISBN'].map(similarity_dict)
                    result_books['source_isbn'] = result_books['ISBN'].map(source_dict)
                    result_books['method'] = 'MF-Similar-Items'
                    
                    print(f"基于内容相似成功推荐 {len(result_books)} 本图书")
                    return result_books.head(n)
        
        # 如果上述方法都失败，返回热门图书
        print("所有方法均失败，返回热门图书")
        popular_books = cf_recommender.get_popular_books(n).copy()
        popular_books['method'] = 'Popular'
        return popular_books
    
    except Exception as e:
        print(f"快速推荐错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        popular_books = cf_recommender.get_popular_books(n).copy()
        popular_books['method'] = 'Popular'
        return popular_books

# 精准推荐策略 - 质量优先
def get_accurate_recommendations(user_id, likes, n=6):
    """
    精准推荐策略 - 使用混合(Hybrid)方法
    """
    global hybrid_recommender
    
    try:
        print(f"为用户 {user_id} 获取精准推荐（Hybrid模型）")
        
        # 确保likes是一个列表而不是集合
        if isinstance(likes, set):
            likes = list(likes)
        
        if not likes or len(likes) == 0:
            print("用户没有收藏，返回热门书籍")
            popular_books = cf_recommender.get_popular_books(n).copy()
            popular_books['method'] = 'Popular'
            return popular_books
        
        # 使用Hybrid混合推荐
        try:
            # 获取用户的最近一次收藏图书 (确保likes是列表)
            if not isinstance(likes, list):
                likes = list(likes)
                
            if len(likes) == 0:
                raise ValueError("用户收藏列表为空")
                
            recent_book_isbn = likes[-1]  # 获取最后一个元素
            
            book_row = cf_recommender.books_df[cf_recommender.books_df['ISBN'] == recent_book_isbn]
            
            if book_row.empty:
                raise ValueError(f"找不到ISBN为 {recent_book_isbn} 的图书信息")
                
            book_title = book_row.iloc[0]['Book-Title']
            print(f"基于用户最近收藏的 '{book_title}' 进行混合推荐")
            
            # 用Hybrid模型推荐
            hybrid_recs = hybrid_recommender.get_hybrid_recommendations(user_id, book_title, n_recommendations=n*2)
            
            if not hybrid_recs or len(hybrid_recs) == 0:
                raise ValueError("Hybrid模型返回空推荐列表")
                
            # 转换推荐结果为DataFrame
            books_info = []
            for title in hybrid_recs:
                book_info = cf_recommender.books_df[cf_recommender.books_df['Book-Title'] == title]
                if not book_info.empty:
                    books_info.append(book_info.iloc[0])
            
            if not books_info:
                raise ValueError("无法找到推荐图书的详细信息")
                
            result = pd.DataFrame(books_info)
            # 排除已收藏
            result = result[~result['ISBN'].isin(likes)]
            result['method'] = 'Hybrid'
            print(f"Hybrid成功推荐 {len(result)} 本图书")
            return result.head(n)
            
        except Exception as hybrid_e:
            print(f"Hybrid推荐出错: {hybrid_e}")
            # 继续后续推荐逻辑
        
        # 如果Hybrid方法失败，尝试使用综合内容和协同过滤推荐
        print("Hybrid方法失败，尝试综合内容和协同过滤推荐")
        
        # 以下保持原有的备选推荐逻辑不变
        # 获取用户收藏的图书信息
        liked_books_df = cf_recommender.get_books_by_isbn(likes)
        
        # 1. 先基于作者推荐
        authors = liked_books_df['Book-Author'].value_counts().head(3).index.tolist()
        author_recs = []
        
        for author in authors:
            if pd.notna(author) and author:
                author_matches = cf_recommender.search_by_author(author)
                if not author_matches.empty:
                    author_matches = author_matches[~author_matches['ISBN'].isin(likes)]
                    author_matches['source'] = f"Author: {author}"
                    author_matches['score'] = 0.8  # 作者匹配得分
                    author_recs.append(author_matches)
        
        # 2. 再基于标题相似推荐
        title_recs = []
        for _, book in liked_books_df.iterrows():
            title = book['Book-Title']
            if pd.notna(title) and title:
                title_matches = cf_recommender.search_by_title(title)
                if not title_matches.empty:
                    title_matches = title_matches[~title_matches['ISBN'].isin(likes)]
                    title_matches['source'] = f"Title: {title}"
                    title_matches['score'] = 0.9  # 标题匹配得分
                    title_recs.append(title_matches)
        
        # 3. 合并推荐结果
        combined_recs = []
        if author_recs:
            combined_recs.extend(author_recs)
        if title_recs:
            combined_recs.extend(title_recs)
        
        if combined_recs:
            all_recs = pd.concat(combined_recs, ignore_index=True)
            # 去重
            all_recs = all_recs.drop_duplicates('ISBN', keep='first')
            # 按相似度排序
            all_recs = all_recs.sort_values('score', ascending=False)
            
            if not all_recs.empty:
                all_recs['method'] = 'Content-Collaborative'
                print(f"综合内容协同成功推荐 {len(all_recs)} 本图书")
                return all_recs.head(n)
        
        # 如果上述方法都失败，返回热门图书
        print("所有方法均失败，返回热门图书")
        popular_books = cf_recommender.get_popular_books(n).copy()
        popular_books['method'] = 'Popular'
        return popular_books
    
    except Exception as e:
        print(f"精准推荐错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        popular_books = cf_recommender.get_popular_books(n).copy()
        popular_books['method'] = 'Popular'
        return popular_books

@app.route('/user/history', methods=['GET'])
def get_user_history():
    try:
        user_id = request.args.get('user_id')

        # If no user ID is provided, return empty
        if not user_id:
            return jsonify({
                'error': 'No user ID provided',
                'message': 'Please enter a user ID to view reading history'
            })
        
        # Try to convert user ID to integer
        try:
            user_id = int(user_id)
        except ValueError:
            return jsonify({
                'error': 'Input error',
                'message': 'Please enter a valid user ID (number)'
            })
        
        # Ensure full data is loaded
        if not cf_recommender.is_full_data_loaded:
            if not load_full_data():
                return jsonify({
                    'error': 'System error',
                    'message': 'Unable to load user data'
                })
        
        # Get user history
        user_history = cf_recommender.get_user_history(user_id)
        
        if user_history.empty:
            return jsonify({
                'error': 'No reading history',
                'message': 'This user has no reading history'
            })
        
        # Format history records
        formatted_history = []
        for _, book in user_history.iterrows():
            formatted_history.append({
                'title': str(book['Book-Title']),
                'author': str(book['Book-Author']),
                'year': str(book['Year-Of-Publication']),
                'publisher': str(book['Publisher']),
                'img': str(book['Image-URL-M']),
                'rating': int(book['Book-Rating'])
            })
            
        return jsonify({
            'error': None,
            'data': formatted_history
        })
    except Exception as e:
        return jsonify({
            'error': 'System error',
            'message': str(e)
        }), 500

@app.route('/user/rate', methods=['POST'])
def rate_book():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        book_id = data.get('book_id')
        rating = data.get('rating')
        
        if not all([user_id, book_id, rating]):
            return jsonify({
                'error': 'Missing parameters',
                'message': 'User ID, book ID and rating are required'
            })
        
    except Exception as e:
        return jsonify({
            'error': 'System error',
            'message': str(e)
        }), 500

@app.route('/user/profile', methods=['GET'])
def user_profile_route():
    try:
        user_id = request.args.get('user_id')
        
        if not user_id:
            return jsonify({
                'error': 'No user ID provided',
                'message': 'Please enter a user ID to view profile'
            })
        
        try:
            user_id = int(user_id)
        except ValueError:
            return jsonify({
                'error': 'Input error',
                'message': 'Please enter a valid user ID (number)'
            })
        
        profile = get_user_profile(user_id)
        if not profile:
            return jsonify({
                'error': 'Profile not found',
                'message': 'Unable to generate user profile'
            })
            
        return jsonify({
            'error': None,
            'data': profile
        })
        
    except Exception as e:
        return jsonify({
            'error': 'System error',
            'message': str(e)
        }), 500

@app.route('/book/<isbn>')
def book_detail(isbn):
    # 读取Books.csv，所有字段为字符串
    df = pd.read_csv(BOOKS_CSV_PATH, dtype=str)
    book_row = df[df['ISBN'] == isbn]
    if book_row.empty:
        abort(404, description="Book not found")
    book = book_row.iloc[0].to_dict()
    if 'Summary' not in book:
        book['Summary'] = ''
    return render_template('book.html', book=book)

@app.route('/book/favorite', methods=['POST'])
def favorite_book():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        if not is_valid_user_id(user_id):
            return jsonify({'error': 'Invalid user_id'}), 400
        book_id = data.get('book_id')
        
        print(f"收藏请求参数: user_id={user_id}, book_id={book_id}")
        
        if not all([user_id, book_id]):
            return jsonify({
                'error': 'Missing parameters',
                'message': 'User ID and book ID are required'
            })
        
        # 获取当前用户收藏记录
        behavior = get_user_behavior_csv(user_id)
        likes = behavior['like']
        
        print(f"当前用户收藏: {likes}")
        
        # 判断是添加还是删除收藏
        is_favorite = book_id in likes
        
        if is_favorite:
            # 如果已收藏，则取消收藏
            print(f"取消收藏: {book_id}")
            record_user_behavior_csv(user_id, 'cancel_like', book_id=book_id)
            status_message = 'Book removed from favorites successfully'
            is_favorite = False
        else:
            # 如果未收藏，则添加收藏
            print(f"添加收藏: {book_id}")
            record_user_behavior_csv(user_id, 'like', book_id=book_id)
            status_message = 'Book added to favorites successfully'
            is_favorite = True
        
        print(f"收藏状态: {status_message}")
        
        return jsonify({
            'error': None,
            'message': status_message,
            'is_favorite': is_favorite
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"收藏错误: {str(e)}")
        print(f"错误详情: {error_trace}")
        return jsonify({
            'error': 'System error',
            'message': str(e)
        }), 500

@app.route('/user/favorites', methods=['GET'])
def user_favorites():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'No user ID provided', 'books': []})
    behavior = get_user_behavior_csv(user_id)
    likes = behavior['like']
    if not likes:
        return jsonify({'error': None, 'books': []})
    books_df = cf_recommender.get_books_by_isbn(likes)
    formatted = format_books(books_df, likes)
    return jsonify({'error': None, 'books': formatted})

@app.route('/favorites')
def favorites():
    try:
        # 获取用户ID（如果存在）
        user_id = session.get('user_id')
        
        # 直接返回收藏页面，具体数据将通过AJAX加载
        return render_template('favorites.html')
    except Exception as e:
        print(f"Favorites error: {str(e)}")
        return render_template('favorites.html', error=str(e))

@app.route('/recommendation_reasons')
def recommendation_reasons():
    """显示推荐原因的页面"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return render_template('recommendation_reasons.html', 
                                  error="Please log in or select a user ID first", 
                                  recommendations=[])
        
        # 获取推荐模式
        recommendation_mode = request.args.get('mode', 'quick')
        
        # 获取用户收藏
        likes = get_user_behavior_csv(user_id)['like']
        if not likes:
            return render_template('recommendation_reasons.html', 
                                  error="You haven't added any books to your favorites yet", 
                                  recommendations=[])
            
        # 获取实际推荐的书籍（与主页相同逻辑）
        books_df = cf_recommender.books_df
        if recommendation_mode == 'accurate':
            print(f"为推荐原因页面使用精准推荐模式")
            recommended_books = get_accurate_recommendations(user_id, likes, n=12)
        else:
            print(f"为推荐原因页面使用快速推荐模式")
            recommended_books = get_quick_recommendations(user_id, n=12)
            
        if recommended_books.empty:
            return render_template('recommendation_reasons.html', 
                                  error="Unable to generate recommendations", 
                                  recommendations=[])
                                  
        # 获取用户收藏的书籍详情
        liked_books = []
        for isbn in likes:
            book = books_df[books_df['ISBN'] == isbn]
            if not book.empty:
                book_info = book.iloc[0]
                liked_books.append({
                    'isbn': isbn,
                    'title': book_info['Book-Title'],
                    'author': book_info['Book-Author'],
                    'image': book_info['Image-URL-M'],
                    'year': book_info['Year-Of-Publication'],
                    'publisher': book_info['Publisher']
                })
        
        # 整理推荐原因
        recommendations = []
        source_books_used = set()  # 跟踪已使用的收藏书
        
        # 从推荐结果提取相似度和来源信息
        for _, rec_book in recommended_books.iterrows():
            isbn = rec_book['ISBN']
            method = rec_book.get('method', 'Unknown')
            similarity = rec_book.get('similarity', 0.7)  # 默认相似度
            source_isbn = rec_book.get('source_isbn', '')
            
            # 找出源书籍（用户收藏的书）
            source_book = None
            if source_isbn and source_isbn in likes:
                # 如果有明确的来源书籍
                for book in liked_books:
                    if book['isbn'] == source_isbn:
                        source_book = book
                        source_books_used.add(source_isbn)
                        break
            
            # 如果没有明确来源，但可能是基于作者或分类推荐
            if source_book is None and method in ['Author-Based', 'Category-Based', 'Content-Collaborative', 'MF-Similar-Items']:
                # 查找作者匹配的收藏书
                rec_author = rec_book['Book-Author']
                for book in liked_books:
                    if book['author'] == rec_author and book['isbn'] not in source_books_used:
                        source_book = book
                        source_books_used.add(book['isbn'])
                        break
            
            # 如果仍然没有来源，可能是MF/Hybrid等模型推荐，随机选一本收藏的书作为"灵感来源"
            if source_book is None and len(liked_books) > 0:
                remaining_books = [b for b in liked_books if b['isbn'] not in source_books_used]
                if remaining_books:
                    source_book = remaining_books[0]
                    source_books_used.add(source_book['isbn'])
                else:
                    # 如果所有书都已用过，重用第一本
                    source_book = liked_books[0]
            
            # 如果还是没有，那就跳过这本推荐
            if source_book is None:
                continue
                
            # 生成推荐原因
            reasons = []
            
            method_map = {
                'MF-Personalized':  "Recommended by Matrix-Factorization (personalized)",
                'MF-Similar-Items': "Similar to books you've liked",
                'Hybrid':           "Hybrid recommendation (content + collaborative)",
                'Author-Based':     f"Same author: {rec_book['Book-Author']}",
                'Content-Collaborative': "Content-based similarity"
            }

            # 如果 method 在映射里就加入对应描述
            if method in method_map:
                reasons.append(method_map[method])
                
            # 构造推荐书的详细信息
            similar_book = {
                'isbn': isbn,
                'title': rec_book['Book-Title'],
                'author': rec_book['Book-Author'],
                'image': rec_book['Image-URL-M'],
                'year': rec_book['Year-Of-Publication'],
                'publisher': rec_book['Publisher'],
                'similarity': similarity,
                'reasons': reasons
            }
            
            # 查找此来源书是否已有推荐组
            found = False
            for rec in recommendations:
                if rec['source_book']['isbn'] == source_book['isbn']:
                    rec['similar_books'].append(similar_book)
                    found = True
                    break
                    
            # 如果没有，创建新的推荐组
            if not found:
                recommendations.append({
                    'source_book': source_book,
                    'similar_books': [similar_book]
                })
        
        # 最后对每个推荐组内的书按相似度排序
        for rec in recommendations:
            rec['similar_books'] = sorted(rec['similar_books'], key=lambda x: x['similarity'], reverse=True)
            # 限制每组推荐数量
            rec['similar_books'] = rec['similar_books'][:5]
        
        return render_template('recommendation_reasons.html', 
                              recommendations=recommendations, 
                              liked_books=liked_books,
                              mode=recommendation_mode,
                              error=None)
    
    except Exception as e:
        import traceback
        print(f"推荐原因页面错误: {str(e)}")
        print(traceback.format_exc())
        return render_template('recommendation_reasons.html', 
                              error=f"Error getting recommendation reasons: {str(e)}", 
                              recommendations=[])

def format_books(books_df, favorite_isbns=None):
    """Format book data for frontend display"""
    try:
        favorite_isbns = set(favorite_isbns or [])
        formatted_books = []
        
        # 确保有评分字段
        if 'rating_count' not in books_df.columns:
            books_df['rating_count'] = 0
        if 'avg_rating' not in books_df.columns:
            books_df['avg_rating'] = 0
        if 'popularity_score' not in books_df.columns:
            books_df['popularity_score'] = 0
        
        # 默认图片URL
        default_img = 'https://s3-us-west-2.amazonaws.com/s.cdpn.io/387928/book%20placeholder.png'
        
        for _, book in books_df.iterrows():
            try:
                isbn = str(book['ISBN'])
                
                # 检查Image-URL-M是否为空或无效
                img_url = str(book['Image-URL-M'])
                if not img_url or img_url.lower() in ['none', 'nan', 'null', '']:
                    img_url = default_img
                
                formatted_book = {
                    'ISBN': isbn,
                    'Book-Title': str(book['Book-Title']),
                    'Book-Author': str(book['Book-Author']),
                    'Year-Of-Publication': str(book['Year-Of-Publication']),
                    'Publisher': str(book['Publisher']),
                    'Image-URL-M': img_url,
                    'rating_count': int(book.get('rating_count', 0)),
                    'avg_rating': float(book.get('avg_rating', 0)),
                    'popularity_score': float(book.get('popularity_score', 0)),
                    'method': str(book.get('method', 'Popular')),
                    'is_favorite': isbn in favorite_isbns
                }
                formatted_books.append(formatted_book)
            except Exception as row_e:
                print(f"格式化单本图书时出错: {str(row_e)}")
                continue
                
        print(f"成功格式化 {len(formatted_books)} 本图书")
        return formatted_books
    except Exception as e:
        print(f"Formatting error: {str(e)}")
        return []

def ensure_user_data_dir():
    """确保用户数据目录存在"""
    os.makedirs(USER_DATA_DIR, exist_ok=True)

def get_user_csv_path(user_id):
    """获取用户CSV文件路径"""
    return os.path.join(USER_DATA_DIR, f"{user_id}.csv")

def init_user_csv(user_id):
    """初始化用户CSV文件，确保包含所有必要字段"""
    csv_path = get_user_csv_path(user_id)
    if os.path.isfile(csv_path):
        return True  # 文件已存在
        
    # 创建CSV文件并写入表头
    fieldnames = ['action_type', 'book_id', 'timestamp', 'extra_data']
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        print(f"为用户 {user_id} 创建了新的行为记录文件")
        return True
    except Exception as e:
        print(f"创建用户CSV文件失败: {e}")
        return False

def record_user_behavior_csv(user_id, action_type, book_id=None, extra_data=None):
    """记录用户行为到CSV文件
    
    Args:
        user_id: 用户ID
        action_type: 行为类型，如'like', 'cancel_like', 'search'
        book_id: 图书ISBN
        extra_data: 额外数据（如搜索词）
    """
    # 确保目录存在
    ensure_user_data_dir()
    
    # 确保用户CSV文件初始化
    if not init_user_csv(user_id):
        return False
        
    # 获取CSV文件路径
    csv_path = get_user_csv_path(user_id)
    
    # 记录用户行为
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['action_type', 'book_id', 'timestamp', 'extra_data'])
            writer.writerow({
                'action_type': action_type,
                'book_id': book_id or '',
                'timestamp': datetime.now().isoformat(),
                'extra_data': extra_data or ''
            })
        print(f"记录用户 {user_id} 的行为: {action_type}, 图书: {book_id}")
        return True
    except Exception as e:
        print(f"记录用户行为失败: {e}")
        return False

def get_user_behavior_csv(user_id):
    """获取用户行为数据
    
    Returns:
        dict: 包含'like'和'search'两个列表
    """
    # 确保用户CSV文件初始化
    if not init_user_csv(user_id):
        return {'like': [], 'search': []}
        
    # 获取CSV文件路径
    csv_path = get_user_csv_path(user_id)
    
    # 读取用户行为
    likes, searches = [], []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'action_type' not in row:
                    continue
                    
                if row['action_type'] == 'like' and row['book_id']:
                    likes.append(row['book_id'])
                elif row['action_type'] == 'cancel_like' and row['book_id'] in likes:
                    likes.remove(row['book_id'])
                elif row['action_type'] == 'search':
                    searches.append({
                        'book_id': row.get('book_id', ''),
                        'search_term': row.get('extra_data', ''),
                        'timestamp': row.get('timestamp', '')
                    })
    except Exception as e:
        print(f"读取用户行为CSV错误: {e}")
    
    return {'like': likes, 'search': searches}

# 添加获取活跃用户ID的函数
def get_random_active_user_id():
    """
    返回格式为user{i}的用户ID，其中i是1-100之间的随机数
    """
    return f"user{random.randint(1, 100)}"

if __name__ == '__main__':
    
    # Load initial data and train MF model on startup
    if load_initial_data() and load_and_train_mf() and load_hybrid_model():
        app.run(debug=True)
    else:
        print("System initialization failed") 