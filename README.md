# 图书推荐系统

这是一个基于Flask的简单图书推荐系统，使用KNN算法为用户提供个性化图书推荐。

## 功能特点

- 展示图书列表
- 基于用户ID的个性化图书推荐
- 使用KNN算法进行协同过滤推荐

## 安装与运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行应用：
```bash
python app.py
```

3. 访问应用：
打开浏览器，访问 http://localhost:5000

## 数据文件

系统使用以下数据文件：
- `dataset/Books.csv`: 图书信息
- `dataset/Users.csv`: 用户信息
- `dataset/Ratings.csv`: 用户评分信息

## 使用方法

1. 在首页可以浏览所有图书
2. 在推荐部分输入用户ID，点击"获取推荐"按钮获取个性化推荐
3. 如果输入的用户ID在数据集中不存在，系统将返回热门图书推荐

## 技术栈

- Flask: Web框架
- Pandas: 数据处理
- Scikit-learn: KNN算法实现
- Bootstrap: 前端UI框架 