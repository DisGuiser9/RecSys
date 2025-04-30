from app import app, init_user_behavior_file, load_initial_data, load_and_train_mf

# 初始化数据和模型
print("正在初始化系统...")
init_user_behavior_file()

# 加载数据和训练模型
if load_initial_data() and load_and_train_mf():
    # 启动应用
    print("初始化成功，正在启动应用...")
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    print("系统初始化失败") 