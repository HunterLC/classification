import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score 
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt

def naive_bayes_classifier(X_train, y_train):
    '''
    朴素贝叶斯分类
    '''
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    return model_nb

def knn_classifier(X_train, y_train):
    '''
    k最近邻分类
    '''
    model_knn = KNeighborsClassifier(n_neighbors = 6, weights = 'distance')
    model_knn.fit(X_train, y_train)
    return model_knn

def decision_tree_classifier(X_train, y_train):
    '''
    决策树分类
    '''
    params = dt_search_best(X_train, y_train)
    print(params['max_depth'],params['min_samples_leaf'],params['min_samples_split'])
    model_dt = DecisionTreeClassifier(max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'],min_samples_split=params['min_samples_split'])
    model_dt.fit(X_train, y_train)
    return model_dt

def dt_search_best(X_train, y_train):
    '''
    采用网格搜索法确定决策树最佳组合参数值
    '''
    #预设各参数的不同选项值
    max_depth = [2, 3, 4, 5, 6]
    min_samples_split = [2, 4, 6, 8]
    min_samples_leaf = [2, 4, 8, 10, 12]
    #将各参数值以字典形式组织起来
    parameters = {'max_depth': max_depth, 'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf}
    # 网格搜索法,测试不同的参数值
    grid_dtcateg= GridSearchCV(estimator = DecisionTreeClassifier(),
                               param_grid = parameters, cv = 10)
    #模型拟合
    grid_dtcateg.fit(X_train, y_train)
    #返回最佳组合的参数值
    print(grid_dtcateg.best_params_)
    return grid_dtcateg.best_params_

def classifier(data_file):
    '''
    data_file : CSV文件
    '''
    df = pd.read_csv(data_file)
    feature_attr = df.columns[:-1]
    label_attr = df.columns[-1]
    
    # 特征预处理 
    obj_attrs = []
    for attr in feature_attr:
        if df.dtypes[attr] == np.dtype(object): #添加离散数据列
            obj_attrs.append(attr)
    if len(obj_attrs) > 0:
        df = pd.get_dummies(df, columns=obj_attrs) #转为哑变量
    
    y = df[label_attr].astype('category').cat.codes.values #将label_attr列转换为分类/分组类型
    df.drop(label_attr, axis=1, inplace=True) #删除列（axis=1指定，默认为行），并将原数据置换为新数据（inplace=True指定，默认为False）
    X = df.values

    #采用10折交叉验证
    kf = KFold(n_splits=10)

    dt_hist = []
    knn_hist = []
    nb_hist = []
    for train_index, test_index in kf.split(X):
        # 加载数据
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 训练
        dt_model = decision_tree_classifier(X_train, y_train)
        knn_model = knn_classifier(X_train, y_train)
        nb_model = naive_bayes_classifier(X_train, y_train)

        # 预测
        dt_pred = dt_model.predict(X_test)
        print('决策树模型的准确率：\n', metrics.accuracy_score(y_test, dt_pred))
        print('决策树模型的评估报告：\n', metrics.classification_report(y_test, dt_pred))
        knn_pred = knn_model.predict(X_test)
        print('K最近邻模型的准确率：\n', metrics.accuracy_score(y_test, knn_pred))
        print('K最近邻模型的评估报告：\n', metrics.classification_report(y_test, knn_pred))
        nb_pred = nb_model.predict(X_test)
        print('朴素贝叶斯模型的准确率：\n', metrics.accuracy_score(y_test,nb_pred))
        print('朴素贝叶斯模型的评估报告：\n',metrics.classification_report(y_test,nb_pred))
         

        #评估阶段：
        #    F度量又称F1分数或F分数
        #    F1= 2 * ( precision * recall ) / ( precison + recall )
        dt_f1 = f1_score(y_test, dt_pred, average='micro')
        knn_f1 = f1_score(y_test, knn_pred, average='micro')
        nb_f1 = f1_score(y_test, nb_pred, average='micro')

        # 结果汇总
        dt_hist.append(dt_f1)
        knn_hist.append(knn_f1)
        nb_hist.append(nb_f1)

    # 绘图

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(18, 10), gridspec_kw={'width_ratios': [5, 3]})
    # 绘制每轮计算的F1值折线图
    ax0.set_title('每轮F1值', color='black')
    ax0.plot(dt_hist, linestyle = '-', color = 'steelblue', label='决策树')
    ax0.plot(knn_hist, linestyle = ':', color = 'black', label='k最近邻')
    ax0.plot(nb_hist, linestyle = '-.', color = 'indianred', label='朴素贝叶斯')
    #添加图例
    ax0.legend()

    # 绘制平均F1值直方图
    ax1.set_title('平均F1值', color='black')
    x_names = ['决策树', 'k最近邻', '朴素贝叶斯']
    y_data = [np.mean(dt_hist), np.mean(knn_hist), np.mean(nb_hist)]
    #print(np.mean(dt_hist), np.mean(knn_hist), np.mean(nb_hist))
    ax1.bar(x = np.arange(len(x_names)), height = y_data)
    #添加直方图数据
    for x,y in enumerate(y_data):
        plt.text(x,y+0.01,'%.2f' %y,ha='center')
    #添加刻度标签
    plt.xticks(np.arange(len(x_names)), x_names, fontsize=12)
    plt.show()

# urls of my data sets:
#     No.1  http://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization 无线室内定位数据集 ok
#     No.2  http://archive.ics.uci.edu/ml/datasets/Lenses 镜头数据集 ok
#     No.3  http://archive.ics.uci.edu/ml/datasets/Caesarian+Section+Classification+Dataset 剖腹产分类数据集 ok
#     No.4  http://archive.ics.uci.edu/ml/datasets/Teaching+Assistant+Evaluation 助教评估数据集 ok
#     No.5  http://archive.ics.uci.edu/ml/datasets/Balloons 气球数据集 ok
classifier(data_file=os.getcwd() + '/wifi.csv')
classifier(data_file=os.getcwd() + '/lenses.csv')
classifier(data_file=os.getcwd() + '/caesarian.csv')
classifier(data_file=os.getcwd() + '/teacher.csv')
classifier(data_file=os.getcwd() + '/balloons.csv')


