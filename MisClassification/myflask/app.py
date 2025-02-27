from flask import Flask,request,jsonify,make_response
import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import numpy as np
import miceforest as mf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import classification_report  # 分类报告
import xgboost as xgb
import json
from sklearn.ensemble import StackingClassifier

import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

with open('stacking.pkl', 'rb') as file:
    stacking = pickle.load(file)

app = Flask(__name__)


@app.route('/',methods=["GET"])
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/predict',methods=["POST"])
def predict():  # put application's code here
    try:
        my_json=request.get_json()
        df = pd.DataFrame.from_dict(my_json)

        for index, row in df.iterrows():
            for col in df.columns:
                if row[col] == '':
                    df.at[index, col] = 0

        if 'sample_id' in df.columns:
            df = df.drop('sample_id', axis=1)
        df = df.drop(['feature57', 'feature77', 'feature100'], axis=1)

        df = df.fillna(0)

        x = np.array(df.iloc[:, 0:104], dtype="double")
        selected_features = np.array([True,
                                      False,
                                      True,
                                      False,
                                      False,
                                      True,
                                      True,
                                      True,
                                      True,
                                      True,
                                      True,
                                      False,
                                      True,
                                      True,
                                      True,
                                      True,
                                      True,
                                      False,
                                      True,
                                      True,
                                      False,
                                      False,
                                      True,
                                      False,
                                      True,
                                      False,
                                      True,
                                      True,
                                      False,
                                      True,
                                      True,
                                      True,
                                      False,
                                      True,
                                      False,
                                      True,
                                      True,
                                      True,
                                      False,
                                      True,
                                      True,
                                      True,
                                      True,
                                      True,
                                      True,
                                      True,
                                      True,
                                      True,
                                      True,
                                      True,
                                      True,
                                      True,
                                      True,
                                      False,
                                      False,
                                      True,
                                      True,
                                      True,
                                      True,
                                      False,
                                      True,
                                      True,
                                      False,
                                      False,
                                      False,
                                      True,
                                      True,
                                      True,
                                      True,
                                      False,
                                      True,
                                      True,
                                      True,
                                      False,
                                      True,
                                      True,
                                      False,
                                      True,
                                      False,
                                      True,
                                      False,
                                      False,
                                      True,
                                      True,
                                      True,
                                      True,
                                      False,
                                      True,
                                      False,
                                      True,
                                      False,
                                      False,
                                      True,
                                      False,
                                      True,
                                      True,
                                      True,
                                      False,
                                      True,
                                      False,
                                      False,
                                      True,
                                      True,
                                      False], dtype=bool)
        x = x[:, selected_features]

        value=stacking.predict(x)



        # 绘制直方图
        plt.hist(value, bins=11, edgecolor='black')
        # 设置标题和标签
        plt.title('Result')
        plt.xlabel('Label')
        plt.ylabel('Count')
        # 生成唯一的文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"histogram_{timestamp}.png"
        # 构建保存路径
        save_path = os.path.join('./static/tempPic', filename)
        # 保存直方图到桌面路径
        plt.savefig(save_path)


        #把value换成json
        value = value.astype(int)
        json_list = [
            {"name": str(i), "label": int(item)}
            for i, item in enumerate(value)
        ]
        json_data = json.dumps(json_list)   
        
        # 创建包含路径和数据的字典
        result = {
            "image_path": filename,
            "data": json_data
        }

        # 将字典转换为 JSON 字符串
        merged_json = json.dumps(result)


        return merged_json


        # return jsonify(results={str(i): int(value[i]) for i in range(len(value))})

    except Exception as e:
        print(e)
        return "!"

@app.route('/train',methods=["POST"])
def train():  # put application's code here
    try:
        my_json=request.get_json()
        df = pd.DataFrame.from_dict(my_json)
        df = df.apply(pd.to_numeric, errors='coerce')

        # %%
        filter_feature = ['sample_id']
        # 遍历每一列，过滤无用特征
        for col in df.columns:
            # 检查该列是否所有值都为空
            if df[col].isnull().all():
                filter_feature.append(col)
                df.drop(col, axis=1)
            # 检查该列是否所有值都相同
            elif len(np.unique(df[col])) == 1:
                filter_feature.append(col)
        df = df.drop(filter_feature, axis=1)
        features = []
        for x in df.columns:  # 取特征
            if x not in filter_feature:
                features.append(x)
        df = df.iloc[:1000, :]
        y = np.array(df["label"], dtype="int")
        # %%
        # 创建内核数据集
        kernel = mf.ImputationKernel(
            data=df,
            save_all_iterations=False,
            random_state=42
        )
        # 执行多重插补
        kernel.mice(3)
        # 获取插补后的数据
        df = kernel.impute_new_data(new_data=df).complete_data(0)
        # %%
        x = np.array(df.iloc[:, 0:104], dtype="double")
        # 创建一个随机森林回归器
        estimator = RandomForestClassifier(random_state=42)
        # 创建一个RFE实例
        selector = RFE(estimator, n_features_to_select=70)
        # 拟合数据
        selector = selector.fit(x, y)
        # 查看选定的特征
        selected_features = selector.support_
        # 更新数据集，只保留选定的特征
        x = x[:, selected_features]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
        DT_clf = DecisionTreeClassifier()
        # 定义参数分布
        param_dist = {
            'max_depth': randint(1, 10),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 10)
        }
        # 创建一个随机搜索实例
        random_search = RandomizedSearchCV(DT_clf, param_distributions=param_dist, n_iter=10, cv=5)
        # 拟合数据
        random_search.fit(x_train, y_train)
        # 查看最佳参数
        DT_Best_params = random_search.best_params_
        # 随机调参
        # 定义超参数分布
        param_dist = {
            'n_estimators': randint(100, 1000),
            'learning_rate': uniform(0.01, 0.6),
            'max_depth': randint(2, 6),
            'min_child_weight': randint(1, 10),
            'subsample': uniform(0.5, 0.5),
            'gamma': uniform(0, 10),
            'colsample_bytree': uniform(0.5, 0.5),  # 表示均匀分布，最小值0.5，最大值1
            'reg_lambda': uniform(0.1, 10)
        }
        # 创建一个XGBClassifier实例
        model = xgb.XGBClassifier(objective='multi:softmax', num_class=6)
        # 创建一个RandomizedSearchCV实例
        search = RandomizedSearchCV(model, param_distributions=param_dist,
                                    n_iter=25, scoring='f1', cv=5)
        # 拟合数据
        search.fit(x_test, y_test)
        # 查看最优超参数组合
        best_params = search.best_params_
        # %%
        base_estimators = [('svc', SVC()),
                           ('dt', DecisionTreeClassifier(**DT_Best_params)),
                           ('mlp', MLPClassifier(solver='adam', learning_rate='adaptive', hidden_layer_sizes=(128, 64),
                                                 activation='tanh')),
                           ('xgb', xgb.XGBClassifier(objective='multi:softmax', num_class=6, **best_params))
                           ]
        # 创建元分类器
        meta_estimator = LogisticRegression()
        # 创建Stacking分类器
        stacking = StackingClassifier(estimators=base_estimators, final_estimator=meta_estimator)
        # 拟合数据
        stacking.fit(x_train, y_train)
        # 预测数据
        y_pred = stacking.predict(x_test)

        #保存模型文件
        # 获取当前的日期和时间
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"model_{timestamp}.pkl"
        # 保存模型到桌面
        # desktop_path = os.path.expanduser("~/Desktop")  # 展开波浪号为用户的家目录路径
        model_path = os.path.join('./static/tempModule', filename)
        with open(model_path, 'wb') as f:
            pickle.dump(stacking, f)


        #打分
        report = classification_report(y_test, y_pred, output_dict=True)  # 提取分类报告中的精确率、召回率和 F1 分数
        macro_precision = report['macro avg']['precision']
        macro_recall = report['macro avg']['recall']
        # 计算 MacroF1
        macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
        # 打印MacroF1
        result={
            "macro_f1": str(macro_f1),
            "pkl_path":filename
        }
        return result

    except Exception as e:
        print(e)
        return "!"


# @app.route('/get_json', methods=['GET'])
# def get_json():
#     try:
#         filename = request.args.get('filename')
#         src_file = os.path.join('./static/tempJSon', filename)#保存json的路径
#         dst_dir = os.path.expanduser('~/Desktop')
#         shutil.copy(src_file, dst_dir)
#         return 'File copied successfully'
#     except Exception as e:
#         print(e)
#         return "!"


import shutil
@app.route('/dldModule', methods=['GET'])
def copy_file():
    try:
        filename = request.args.get('moduleName')
        src_file = os.path.join('./static/module', filename)
        dst_dir = os.path.expanduser('~/Desktop')
        shutil.copy(src_file, dst_dir)
        return 'File copied successfully'
    except Exception as e:
        print(e)
        return "!"


if __name__ == '__main__':
    app.run()
