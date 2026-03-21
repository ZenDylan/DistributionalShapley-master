import os  # 导入操作系统接口模块
import sys  # 导入系统模块
import numpy as np  # 导入NumPy库，用于数值计算
import inspect  # 导入检查活动对象的模块
from scipy.stats import logistic  # 从scipy库导入logistic函数
from scipy.stats import spearmanr  # 从scipy库导入斯皮尔曼等级相关系数
from sklearn.naive_bayes import MultinomialNB  # 从scikit-learn导入多项式朴素贝叶斯
from sklearn.linear_model import LogisticRegression  # 从scikit-learn导入逻辑回归
from sklearn.linear_model import LinearRegression, Ridge  # 从scikit-learn导入线性回归和岭回归
from sklearn.metrics import r2_score  # 从scikit-learn导入R2分数
from sklearn.neural_network import MLPRegressor, MLPClassifier  # 从scikit-learn导入多层感知机回归器和分类器
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier  # 从scikit-learn导入随机森林、AdaBoost、梯度提升分类器
from sklearn.neighbors import KNeighborsClassifier  # 从scikit-learn导入K近邻分类器
from sklearn.tree import DecisionTreeClassifier  # 从scikit-learn导入决策树分类器
from sklearn.naive_bayes import MultinomialNB, GaussianNB  # 从scikit-learn导入多项式和高斯朴素贝叶斯
from sklearn.gaussian_process import GaussianProcessClassifier  # 从scikit-learn导入高斯过程分类器
from sklearn.svm import SVC, LinearSVC  # 从scikit-learn导入支持向量机分类器
from sklearn.base import clone  # 从scikit-learn导入模型克隆工具
import inspect  # 再次导入inspect模块（冗余）
from Shapley import ShapNN, CShapNN  # 从Shapley模块导入ShapNN和CShapNN类
from multiprocessing import dummy as multiprocessing  # 导入多进程模块的线程版本
from sklearn.metrics import roc_auc_score, f1_score  # 从scikit-learn导入ROC AUC和F1分数
import warnings  # 导入警告控制模块
import tensorflow as tf  # 导入TensorFlow深度学习框架
import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块
from sklearn.ensemble import RandomForestRegressor  # 从scikit-learn导入随机森林回归器
from sklearn.linear_model import Lasso, Ridge  # 从scikit-learn导入Lasso和Ridge回归
from sklearn.linear_model import Ridge  # 再次导入Ridge回归（冗余）
from sklearn.neighbors import KNeighborsRegressor  # 从scikit-learn导入K近邻回归器
from sklearn.model_selection import cross_validate  # 从scikit-learn导入交叉验证


def one_hotisze(X, impute=True, missing_key=-10000):  # 对输入数据进行独热编码
    
    X_oh = []  # 初始化独热编码后的列表
    for col in range(X.shape[-1]):  # 遍历数据的每一列
        column = X[:, col]  # 获取当前列
        vals = np.sort(list(set(column)))  # 获取列中的唯一值并排序
        if impute and missing_key in vals:  # 如果需要插补缺失值且缺失值存在
            counts = np.zeros(len(vals))  # 初始化计数值数组
            for i in range(len(vals)):  # 遍历唯一值
                counts[i] = np.sum(column == vals[i])  # 统计每个值的出现次数
            column[column==missing_key] = vals[np.argmax(counts)]  # 用众数替换缺失值
        column_oh = np.zeros((len(column), len(vals)))  # 初始化独热编码矩阵
        for i, val in enumerate(np.sort(vals)):  # 遍历排序后的唯一值
            column_oh[column==val, i] = 1  # 对相应位置进行独热编码
        X_oh.append(column_oh)  # 将编码后的列添加到列表
    return np.concatenate(X_oh, -1)  # 拼接所有列的独热编码结果


def convergence_plots(marginals):  # 绘制边际贡献的收敛图
    
    plt.rcParams['figure.figsize'] = 15, 15  # 设置图像大小
    for i, idx in enumerate(np.arange(min(25, marginals.shape[-1]))):  # 遍历前25个数据点的边际贡献
        plt.subplot(5,5,i+1)  # 创建子图
        plt.plot(np.cumsum(marginals[:, idx])/np.arange(1, len(marginals)+1))  # 绘制累积平均值    
        
    
def is_integer(array):  # 检查数组是否全为整数
    return (np.equal(np.mod(array, 1), 0).mean()==1)  # 通过取模运算判断


def is_fitted(model):  # 检查模型是否已经训练过
        """Checks if model object has any attributes ending with an underscore"""  # 检查模型对象是否有以下划线结尾的属性
        return 0 < len( [k for k,v in inspect.getmembers(model) if k.endswith('_') and not k.startswith('__')] )  # 返回符合条件的属性数量是否大于0


def return_model(mode, **kwargs):  # 根据指定的模式返回一个模型实例
    
    
    if inspect.isclass(mode):  # 如果mode是一个类
        assert getattr(mode, 'fit', None) is not None, 'Custom model family should have a fit() method'  # 检查该类是否有fit方法
        model = mode(**kwargs)  # 实例化该类
    elif mode=='logistic':  # 如果模式是'logistic'
        solver = kwargs.get('solver', 'liblinear')  # 获取求解器
        n_jobs = kwargs.get('n_jobs', None)  # 获取并行任务数
        C = kwargs.get('C', 1.)  # 获取正则化强度
        max_iter = kwargs.get('max_iter', 5000)  # 获取最大迭代次数
        model = LogisticRegression(solver=solver, n_jobs=n_jobs, C=C,  # 创建逻辑回归模型
                                 max_iter=max_iter, random_state=666)
    elif mode=='Tree':  # 如果模式是'Tree'
        model = DecisionTreeClassifier(random_state=666)  # 创建决策树分类器
    elif mode=='RandomForest':  # 如果模式是'RandomForest'
        n_estimators = kwargs.get('n_estimators', 50)  # 获取树的数量
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=666)  # 创建随机森林分类器
    elif mode=='GB':  # 如果模式是'GB'
        n_estimators = kwargs.get('n_estimators', 50)  # 获取树的数量
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=666)  # 创建梯度提升分类器
    elif mode=='AdaBoost':  # 如果模式是'AdaBoost'
        n_estimators = kwargs.get('n_estimators', 50)  # 获取树的数量
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=666)  # 创建AdaBoost分类器
    elif mode=='SVC':  # 如果模式是'SVC'
        kernel = kwargs.get('kernel', 'rbf')  # 获取核函数
        model = SVC(kernel=kernel, random_state=666, probability=True)  # 创建支持向量机分类器
    elif mode=='LinearSVC':  # 如果模式是'LinearSVC'
        model = LinearSVC(loss='hinge', random_state=666)  # 创建线性支持向量机分类器
    elif mode=='GP':  # 如果模式是'GP'
        model = GaussianProcessClassifier(random_state=666)  # 创建高斯过程分类器
    elif mode=='KNN':  # 如果模式是'KNN'
        n_neighbors = kwargs.get('n_neighbors', 5)  # 获取邻居数
        model = KNeighborsClassifier(n_neighbors=n_neighbors)  # 创建K近邻分类器
    elif mode=='NB':  # 如果模式是'NB'
        model = MultinomialNB()  # 创建多项式朴素贝叶斯分类器
    elif mode=='linear':  # 如果模式是'linear'
        model = LinearRegression()  # 创建线性回归模型
    elif mode=='ridge':  # 如果模式是'ridge'
        alpha = kwargs.get('alpha', 1.0)  # 获取正则化强度
        model = Ridge(alpha=alpha, random_state=666)  # 创建岭回归模型
    elif 'conv' in mode:  # 如果模式包含'conv'
        tf.reset_default_graph()  # 重置TensorFlow默认计算图
        address = kwargs.get('address', 'weights/conv')  # 获取模型保存地址
        hidden_units = kwargs.get('hidden_layer_sizes', [20])  # 获取隐藏层大小
        activation = kwargs.get('activation', 'relu')  # 获取激活函数
        weight_decay = kwargs.get('weight_decay', 1e-4)  # 获取权重衰减系数
        learning_rate = kwargs.get('learning_rate', 0.001)  # 获取学习率
        max_iter = kwargs.get('max_iter', 1000)  # 获取最大迭代次数
        dropout = kwargs.get('dropout', 0.)  # 获取dropout率
        early_stopping= kwargs.get('early_stopping', 10)  # 获取早停轮数
        warm_start = kwargs.get('warm_start', False)  # 获取是否热启动
        batch_size = kwargs.get('batch_size', 256)  # 获取批次大小
        kernel_sizes = kwargs.get('kernel_sizes', [5])  # 获取卷积核大小
        strides = kwargs.get('strides', [5])  # 获取步长
        channels = kwargs.get('channels', [1])  # 获取通道数
        validation_fraction = kwargs.get('validation_fraction', 0.)  # 获取验证集比例
        global_averaging = kwargs.get('global_averaging', 0.)  # 获取是否使用全局平均池化
        optimizer = kwargs.get('optimizer', 'sgd')  # 获取优化器
        if mode=='conv':  # 如果是'conv'（分类）
            model = CShapNN(mode='classification', batch_size=batch_size, max_epochs=max_iter,  # 创建CShapNN分类模型
                          learning_rate=learning_rate, dropout=dropout,
                          weight_decay=weight_decay, validation_fraction=validation_fraction,
                          early_stopping=early_stopping,
                         optimizer=optimizer, warm_start=warm_start, address=address,
                          hidden_units=hidden_units,
                          strides=strides, global_averaging=global_averaging,
                         kernel_sizes=kernel_sizes, channels=channels, random_seed=666)
        elif mode=='conv_reg':  # 如果是'conv_reg'（回归）
            model = CShapNN(mode='regression', batch_size=batch_size, max_epochs=max_iter,  # 创建CShapNN回归模型
                          learning_rate=learning_rate, dropout=dropout,
                          weight_decay=weight_decay, validation_fraction=validation_fraction,
                          early_stopping=early_stopping,
                         optimizer=optimizer, warm_start=warm_start, address=address,
                          hidden_units=hidden_units,
                          strides=strides, global_averaging=global_averaging,
                         kernel_sizes=kernel_sizes, channels=channels, random_seed=666)
    elif 'NN' in mode:  # 如果模式包含'NN'
        solver = kwargs.get('solver', 'adam')  # 获取求解器
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (20,))  # 获取隐藏层大小
        if isinstance(hidden_layer_sizes, list):  # 如果是列表
            hidden_layer_sizes = tuple(hidden_layer_sizes)  # 转换为元组
        activation = kwargs.get('activation', 'relu')  # 获取激活函数
        learning_rate_init = kwargs.get('learning_rate', 0.001)  # 获取初始学习率
        max_iter = kwargs.get('max_iter', 5000)  # 获取最大迭代次数
        early_stopping= kwargs.get('early_stopping', False)  # 获取是否早停
        warm_start = kwargs.get('warm_start', False)  # 获取是否热启动
        batch_size = kwargs.get('batch_size', 'auto')  # 获取批次大小
        
        if mode=='NN':  # 如果是'NN'（分类）
            model = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layer_sizes,  # 创建MLP分类器
                                activation=activation, learning_rate_init=learning_rate_init,
                                warm_start = warm_start, max_iter=max_iter,
                                early_stopping=early_stopping, batch_size=batch_size, random_state=666)
        if mode=='NN_reg':  # 如果是'NN_reg'（回归）
            model = MLPRegressor(solver=solver, hidden_layer_sizes=hidden_layer_sizes,  # 创建MLP回归器
                                activation=activation, learning_rate_init=learning_rate_init,
                                warm_start=warm_start, max_iter=max_iter, early_stopping=early_stopping,
                                batch_size=batch_size, random_state=666)
    else:  # 如果模式无效
        raise ValueError("Invalid mode!")  # 抛出错误
    return model  # 返回创建的模型



def generate_features(latent, dependency):  # 根据潜在变量生成高阶特征

    features = []  # 初始化特征列表
    n = latent.shape[0]  # 获取样本数量
    exp = latent  # 初始指数项
    holder = latent  # 初始持有项
    for order in range(1, dependency+1):  # 遍历指定的依赖阶数
        features.append(np.reshape(holder,[n,-1]))  # 将当前阶的特征展平并添加到列表
        exp = np.expand_dims(exp,-1)  # 扩展指数项的维度
        holder = exp * np.expand_dims(holder,1)  # 计算下一阶的特征
    return np.concatenate(features,axis=-1)  # 拼接所有阶的特征并返回


def label_generator(problem, X, param, difficulty=1, beta=None, important=None):  # 生成合成数据的标签
        
    if important is None or important > X.shape[-1]:  # 如果未指定重要特征数或超出范围
        important = X.shape[-1]  # 使用所有特征
    dim_latent = sum([important**i for i in range(1, difficulty+1)])  # 计算潜在空间的维度
    if beta is None:  # 如果未提供beta系数
        beta = np.random.normal(size=[1, dim_latent])  # 随机生成beta系数
    important_dims = np.random.choice(X.shape[-1], important, replace=False)  # 随机选择重要特征的维度
    funct_init = lambda inp: np.sum(beta * generate_features(inp[:,important_dims], difficulty), -1)  # 定义生成真实潜在值的函数
    batch_size = max(100, min(len(X), 10000000//dim_latent))  # 根据内存计算合适的批次大小
    y_true = np.zeros(len(X))  # 初始化真实潜在值数组
    while True:  # 循环直到成功计算
        try:  # 尝试计算
            for itr in range(int(np.ceil(len(X)/batch_size))):  # 分批次计算
                y_true[itr * batch_size: (itr+1) * batch_size] = funct_init(  # 计算批次的真实潜在值
                    X[itr * batch_size: (itr+1) * batch_size])
            break  # 成功后跳出循环
        except MemoryError:  # 如果发生内存错误
            batch_size = batch_size//2  # 减小批次大小
    mean, std = np.mean(y_true), np.std(y_true)  # 计算真实潜在值的均值和标准差
    funct = lambda x: (np.sum(beta * generate_features(  # 定义标准化的函数
        x[:, important_dims], difficulty), -1) - mean) / std
    y_true = (y_true - mean) / std  # 对真实潜在值进行标准化
    if problem is 'classification':  # 如果是分类问题
        y_true = logistic.cdf(param * y_true)  # 使用logistic函数转换为概率
        y = (np.random.random(X.shape[0]) < y_true).astype(int)  # 根据概率生成二分类标签
    elif problem is 'regression':  # 如果是回归问题
        y = y_true + param * np.random.normal(size=len(y_true))  # 添加高斯噪声生成回归标签
    else:  # 如果问题类型无效
        raise ValueError('Invalid problem specified!')  # 抛出错误
    return beta, y, y_true, funct  # 返回beta系数、生成的标签、真实的潜在值和生成函数


def one_iteration(clf, X, y, X_test, y_test, mean_score, tol=0.0, c=None, metric='accuracy'):  # TMC-Shapley的单次迭代
    """Runs one iteration of TMC-Shapley."""  # 运行一次TMC-Shapley迭代
    
    if metric == 'auc':  # 如果指标是AUC
        def score_func(clf, a, b):  # 定义评分函数
            return roc_auc_score(b, clf.predict_proba(a)[:,1])
    elif metric == 'accuracy':  # 如果指标是准确率
        def score_func(clf, a, b):  # 定义评分函数
            return clf.score(a, b)
    else:  # 如果指标无效
        raise ValueError("Wrong metric!")  # 抛出错误
    if c is None:  # 如果未提供数据源分组
        c = {i:np.array([i]) for i in range(len(X))}  # 每个点都是一个独立的组
    idxs, marginal_contribs = np.random.permutation(len(c.keys())), np.zeros(len(X))  # 随机排列索引并初始化边际贡献
    new_score = np.max(np.bincount(y)) * 1./len(y) if np.mean(y//1 == y/1)==1 else 0.  # 计算初始得分（随机猜测）
    start = 0  # 起始点
    if start:  # 如果起始点非0（此代码块当前不会执行）
        X_batch, y_batch =\
        np.concatenate([X[c[idx]] for idx in idxs[:start]]), np.concatenate([y[c[idx]] for idx in idxs[:start]])
    else:  # 否则
        X_batch, y_batch = np.zeros((0,) +  tuple(X.shape[1:])), np.zeros(0).astype(int)  # 初始化空批次
    for n, idx in enumerate(idxs[start:]):  # 遍历排列后的索引
        try:  # 尝试克隆模型
            clf = clone(clf)
        except:  # 如果失败
            clf.fit(np.zeros((0,) +  X.shape[1:]), y)  # 用空数据拟合以重新初始化
        old_score = new_score  # 保存上一轮得分
        X_batch, y_batch = np.concatenate([X_batch, X[c[idx]]]), np.concatenate([y_batch, y[c[idx]]])  # 将当前点加入批次
        with warnings.catch_warnings():  # 捕获警告
            warnings.simplefilter("ignore")  # 忽略警告
            try:  # 尝试训练和评估
                clf.fit(X_batch, y_batch)  # 训练模型
                temp_score = score_func(clf, X_test, y_test)  # 评估模型
                if temp_score>-1 and temp_score<1.: #Removing measningless r2 scores  # 移除无意义的R2分数
                    new_score = temp_score  # 更新得分
            except:  # 如果失败
                continue  # 继续下一次循环
        marginal_contribs[c[idx]] = (new_score - old_score)/len(c[idx])  # 计算边际贡献
        if np.abs(new_score - mean_score)/mean_score < tol:  # 如果满足截断条件
            break  # 提前终止
    return marginal_contribs, idxs  # 返回边际贡献和排列索引


def marginals(clf, X, y, X_test, y_test, c=None, tol=0., trials=3000, mean_score=None, metric='accuracy'):  # 计算多次迭代的边际贡献
    
    if metric == 'auc':  # 如果指标是AUC
        def score_func(clf, a, b):  # 定义评分函数
            return roc_auc_score(b, clf.predict_proba(a)[:,1])
    elif metric == 'accuracy':  # 如果指标是准确率
        def score_func(clf, a, b):  # 定义评分函数
            return clf.score(a, b)
    else:  # 如果指标无效
        raise ValueError("Wrong metric!")  # 抛出错误
    if mean_score is None:  # 如果未提供平均得分
        accs = []  # 初始化准确率列表
        for _ in range(100):  # 循环100次
            bag_idxs = np.random.choice(len(y_test), len(y_test))  # 从测试集中有放回抽样
            accs.append(score_func(clf, X_test[bag_idxs], y_test[bag_idxs]))  # 计算得分
        mean_score = np.mean(accs)  # 计算平均得分
    marginals, idxs = [], []  # 初始化边际贡献和索引列表
    for trial in range(trials):  # 循环指定的试验次数
        if 10*(trial+1)/trials % 1 == 0:  # 每10%进度打印一次
            print('{} out of {}'.format(trial + 1, trials))  # 打印进度
        marginal, idx = one_iteration(clf, X, y, X_test, y_test, mean_score, tol=tol, c=c, metric=metric)  # 运行一次迭代
        marginals.append(marginal)  # 添加边际贡献
        idxs.append(idx)  # 添加排列索引
    return np.array(marginals), np.array(idxs)  # 返回所有迭代的结果


def shapley(mode, X, y, X_test, y_test, stop=None, tol=0., trials=3000, **kwargs):  # 计算Shapley值（已弃用或不完整）
    
    try:  # 尝试执行
        vals = np.zeros(len(X))  # 初始化Shapley值数组
        example_idxs = np.random.choice(len(X), min(25, len(X)), replace=False)  # 随机选择示例索引
        example_marginals = np.zeros((trials, len(example_idxs)))  # 初始化示例的边际贡献数组
        for i in range(trials):  # 循环指定的试验次数
            print(i)  # 打印当前次数
            output = one_pass(mode, X, y, X_test, y_test, tol=tol, stop=stop, **kwargs)  # 调用一个未定义的函数 one_pass
            example_marginals[i] = output[0][example_idxs]  # 记录示例的边际贡献
            vals = vals/(i+1) + output[0]/(i+1)  # 增量式更新Shapley值
        return vals, example_marginals  # 返回Shapley值和示例的边际贡献
    except KeyboardInterrupt:  # 捕获键盘中断
        print('Interrupted!')  # 打印中断信息
        return vals, example_marginals  # 返回当前计算的结果

    
def early_stopping(marginals, idxs, stopping):  # 对边际贡献应用早停
    
    stopped_marginals = np.zeros_like(marginals)  # 初始化早停后的边际贡献数组
    for i in range(len(marginals)):  # 遍历每次迭代的结果
        stopped_marginals[i][idxs[i][:stopping]] = marginals[i][idxs[i][:stopping]]  # 只保留截断点之前的边际贡献
    return np.mean(stopped_marginals, 0)  # 返回所有迭代的平均值


def error(mem):  # 计算收敛误差
    
    if len(mem) < 100:  # 如果迭代次数少于100
        return 1.0  # 返回最大误差1.0
    all_vals = (np.cumsum(mem, 0)/np.reshape(np.arange(1, len(mem)+1), (-1,1)))[-100:]  # 计算最后100次迭代的累积平均值
    errors = np.mean(np.abs(all_vals[-100:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)  # 计算相对误差
    return np.max(errors)  # 返回最大误差


def my_accuracy_score(clf, X, y):  # 自定义准确率评分函数
    
    probs = clf.predict_proba(X)  # 预测概率
    predictions = np.argmax(probs, -1)  # 获取最大概率的类别
    return np.mean(np.equal(predictions, y))  # 计算准确率


def my_f1_score(clf, X, y):  # 自定义F1评分函数
    
    predictions = clf.predict(X)  # 预测类别
    if len(set(y)) == 2:  # 如果是二分类
        return f1_score(y, predictions)  # 计算二分类F1分数
    return f1_score(y, predictions, average='macro')  # 计算多分类的宏平均F1分数


def my_auc_score(clf, X, y):  # 自定义AUC评分函数
    
    probs = clf.predict_proba(X)  # 预测概率
    if len(probs.shape) == 2:
        return roc_auc_score(y, probs[:,1])
    return roc_auc_score(y, probs)


def my_xe_score(clf, X, y):  # 自定义交叉熵评分函数
    
    probs = clf.predict_proba(X)  # 预测概率
    true_probs = probs[np.arange(len(y)), y]  # 获取真实类别的概率
    true_log_probs = np.log(np.clip(true_probs, 1e-12, None))  # 计算对数概率，避免log(0)
    return np.mean(true_log_probs)  # 返回平均对数概率（负交叉熵）


def portion_performance(dshap, order, points, X_new, y_new, X_init, y_init, X_test, y_test):  # 计算添加部分数据后的性能
    
    dshap.model.fit(X_init, y_init)  # 在初始数据上训练模型
    val_init = dshap.value(dshap.model, dshap.metric, X=X_test, y=y_test)  # 计算初始性能
    vals = [val_init]  # 初始化性能列表
    for point in points:  # 遍历指定的点数
        if not point:  # 如果点数为0
            continue  # 跳过
        dshap.model.fit(np.concatenate([X_init, X_new[order[:point]]]),  # 在初始数据和新添加的数据上训练
                       np.concatenate([y_init, y_new[order[:point]]]))
        vals.append(dshap.value(dshap.model, dshap.metric, X=X_test, y=y_test))  # 计算并记录新性能
    return np.array(vals)  # 返回性能数组


def find_best_regressor(X, y, vals, cv=10, verbose=False):  # 查找最佳回归器来预测Shapley值
    
    def return_model(model_family):  # 内部函数：根据模型族返回模型和参数
        
        if model_family == 'Ridge':  # 如果是岭回归
            params = 10 ** np.arange(0, -6, 1).astype(float)  # 定义超参数范围
            model = lambda param: Ridge(alpha=param)  # 定义模型
        if model_family == 'Lasso':  # 如果是Lasso回归
            params = 10 ** np.arange(0, -6, 1).astype(float)  # 定义超参数范围
            model = lambda param: Lasso(alpha=param)  # 定义模型
        if model_family == 'RF':  # 如果是随机森林
            params = [5, 10, 25, 50, 100]  # 定义超参数范围
            model = lambda param: RandomForestRegressor(n_estimators=param)  # 定义模型
        if model_family == 'KNN':  # 如果是K近邻
            params = np.arange(1, 9, 2)  # 定义超参数范围
            model = lambda param: KNeighborsRegressor(n_neighbors=param)  # 定义模型
        return model, params  # 返回模型和参数
    
    regs = {}  # 初始化回归器字典
    for label in np.sort(list(set(y))):  # 按类别分别寻找最佳回归器
        best_score = -100  # 初始化最佳分数
        label_idxs = np.where(y == label)[0]  # 获取当前类别的索引
        if len(label_idxs) == 1:  # 如果该类别只有一个样本
            model, params = return_model('RF')  # 使用随机森林
            best_reg = model(10)  # 创建模型
            best_reg.fit(X[label_idxs], vals[label_idxs])  # 训练模型
            regs[label] = best_reg  # 保存模型
            continue  # 继续下一个类别
        for model_family in ['RF', 'Lasso', 'Ridge', 'KNN']:  # 遍历不同的模型族
            model, params = return_model(model_family)  # 获取模型和参数
            for param in params:  # 遍历超参数
                if verbose:  # 如果需要打印详细信息
                    print(model_family, param)  # 打印模型和参数
                reg = model(param)  # 创建模型实例
                cv_scores = cross_validate(  # 进行交叉验证
                    reg,
                    X[label_idxs],
                    vals[label_idxs],
                    cv=min(len(label_idxs), cv))
                if np.mean(cv_scores['test_score']) > best_score:  # 如果分数更高
                    best_score = np.mean(cv_scores['test_score'])  # 更新最佳分数
                    best_reg = reg  # 更新最佳回归器
        print(label, best_reg, best_score)  # 打印最佳结果
        best_reg.fit(X[label_idxs], vals[label_idxs])  # 在所有数据上训练最佳回归器
        regs[label] = best_reg  # 保存训练好的回归器
    return regs  # 返回所有类别的最佳回归器


def predict_vals(X, y, regs):  # 使用训练好的回归器预测数据值
    
    predicted_vals = np.zeros(len(X))  # 初始化预测值数组
    for label in set(y):  # 遍历所有类别
        label_idxs = np.where(y == label)[0]  # 获取当前类别的索引
        predicted_vals[label_idxs] = regs[label].predict(X[label_idxs])  # 使用对应类别的回归器进行预测
    return predicted_vals  # 返回所有样本的预测值
    
def s_regress(model, vals, alpha, init):  # 使用插值/回归模型平滑值（似乎未完全实现或在此上下文中不明确）
    
    predicted_vals = np.zeros(truncation)  # 初始化预测值数组 (truncation未定义)
    t = int(truncation ** (1./(1 + alpha)))  # 计算采样点数 (truncation未定义)
    t_idxs = (np.arange(t) ** (1+alpha)).astype(int)  # 生成采样点的索引
    t_idxs = np.sort(np.array(list(set(t_idxs))))  # 去重并排序
    t_idxs = t_idxs[t_idxs>=init]  # 只保留大于等于初始点的索引
    x = t_idxs  # 采样点的x坐标
    y = vals[t_idxs]  # 采样点的y坐标
    model.fit(x, y)  # 训练模型
    predicted_vals = model.predict(np.arange(truncation))  # 预测所有点的值 (truncation未定义)
    predicted_vals[:init] = vals[:init]  # 保持初始点的值不变
    predicted_vals[t_idxs] = vals[t_idxs]  # 保持采样点的值不变
    return predicted_vals  # 返回平滑/插值后的值

def interpolator(model):  # 根据模型名称返回一个插值器对象（依赖于未导入的类）
    
    if 'spline_' in model:  # 如果是样条插值
        return Spline(model[7:])  # 返回Spline对象 (Spline未定义)
    if model == 'lin':  # 如果是线性插值
        return LinInt()  # 返回LinInt对象 (LinInt未定义)
    if 'poly_' in model:  # 如果是多项式插值
        return Poly(int(model[5:]))  # 返回Poly对象 (Poly未定义)
    if model == 'NN':  # 如果是神经网络
        return NN(activation='logistic')  # 返回NN对象 (NN未定义)
    raise ValueError('Invalid Model')  # 如果模型无效则抛出错误

    
def compute_eff(alpha, truncation):  # 计算采样效率（似乎与s_regress相关）
    
    t = int(truncation ** (1./(1 + alpha)))  # 计算采样点数
    t_idxs = (np.arange(t) ** (1+alpha)).astype(int)  # 生成采样点索引
    t_idxs = np.sort(np.array(list(set(t_idxs))))  # 去重并排序
    t_idxs = t_idxs[t_idxs>=init]  # 只保留大于等于初始点的索引 (init未定义)
    return (np.sum(np.arange(init)) + np.sum(t_idxs)) / np.sum(np.arange(truncation))   # 计算采样代价与总代价的比率


def reverse_compute_eff(x, truncation):  # 反向计算采样效率对应的alpha值
    
    a1 = 0.  # alpha下界
    a2 = 10.  # alpha上界
    while True:  # 循环进行二分查找
        if compute_eff((a1+a2)/2, truncation) < x:  # 如果中间点的效率小于目标
            a1, a2 = a1, (a1+a2)/2  # 缩小上界
        else:  # 否则
            a1, a2 = (a1+a2)/2, a2  # 缩小下界
        if a2 - a1 < 1e-4:  # 如果区间足够小
            break  # 退出循环
    return (a1+a2)/2  # 返回alpha的近似值

def performance_plots(npoints, points, perf):  # 绘制性能对比图
    
    plt.rcParams['font.size'] = 15  # 设置全局字体大小
    fig = plt.figure(figsize = (16, 8))  # 创建图像
    markers = ['-', ':', '-.', '--']  # 定义线型
    colors = ['b', 'r', 'g', 'orange']  # 定义颜色
    default_legends = ['Dist-Shapley', 'Random', 'LOO', 'TMC-Shapley']  # 定义默认图例
    
    plt.subplot(1, 2, 1)  # 创建第一个子图（移除高价值点）
    pos_keys = ['pos_dist', 'rnd', 'pos_loo', 'pos_tmc']  # 定义高价值点的键
    legends = []  # 初始化图例列表
    for i, (key, legend) in enumerate(zip(pos_keys, default_legends)):  # 遍历键和图例
        if key not in perf:  # 如果性能字典中没有该键
            continue  # 跳过
        plt.plot(points / npoints * 100, 100 * np.array(perf[key]), markers[i], color=colors[i], lw=8)  # 绘制性能曲线
        legends.append(legend)  # 添加图例
    plt.legend(legends, fontsize=25)  # 显示图例
    res = (points[-1] - points[0]) / (len(points) - 1) if len(points) > 1 else 1 # 计算x轴刻度间隔
    plt.xticks(100 * np.linspace(points[0], points[-1] + res, 6) / npoints)  # 设置x轴刻度
    plt.xlabel('Fraction of points removed (%)', fontsize=25)  # 设置x轴标签
    min_p = np.min([np.min(perf[k]) for k in perf if k in pos_keys]) if any(k in perf for k in pos_keys) else 0 # 计算y轴最小值
    max_p = np.max([np.max(perf[k]) for k in perf if k in pos_keys]) if any(k in perf for k in pos_keys) else 1 # 计算y轴最大值
    p_res = 0.01  # 默认y轴刻度间隔
    for p in [0.2, 0.1, 0.05, 0.03, 0.02, 0.01]:  # 寻找合适的刻度间隔
        num_p = np.ceil(max_p / p) - np.floor(min_p / p)
        if num_p >= 4 and num_p <= 8:
            p_res = p
            break
    plt.yticks(100 * np.arange(np.floor(min_p/p_res) * p_res, np.ceil(max_p/p_res) * p_res + 0.01, p_res))  # 设置y轴刻度
    plt.ylabel('Performance (%)', fontsize=25)  # 设置y轴标签
    
    plt.subplot(1, 2, 2)  # 创建第二个子图（移除低价值点）
    legends = []  # 初始化图例列表
    neg_keys = ['neg_dist', 'rnd', 'neg_loo', 'neg_tmc']  # 定义低价值点的键
    for i, (key, legend) in enumerate(zip(neg_keys, default_legends)):  # 遍历键和图例
        if key not in perf:  # 如果性能字典中沒有该键
            continue  # 跳过
        plt.plot(points / npoints * 100, 100 * np.array(perf[key]), markers[i], color=colors[i], lw=8)  # 绘制性能曲线
        legends.append(legend)  # 添加图例
    plt.legend(legends, fontsize=25)  # 显示图例
    res = (points[-1] - points[0]) / (len(points) - 1) if len(points) > 1 else 1 # 计算x轴刻度间隔
    plt.xticks(100 * np.linspace(points[0], points[-1] + res, 6) / npoints)  # 设置x轴刻度
    plt.xlabel('Fraction of points added (%)', fontsize=25)  # 设置x轴标签 (标签似乎应为removed)
    min_p = np.min([np.min(perf[k]) for k in perf if k in neg_keys]) if any(k in perf for k in neg_keys) else 0 # 计算y轴最小值
    max_p = np.max([np.max(perf[k]) for k in perf if k in neg_keys]) if any(k in perf for k in neg_keys) else 1 # 计算y轴最大值
    p_res = 0.01  # 默认y轴刻度间隔
    for p in [0.2, 0.1, 0.05, 0.03, 0.02, 0.01]:  # 寻找合适的刻度间隔
        num_p = np.ceil(max_p / p) - np.floor(min_p / p)
        if num_p >= 4 and num_p <= 8:
            p_res = p
            break
    plt.yticks(100 * np.arange(np.floor(min_p/p_res) * p_res, np.ceil(max_p/p_res) * p_res + 0.01, p_res))  # 设置y轴刻度
    plt.ylabel('Performance (%)', fontsize=25)  # 设置y轴标签
