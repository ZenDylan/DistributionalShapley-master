
#______________________________________PEP8____________________________________  # PEP8风格检查（非代码）
#_______________________________________________________________________  # 分隔线（非代码）
import matplotlib  # 导入matplotlib库
# matplotlib.use('Agg')  # 注释掉Agg后端，让Jupyter自动选择合适的后端
import numpy as np  # 导入NumPy库，用于数值计算
import os  # 导入操作系统接口模块
import tensorflow as tf  # 导入TensorFlow深度学习框架
import sys  # 导入系统模块
from shap_utils import *  # 从shap_utils模块导入所有内容
from scipy.stats import spearmanr  # 从scipy库导入斯皮尔曼等级相关系数
import shutil  # 导入高级文件操作模块
from sklearn.base import clone  # 从scikit-learn导入模型克隆工具
import time  # 导入时间模块
import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块
import itertools  # 导入迭代器工具
import inspect  # 导入检查活动对象的模块
import _pickle as pkl  # 导入pickle模块，用于序列化
from sklearn.metrics import f1_score, roc_auc_score  # 从scikit-learn导入评估指标
import socket  # 导入套接字模块，用于网络编程
import warnings  # 导入警告控制模块
import math  # 导入数学函数模块，用于概率分布计算
warnings.filterwarnings("ignore")  # 忽略所有警告



class DistShap(object):  # 定义DistShap类，用于计算分布式的Shapley值
    
    @staticmethod
    def _generate_paper_distribution(n):  # 生成论文中的理论概率分布P(s)
        """Generate probability distribution over coalition sizes according to S-SVARM paper."""
        dist = [0 for i in range(n + 1)]  # 初始化概率分布数组
        
        if n % 2 == 0:  # 如果n为偶数
            nlogn = n * math.log(n)  # 计算n*log(n)
            H = sum([1 / s for s in range(1, int(n / 2))])  # 计算调和数H_{n/2-1}
            nominator = nlogn - 1  # 分子：n*log(n) - 1
            denominator = 2 * nlogn * (H - 1)  # 分母：2*n*log(n)*(H_{n/2-1} - 1)
            frac = nominator / denominator  # 计算分数部分
            
            for s in range(2, int(n / 2)):  # 对于s∈[2, n/2-1]
                dist[s] = frac / s  # P(s) = (n*log(n) - 1) / (2*s*n*log(n)*(H_{n/2-1} - 1))
                dist[n - s] = frac / s  # P(n-s) = P(s)，保持对称性
            
            dist[int(n / 2)] = 1 / nlogn  # P(n/2) = 1 / (n*log(n))
            
        else:  # 如果n为奇数
            H = sum([1 / s for s in range(1, int((n - 1) / 2 + 1))])  # 计算调和数H_{(n-1)/2}
            frac = 1 / (2 * (H - 1))  # 计算分数部分：1 / (2*(H_{(n-1)/2} - 1))
            
            for s in range(2, int((n - 1) / 2 + 1)):  # 对于s∈[2, (n-1)/2]
                dist[s] = frac / s  # P(s) = 1 / (2*s*(H_{(n-1)/2} - 1))
                dist[n - s] = frac / s  # P(n-s) = P(s)，保持对称性
        
        return dist  # 返回概率分布数组
    
    @staticmethod
    def _generate_distribution(dist_type, n):  # 根据类型生成概率分布
        """Generate probability distribution over coalition sizes."""
        if dist_type == "paper":  # 如果使用论文中的理论分布
            return DistShap._generate_paper_distribution(n)  # 调用论文分布生成函数
        elif dist_type == "uniform":  # 如果使用均匀分布
            dist = [0 for i in range(n + 1)]  # 初始化分布数组
            for s in range(2, n - 1):  # 对于s∈[2, n-2]
                dist[s] = 1 / (n - 3)  # 均匀分布：P(s) = 1/(n-3)
            return dist  # 返回均匀分布
        else:  # 如果类型未知
            raise ValueError(f"Unknown distribution type: {dist_type}")  # 抛出错误
    
    def __init__(self, X, y, X_test, y_test, num_test, X_tot=None, y_tot=None,  # 初始化函数
                 sources=None,   # 数据源分组
                 sample_weight=None, directory=None, problem='classification',  # 样本权重、结果目录、问题类型
                 model_family='logistic', metric='accuracy', seed=None,  # 模型类型、评估指标、随机种子
                 overwrite=False,  # 是否覆盖已有结果
                 **kwargs):  # 其他模型参数
        """
        Args:
            X: Data covariates  # 训练数据特征
            y: Data labels  # 训练数据标签
            X_test: Test+Held-out covariates  # 测试集和留出集特征
            y_test: Test+Held-out labels  # 测试集和留出集标签
            sources: An array or dictionary assiging each point to its group.  # 数据点分组
                If None, evey points gets its individual value.  # 如果为None，每个点都是一个独立的组
            samples_weights: Weight of train samples in the loss function  # 训练样本在损失函数中的权重
                (for models where weighted training method is enabled.)  # （适用于支持加权训练的模型）
            num_test: Number of data points used for evaluation metric.  # 用于评估指标的数据点数量
            directory: Directory to save results and figures.  # 保存结果和图像的目录
            problem: "Classification" or "Regression"(Not implemented yet.)  # 问题类型：“分类”或“回归”（回归尚未实现）
            model_family: The model family used for learning algorithm  # 学习算法使用的模型族
            metric: Evaluation metric  # 评估指标
            seed: Random seed. When running parallel monte-carlo samples,  # 随机种子。在运行并行蒙特卡洛采样时，
                we initialize each with a different seed to prevent getting   # 我们为每个样本初始化不同的种子以避免获得
                same permutations.  # 相同的排列。
            overwrite: Delete existing data and start computations from   # 是否删除现有数据并从头开始计算
                scratch  # 从零开始
            **kwargs: Arguments of the model  # 模型的参数
        """
            
        if seed is not None:  # 如果指定了随机种子
            np.random.seed(seed)  # 设置NumPy的随机种子
            tf.random.set_random_seed(seed)  # 设置TensorFlow的随机种子
        self.problem = problem  # 问题类型
        self.model_family = model_family  # 模型类型
        self.metric = metric  # 评估指标
        self.directory = directory  # 结果目录
        self.hidden_units = kwargs.get('hidden_layer_sizes', [])  # 获取隐藏层大小，默认为空列表
        if self.model_family is 'logistic':  # 如果是逻辑回归模型
            self.hidden_units = []  # 隐藏层为空
        if self.directory is not None:  # 如果指定了结果目录
            if overwrite and os.path.exists(directory):  # 如果覆盖且目录存在
                tf.gfile.DeleteRecursively(directory)  # 递归删除目录
            if not os.path.exists(directory):  # 如果目录不存在
                os.makedirs(directory)  # 创建目录
                os.makedirs(os.path.join(directory, 'weights'))  # 创建权重子目录
                os.makedirs(os.path.join(directory, 'plots'))  # 创建图像子目录
            self._initialize_instance(X, y, X_test, y_test, num_test,  # 初始化实例数据
                                      X_tot, y_tot, sources, sample_weight)  # 传入数据源和样本权重
        if len(set(self.y)) > 2:  # 如果类别数大于2
            assert self.metric != 'f1', 'Invalid metric for multiclass!'  # F1指标不适用于多分类
            assert self.metric != 'auc', 'Invalid metric for multiclass!'  # AUC指标不适用于多分类
        is_regression = (np.mean(self.y//1 == self.y) != 1)  # 判断是否为回归问题（通过检查标签是否为整数）
        is_regression = is_regression or isinstance(self.y[0], np.float32)  # 判断是否为回归问题（通过检查标签类型）
        self.is_regression = is_regression or isinstance(self.y[0], np.float64)  # 判断是否为回归问题（通过检查标签类型）
        if self.is_regression:  # 如果是回归问题
            warnings.warn("Regression problem is no implemented.")  # 发出警告，回归问题未实现
        self.model = return_model(self.model_family, **kwargs)  # 根据模型类型返回模型实例
        self.random_score = self.init_score(self.metric)  # 计算随机模型的得分作为基线
        #if seed is None and self.directory is not None:  # 如果未指定种子且有目录
            #np.random.seed(int(self.experiment_number))  # 使用实验编号设置种子
            #tf.random.set_random_seed(int(self.experiment_number))  # 使用实验编号设置种子
            
            
    def _initialize_instance(self, X, y, X_test, y_test, num_test,  # 私有方法：初始化实例数据
                             X_tot=None, y_tot=None,  # 全部数据
                             sources=None, sample_weight=None):  # 数据源和样本权重
        """Loads or creates sets of data."""  # 加载或创建数据集
        data_dir = os.path.join(self.directory, 'data.pkl')  # 数据文件路径
        if not os.path.exists(data_dir):  # 如果数据文件不存在
            self._save_dataset(data_dir, X, y, X_test, y_test, num_test,  # 保存数据集
                               X_tot, y_tot, sources, sample_weight)  # 传入数据
        self._load_dataset(data_dir)  # 加载数据集
        loo_dir = os.path.join(self.directory, 'loo.npy')  # 留一法（LOO）结果文件路径
        self.vals_loo = None  # 初始化LOO值
        if os.path.exists(loo_dir):  # 如果LOO结果文件存在
            self.vals_loo = np.load(loo_dir)  # 加载LOO值
        self.experiment_number = self._find_experiment_number(self.directory)  # 查找或创建实验编号
        self._create_results_placeholder(  # 创建结果占位符
            self.experiment_number, len(self.X), len(self.sources))  # 传入实验编号和数据大小
    
    def _save_dataset(self, data_dir, X, y, X_test, y_test, num_test,  # 私有方法：保存数据集
                      X_tot, y_tot, sources, sample_weight):  # 待保存的数据
        '''Save the different sets of data if already does not exist.'''  # 如果数据集不存在则保存
        data_dic = {  # 创建数据字典
            'X': X, 'y': y,  # 训练数据
            'X_test': X_test[-num_test:], 'y_test': y_test[-num_test:],  # 测试数据
            'X_heldout': X_test[:-num_test], 'y_heldout': y_test[:-num_test]  # 留出集数据
        }
        if sources is not None:  # 如果有数据源分组
            data_dic['sources'] = sources  # 保存数据源
        if X_tot is not None:  # 如果有全部数据
            data_dic['X_tot'] = X_tot  # 保存全部特征
            data_dic['y_tot'] = y_tot  # 保存全部标签
        if sample_weight is not None:  # 如果有样本权重
            data_dic['sample_weight'] = sample_weight  # 保存样本权重
            warnings.warn("Sample weight not implemented for G-Shapley")  # 警告：G-Shapley未实现样本权重
        pkl.dump(data_dic, open(data_dir, 'wb'))  # 使用pickle保存数据字典
                      
    def _load_dataset(self, data_dir):  # 私有方法：加载数据集
        '''Load the different sets of data if they already exist.'''  # 如果数据集存在则加载
        
        data_dic = pkl.load(open(data_dir, 'rb'))  # 使用pickle加载数据字典
        self.X = data_dic['X']   # 加载训练特征
        self.y = data_dic['y']  # 加载训练标签
        self.X_test = data_dic['X_test']  # 加载测试特征
        self.y_test = data_dic['y_test']  # 加载测试标签
        self.X_heldout = data_dic['X_heldout']  # 加载留出集特征
        self.y_heldout = data_dic['y_heldout']  # 加载留出集标签
        if 'sources' in data_dic.keys() and data_dic['sources'] is not None:  # 如果存在数据源
            self.sources = data_dic['sources']  # 加载数据源
        else:  # 否则
            self.sources = {i: np.array([i])  # 每个点作为一个独立的数据源
                            for i in range(len(self.X))}
        if 'X_tot' in data_dic.keys():  # 如果存在全部数据
            self.X_tot = data_dic['X_tot']  # 加载全部特征
            self.y_tot = data_dic['y_tot']  # 加载全部标签
        else:  # 否则
            self.X_tot = self.X  # 使用训练数据作为全部数据
            self.y_tot = self.y  # 使用训练标签作为全部标签
        if 'sample_weight' in data_dic.keys():  # 如果存在样本权重
            self.sample_weight = data_dic['sample_weight']  # 加载样本权重
        else:  # 否则
            self.sample_weight = None   # 样本权重为空
          
    def _find_experiment_number(self, directory):  # 私有方法：查找或创建实验编号
        
        '''Prevent conflict with parallel runs.'''  # 防止并行运行时发生冲突
        if 'arthur' in socket.gethostname():  # 根据主机名设置标志
            flag = socket.gethostname()[-1]  # 使用主机名最后一个字符
        else:  # 否则
            flag = '0'  # 默认标志为'0'
        previous_results = os.listdir(directory)  # 列出目录中已有结果
        nmbrs = [int(name.split('.')[-2].split('_')[0][1:])  # 从文件名中解析实验编号
                 for name in previous_results  # 遍历所有文件名
                 if '_result.pkl' in name and name[0] == flag]  # 筛选符合条件的文件
        experiment_number = str(np.max(nmbrs) + 1) if len(nmbrs) else '0'   # 计算新的实验编号
        experiment_number = flag + experiment_number.zfill(5)  # 格式化实验编号
        print(experiment_number)  # 打印实验编号
        return experiment_number  # 返回实验编号
    
    def _create_results_placeholder(self, experiment_number, n_points, n_sources):  # 私有方法：创建结果占位符
        '''Creates placeholder for results.'''  # 为结果创建占位符
        self.results = {}  # 初始化结果字典
        self.results['mem_dist'] = np.zeros((0, n_points))  # 分布式Shapley值的内存
        self.results['mem_tmc'] = np.zeros((0, n_points))  # TMC-Shapley值的内存
        self.results['mem_g'] = np.zeros((0, n_points))  # G-Shapley值的内存
        self.results['mem_dsvarm'] = np.zeros((0, n_points)) # D-SVARM值的内存
        self.results['idxs_dist'] = []  # 分布式Shapley值的索引
        self.results['idxs_tmc'] = []  # TMC-Shapley值的索引
        self.results['idxs_g'] = []  # G-Shapley值的索引
        self.results['idxs_dsvarm'] = []  # D-SVARM值的索引
        self.save_results()  # 保存初始结果
        
    def save_results(self, overwrite=False):  # 保存结果方法
        """Saves results computed so far."""  # 保存目前为止计算的结果
        if self.directory is None:  # 如果目录为空
            return  # 直接返回
        results_dir = os.path.join(  # 构建结果文件路径
            self.directory,   # 结果目录
            '{}_result.pkl'.format(self.experiment_number.zfill(6))  # 格式化文件名
        )
        pkl.dump(self.results, open(results_dir, 'wb'))  # 使用pickle保存结果
    
    def restart_model(self):  # 重启模型方法
        '''Restarts the model.'''  # 重启模型
        try:  # 尝试克隆模型
            self.model = clone(self.model)  # 克隆一个新模型
        except:  # 如果失败
            self.model.fit(np.zeros((0,) + self.X.shape[1:]), self.y)  # 用空数据拟合模型以重新初始化
    
    def init_score(self, metric):  # 计算初始得分
        """ Gives the value of an initial untrained model."""  # 给出一个未经训练的初始模型的值
        if metric == 'accuracy':  # 如果指标是准确率
            hist = np.bincount(self.y_test).astype(float)/len(self.y_test)  # 计算测试集标签的分布
            return np.max(hist)  # 返回最大类别频率作为随机猜测的准确率
        if metric == 'f1':  # 如果指标是F1分数
            rnd_f1s = []  # 初始化F1分数列表
            for _ in range(1000):  # 循环1000次
                rnd_y = np.random.permutation(self.y)  # 随机打乱标签
                rnd_f1s.append(f1_score(self.y_test, rnd_y))  # 计算F1分数
            return np.mean(rnd_f1s)  # 返回平均F1分数
        if metric == 'auc':  # 如果指标是AUC
            return 0.5  # 对于随机模型，AUC为0.5
        random_scores = []  # 初始化随机得分列表
        for _ in range(100):  # 循环100次
            rnd_y = np.random.permutation(self.y)  # 随机打乱标签
            if self.sample_weight is None:  # 如果没有样本权重
                self.model.fit(self.X, rnd_y)  # 用随机标签训练模型
            else:  # 如果有样本权重
                self.model.fit(self.X, rnd_y,   # 用随机标签和样本权重训练模型
                               sample_weight=self.sample_weight)
            random_scores.append(self.value(self.model, metric))  # 评估模型并记录得分
        return np.mean(random_scores)  # 返回平均得分
        
    def value(self, model, metric=None, X=None, y=None):  # 计算模型值
        """Computes the values of the given model.
        Args:
            model: The model to be evaluated.  # 要评估的模型
            metric: Valuation metric. If None the object's default  # 评估指标。如果为None，则使用对象的默认指标
                metric is used.
            X: Covariates, valuation is performed on a data   # 特征，在不同于测试集的数据上进行评估
                different from test set.
            y: Labels, if valuation is performed on a data   # 标签，在不同于测试集的数据上进行评估
                different from test set.
            """
        if metric is None:  # 如果未指定指标
            metric = self.metric  # 使用默认指标
        if X is None:  # 如果未指定特征
            X = self.X_test  # 使用测试集特征
        if y is None:  # 如果未指定标签
            y = self.y_test  # 使用测试集标签
        if inspect.isfunction(metric):  # 如果指标是一个函数
            return metric(model, X, y)  # 调用该函数计算值
        if metric == 'accuracy':  # 如果指标是准确率
            return model.score(X, y)  # 返回模型的准确率得分
        if metric == 'f1':  # 如果指标是F1分数
            assert len(set(y)) == 2, 'Data has to be binary for f1 metric.'  # 检查是否为二分类数据
            return f1_score(y, model.predict(X))  # 计算并返回F1分数
        if metric == 'auc':  # 如果指标是AUC
            assert len(set(y)) == 2, 'Data has to be binary for auc metric.'  # 检查是否为二分类数据
            return my_auc_score(model, X, y)  # 计算并返回AUC分数
        if metric == 'xe':  # 如果指标是交叉熵
            return my_xe_score(model, X, y)  # 计算并返回交叉熵得分
        raise ValueError('Invalid metric!')  # 如果指标无效则抛出错误
        
    def _tol_mean_score(self):  # 计算平均得分和容忍度
        """Computes the average performance and its error using bagging."""  # 使用bagging计算平均性能及其误差
        scores = []  # 初始化得分列表
        self.restart_model()  # 重启模型
        for _ in range(1):  # 循环1次（可修改为多次以获得更稳定的估计）
            if self.sample_weight is None:  # 如果没有样本权重
                self.model.fit(self.X, self.y)  # 在整个数据集上训练模型
            else:  # 如果有样本权重
                self.model.fit(self.X, self.y,  # 使用样本权重训练模型
                              sample_weight=self.sample_weight)
            for _ in range(100):  # 循环100次进行bagging
                bag_idxs = np.random.choice(len(self.y_test), len(self.y_test))  # 从测试集中有放回地抽样
                scores.append(self.value(  # 评估模型并记录得分
                    self.model, 
                    metric=self.metric,
                    X=self.X_test[bag_idxs], 
                    y=self.y_test[bag_idxs]
                ))
        self.tol = np.std(scores)  # 计算得分的标准差作为容忍度
        self.mean_score = np.mean(scores)  # 计算平均得分

    def run(self, save_every, err, tolerance=None, truncation=None, alpha=None,  # 主运行方法
            dist_run=False, tmc_run=False, loo_run=False, dsvarm_run=False, # 控制运行哪些Shapley值计算
            max_iters=None):  # 最大迭代次数
        """Calculates data sources(points) values.
        
        Args:
            save_every: Number of samples to to take at every iteration.  # 每次迭代采样的数量
            err: stopping criteria (maximum deviation of value in the past 100 iterations).  # 停止标准（过去100次迭代中值的最大偏差）
            tolerance: Truncation tolerance. If None, it's computed.  # 截断容忍度。如果为None，则自动计算
            truncation: truncation for D-Shapley (if none will use data size).  # D-Shapley的截断值（如果为None，则使用数据大小）
            alpha: Weighted sampling parameter. If None, biased sampling is not performed.  # 加权采样参数。如果为None，则不执行有偏采样
            dist_run:  If True, computes and saves D-Shapley values.  # 如果为True，则计算并保存D-Shapley值
            tmc_run:  If True, computes and saves TMC-Shapley values.  # 如果为True，则计算并保存TMC-Shapley值
            loo_run: If True, computes and saves leave-one-out scores.  # 如果为True，则计算并保存留一法得分
            dsvarm_run: If True, computes and saves D-SVARM values.
            max_iters: If not None, maximum number of iterations.  # 如果不为None，则为最大迭代次数
        """
        if loo_run:  # 如果运行留一法
            try:  # 尝试检查LOO值是否存在
                len(self.vals_loo)
            except:  # 如果不存在
                self.vals_loo = self._calculate_loo_vals(sources=self.sources)  # 计算LOO值
                np.save(os.path.join(self.directory, 'loo.npy'), self.vals_loo)  # 保存LOO值
            print('LOO values calculated!')  # 打印LOO值计算完成信息
        iters = 0  # 初始化迭代计数器
        while dist_run or tmc_run or dsvarm_run:  # 当需要运行时循环
            if dist_run:  # 如果运行D-Shapley
                if error(self.results['mem_dist']) < err:  # 如果收敛误差小于阈值
                    dist_run = False  # 停止运行D-Shapley
                    print('Distributional Shapley has converged!')  # 打印收敛信息
                else:  # 如果未收敛
                    self._dist_shap(  # 运行一轮D-Shapley计算
                        save_every, 
                        truncation=truncation, 
                        sources=self.sources,
                        alpha=alpha
                    )
                    self.vals_dist = np.mean(self.results['mem_dist'], 0)  # 更新D-Shapley的平均值
            if tmc_run:  # 如果运行TMC-Shapley
                if error(self.results['mem_tmc']) < err:  # 如果收敛误差小于阈值
                    tmc_run = False  # 停止运行TMC-Shapley
                    print('Data Shapley has converged!')  # 打印收敛信息
                else:  # 如果未收敛
                    self._tmc_shap(  # 运行一轮TMC-Shapley计算
                        save_every, 
                        tolerance=tolerance, 
                        sources=self.sources
                    )
                    self.vals_tmc = np.mean(self.results['mem_tmc'], 0)  # 更新TMC-Shapley的平均值
            
            if dsvarm_run: #如果运行D-SVARM
                if error(self.results['mem_dsvarm']) < err: # 如果收敛误差小于阈值
                    dsvarm_run = False # 停止运行D-SVARM
                    print('D-SVARM Shapley has converged!') # 打印收敛信息
                else: # 如果未收敛
                    self._d_svarm_shap(
                        save_every,
                        sources=self.sources
                    )
                    self.vals_dsvarm = np.mean(self.results['mem_dsvarm'], 0)
            
            if self.directory is not None:  # 如果指定了目录
                self.save_results()  # 保存结果
            iters += 1  # 迭代次数加1
            if max_iters is not None and iters >= max_iters:  # 如果达到最大迭代次数
                print('Reached to maximum number of iterations!')  # 打印达到最大迭代次数信息
                break  # 退出循环
        print('All methods have converged!')  # 打印所有方法已收敛信息
        
    def _dist_shap(self, iterations, truncation, sources=None, alpha=None):  # 私有方法：运行Distributional-Shapley
        """Runs Distribution-Shapley algorithm.
        
        Args:
            iterations: Number of iterations to run.  # 运行的迭代次数
            tolerance: Truncation tolerance ratio.  # 截断容忍度比率（在此方法中未使用）
            sources: If values are for sources of data points rather than  # 如果值是针对数据点源而不是
                   individual points. In the format of an assignment array  # 单个点。格式为分配数组
                   or dict.  # 或字典
        """
        if sources is None:  # 如果未提供数据源
            sources = {i: np.array([i]) for i in range(len(self.X))}  # 每个点都是一个独立源
        elif not isinstance(sources, dict):  # 如果sources不是字典
            sources = {i: np.where(sources == i)[0] for i in set(sources)}  # 将数组转换为字典格式
        model = self.model                # 获取当前模型
        marginals, idxs = [], []  # 初始化边际贡献和索引列表
        for iteration in range(iterations):  # 循环指定的迭代次数
            if 10*(iteration+1)/iterations % 1 == 0:  # 每10%的进度打印一次
                print('{} out of {} Dist_Shapley iterations.'.format(  # 打印进度信息
                    iteration + 1, iterations))
            marginals, idxs = self.dist_iteration(  # 执行一次Distributional-Shapley迭代
                truncation=truncation, 
                sources=sources,
                alpha=alpha
            )
            self.results['mem_dist'] = np.concatenate([  # 将新的边际贡献追加到结果中
                self.results['mem_dist'], 
                np.reshape(marginals, (1,-1))
            ])
            self.results['idxs_dist'].append(idxs)  # 追加当前迭代使用的索引

    def _d_svarm_shap(self, iterations, sources=None):  # D-SVARM算法的主控制方法
        """Runs D-SVARM algorithm."""  # 运行D-SVARM算法的文档字符串
        if sources is None:  # 如果未提供数据源分组
            sources = {i: np.array([i]) for i in range(len(self.X))}  # 每个数据点作为独立源
        elif not isinstance(sources, dict):  # 如果sources不是字典格式
            sources = {i: np.where(sources == i)[0] for i in set(sources)}  # 将数组格式转换为字典格式
        
        for iteration in range(iterations):  # 循环执行指定次数的D-SVARM迭代
            if 10 * (iteration + 1) / iterations % 1 == 0:  # 每完成10%进度时打印
                print('{} out of {} D-SVARM iterations.'.format(  # 打印当前进度信息
                    iteration + 1, iterations))  # 显示当前迭代数和总迭代数
            
            marginals, idxs = self.d_svarm_iteration(sources=sources)  # 执行一次D-SVARM迭代，获取边际贡献和背景集索引
            
            self.results['mem_dsvarm'] = np.concatenate([  # 将新的边际贡献追加到结果矩阵中
                self.results['mem_dsvarm'],  # 现有的D-SVARM结果矩阵
                np.reshape(marginals, (1, -1))  # 将当前迭代结果重塑为行向量并追加
            ])
            self.results['idxs_dsvarm'].append(idxs)  # 追加当前迭代使用的背景集索引

    def d_svarm_iteration(self, sources=None):  # D-SVARM算法的单次迭代方法
        """Runs one iteration of D-SVARM algorithm."""  # 执行一次D-SVARM算法迭代的文档字符串
        
        n_players = len(self.X)  # 获取目标数据集中的数据点总数（玩家数量）
        
        # 1. Sample background set S (from D-Shapley)  # 第一步：从D-Shapley采样背景集S
        # For simplicity, sample a background set of random size up to n_players  # 为简化，采样随机大小的背景集
        k = np.random.choice(np.arange(1, n_players + 1))  # 随机选择背景集大小k，范围[1, n_players]
        S_background_indices = np.random.choice(len(self.X_tot), k, replace=False)  # 从总数据集中无放回采样k个索引
        X_S = self.X_tot[S_background_indices]  # 根据索引获取背景集的特征数据
        y_S = self.y_tot[S_background_indices]  # 根据索引获取背景集的标签数据
        
        # 2. Define derived game g_S(A)  # 第二步：定义派生博弈g_S(A)
        self.restart_model()  # 重新初始化模型以确保干净的起始状态
        try:  # 尝试在背景集上训练模型
            self.model.fit(X_S, y_S)  # 在背景集S上训练机器学习模型
            U_S_base = self.value(self.model, metric=self.metric)  # 计算背景集S的基准效用值
        except Exception:  # 如果训练失败（例如数据不足或类别不全）
            U_S_base = self.random_score  # 使用随机得分作为基准效用值

        def g_S(A_indices):  # 定义派生博弈函数g_S(A)，计算联盟A相对于背景集S的边际贡献
            # A_indices are indices from self.X  # A_indices是目标数据集self.X中的索引
            X_A = self.X[A_indices]  # 根据索引从目标数据集中提取联盟A的特征
            y_A = self.y[A_indices]  # 根据索引从目标数据集中提取联盟A的标签
            
            X_train = np.concatenate([X_S, X_A])  # 将背景集S和联盟A的特征合并作为训练数据
            y_train = np.concatenate([y_S, y_A])  # 将背景集S和联盟A的标签合并作为训练标签
            
            self.restart_model()  # 重新初始化模型以避免之前训练的影响
            try:  # 尝试在合并数据上训练模型
                self.model.fit(X_train, y_train)  # 在S∪A上训练模型
                u_val = self.value(self.model, metric=self.metric)  # 计算S∪A的效用值U(S∪A)
            except Exception:  # 如果训练失败
                u_val = self.random_score  # 使用随机得分作为效用值
            
            return u_val - U_S_base  # 返回边际贡献：U(S∪A) - U(S)

        # 3. S-SVARM efficient update  # 第三步：S-SVARM高效更新机制
        phi_i_l_plus = np.zeros((n_players, n_players))  # 初始化正向Shapley子值矩阵φ_i_l_plus
        phi_i_l_minus = np.zeros((n_players, n_players))  # 初始化负向Shapley子值矩阵φ_i_l_minus
        c_i_l_plus = np.zeros((n_players, n_players))  # 初始化正向计数矩阵c_i_l_plus
        c_i_l_minus = np.zeros((n_players, n_players))  # 初始化负向计数矩阵c_i_l_minus
        
        # Generate probability distribution over coalition sizes according to S-SVARM paper  # 根据S-SVARM论文生成联盟大小的概率分布
        distribution = self._generate_distribution("paper", n_players)  # 生成论文中的理论概率分布P(s)
        probs = [distribution[s] for s in range(n_players + 1)]  # 构建概率数组，索引s对应概率P(s)
        
        # Inner loop for S-SVARM sampling  # S-SVARM采样的内部循环
        # Let's do n_players iterations for a reasonable estimate  # 执行n_players次迭代以获得合理估计
        for _ in range(n_players):  # 循环n_players次进行S-SVARM采样
            # Sample coalition size s according to theoretical distribution P(s)  # 根据理论分布P(s)采样联盟大小s
            s = np.random.choice(range(0, n_players + 1), p=probs)  # 按照论文概率分布采样联盟大小s
            if s == 0 or s == n_players:  # 如果联盟大小为0（空集）或n_players（全集）
                continue # Skip empty or full set for this simplified sampling  # 跳过空集和全集以简化采样
            
            A_indices = np.random.choice(np.arange(n_players), s, replace=False)  # 从n_players个玩家中无放回采样s个，形成联盟A
            
            v_A = g_S(A_indices)  # 计算联盟A在派生博弈g_S中的价值
            
            # Update procedure from StratifiedSVARM  # 来自StratifiedSVARM的更新程序
            for i in A_indices:  # 对联盟A中的每个玩家i进行正向更新
                phi_i_l_plus[i][s - 1] = (phi_i_l_plus[i][s - 1] * c_i_l_plus[i][s - 1] + v_A) / (c_i_l_plus[i][s - 1] + 1)  # 更新玩家i在层级s-1的正向Shapley子值
                c_i_l_plus[i][s - 1] += 1  # 增加玩家i在层级s-1的正向计数
            
            not_A_indices = np.setdiff1d(np.arange(n_players), A_indices)  # 找到不在联盟A中的所有玩家
            for i in not_A_indices:  # 对不在联盟A中的每个玩家i进行负向更新
                phi_i_l_minus[i][s] = (phi_i_l_minus[i][s] * c_i_l_minus[i][s] + v_A) / (c_i_l_minus[i][s] + 1)  # 更新玩家i在层级s的负向Shapley子值
                c_i_l_minus[i][s] += 1  # 增加玩家i在层级s的负向计数

        # 4. Calculate final Shapley values for this iteration  # 第四步：计算本次迭代的最终Shapley值
        shapley_values = np.zeros(n_players)  # 初始化Shapley值数组
        for i in range(n_players):  # 遍历每个玩家i
            # Using the formula from the user request  # 使用用户请求中的公式
            # We only sum over strata that have been visited.  # 只对访问过的层级进行求和
            plus_sum = np.sum(phi_i_l_plus[i][np.where(c_i_l_plus[i] > 0)])  # 计算玩家i在所有访问过层级的正向Shapley子值之和
            minus_sum = np.sum(phi_i_l_minus[i][np.where(c_i_l_minus[i] > 0)])  # 计算玩家i在所有访问过层级的负向Shapley子值之和
            shapley_values[i] = (plus_sum - minus_sum) / n_players  # 应用公式：φ_i = (1/n) * Σ(φ_i_l_plus - φ_i_l_minus)

        return shapley_values, list(S_background_indices)  # 返回本次迭代计算得到的所有玩家的Shapley值估计和背景集索引
        
    def dist_iteration(self, truncation, sources=None, alpha=None):  # Distributional-Shapley的单次迭代
        
        num_classes = len(set(self.y_test))  # 获取测试集中的类别数
        if sources is None:  # 如果未提供数据源
            sources = {i: np.array([i]) for i in range(len(self.X))}  # 每个点都是一个独立源
        elif not isinstance(sources, dict):  # 如果sources不是字典
            sources = {i: np.where(sources == i)[0] for i in set(sources)}  # 将数组转换为字典格式
        marginal_contribs = np.zeros(len(self.X))  # 初始化边际贡献数组
        while True:  # 循环直到满足采样条件
            k = np.random.choice(np.arange(1, truncation + 1))  # 从[1, truncation]中随机选择一个子集大小k
            if alpha is None or np.random.random() < (1. / (k ** alpha)):  # 根据alpha参数进行有偏采样
                break  # 满足条件则跳出循环
        if k == 1:  # 如果子集大小为1
            return marginal_contribs, []  # 边际贡献为0，返回空索引
        S = np.random.choice(len(self.X_tot), k - 1, replace=False)  # 从总数据集中无放回地随机采样k-1个点
        X_init = self.X_tot[S]  # 获取采样点的特征
        y_init = self.y_tot[S]  # 获取采样点的标签
        self.restart_model()  # 重启模型
        if len(set(y_init)) != num_classes and not self.is_regression:  # 如果采样点的类别不全且不是回归问题
            init_score = self.random_score  # 初始得分设为随机得分
        else:  # 否则
            try:  # 尝试训练模型
                self.model.fit(X_init, y_init)  # 在采样点上训练模型
                init_score = self.value(self.model, metric=self.metric)  # 计算初始得分
            except:  # 如果训练失败
                init_score = self.random_score  # 初始得分设为随机得分
        time_init = time.time()  # 记录初始时间
        for idx in range(len(sources.keys())):  # 遍历所有数据源
            X_batch = np.concatenate([X_init, self.X[sources[idx]]])  # 将当前源的数据点加入采样集
            y_batch = np.concatenate([y_init, self.y[sources[idx]]])  # 将当前源的标签加入采样集
            if len(set(y_batch)) != num_classes and not self.is_regression:  # 如果类别不全且不是回归问题
                continue  # 跳过当前源
            self.restart_model()  # 重启模型
            try:  # 尝试训练模型
                self.model.fit(X_batch, y_batch)  # 在新数据上训练模型
                score = self.value(self.model, metric=self.metric)  # 计算新得分
                marginal_contribs[sources[idx]] = score - init_score  # 计算边际贡献
                marginal_contribs[sources[idx]] /= len(sources[idx])  # 对源内所有点取平均
            except:  # 如果训练失败
                continue  # 跳过当前源
        return marginal_contribs, list(S)  # 返回边际贡献和采样索引
        
    def _tmc_shap(self, iterations, tolerance=None, sources=None):  # 私有方法：运行TMC-Shapley
        """Runs TMC-Shapley algorithm.
        
        Args:
            iterations: Number of iterations to run.  # 运行的迭代次数
            tolerance: Truncation tolerance ratio.  # 截断容忍度比率
            sources: If values are for sources of data points rather than  # 如果值是针对数据点源而不是
                   individual points. In the format of an assignment array  # 单个点。格式为分配数组
                   or dict.  # 或字典
        """
        if sources is None:  # 如果未提供数据源
            sources = {i: np.array([i]) for i in range(len(self.X))}  # 每个点都是一个独立源
        elif not isinstance(sources, dict):  # 如果sources不是字典
            sources = {i: np.where(sources == i)[0] for i in set(sources)}  # 将数组转换为字典格式
        model = self.model  # 获取当前模型
        try:  # 尝试访问mean_score
            self.mean_score
        except:  # 如果不存在
            self._tol_mean_score()  # 计算平均得分和容忍度
        if tolerance is None:  # 如果未提供容忍度
            tolerance = self.tol         # 使用计算出的容忍度
        marginals, idxs = [], []  # 初始化边际贡献和索引列表
        for iteration in range(iterations):  # 循环指定的迭代次数
            if 10*(iteration+1)/iterations % 1 == 0:  # 每10%的进度打印一次
                print('{} out of {} TMC_Shapley iterations.'.format(  # 打印进度信息
                    iteration + 1, iterations))
            marginals, idxs = self.tmc_iteration(  # 执行一次TMC-Shapley迭代
                tolerance=tolerance, 
                sources=sources
            )
            self.results['mem_tmc'] = np.concatenate([  # 将新的边际贡献追加到结果中
                self.results['mem_tmc'], 
                np.reshape(marginals, (1,-1))
            ])
            self.results['idxs_tmc'].append(idxs)  # 追加当前迭代使用的索引
        
    def tmc_iteration(self, tolerance, sources=None):  # TMC-Shapley的单次迭代
        """Runs one iteration of TMC-Shapley algorithm."""  # 运行一次TMC-Shapley算法迭代
        num_classes = len(set(self.y_test))  # 获取测试集中的类别数
        if sources is None:  # 如果未提供数据源
            sources = {i: np.array([i]) for i in range(len(self.X))}  # 每个点都是一个独立源
        elif not isinstance(sources, dict):  # 如果sources不是字典
            sources = {i: np.where(sources == i)[0] for i in set(sources)}  # 将数组转换为字典格式
        idxs = np.random.permutation(len(sources))  # 随机打乱数据源的顺序
        marginal_contribs = np.zeros(len(self.X))  # 初始化边际贡献数组
        X_batch = np.zeros((0,) + tuple(self.X.shape[1:]))  # 初始化空的特征批次
        y_batch = np.zeros(0, int)  # 初始化空的标签批次
        truncation_counter = 0  # 初始化截断计数器
        new_score = self.random_score  # 初始得分设为随机得分
        for n, idx in enumerate(idxs):  # 遍历打乱顺序后的数据源
            old_score = new_score  # 保存上一轮的得分
            X_batch = np.concatenate([X_batch, self.X[sources[idx]]])  # 将当前源的特征加入批次
            y_batch = np.concatenate([y_batch, self.y[sources[idx]]])  # 将当前源的标签加入批次
            if self.sample_weight is None:  # 如果没有样本权重
                sample_weight_batch = None  # 样本权重批次为空
            else:  # 如果有样本权重
                sample_weight_batch = np.concatenate([  # 将当前源的样本权重加入批次
                    sample_weight_batch, 
                    self.sample_weight[sources[idx]]
                ])
            if len(set(y_batch)) != num_classes and not self.is_regression:  # 如果批次中的类别不全且不是回归问题
                continue  # 跳过当前迭代
            self.restart_model()  # 重启模型
            #try:
            if True:  # 总是尝试训练
                if sample_weight_batch is None:  # 如果没有样本权重
                    self.model.fit(X_batch, y_batch)  # 训练模型
                else:  # 如果有样本权重
                    self.model.fit(  # 使用样本权重训练模型
                        X_batch, 
                        y_batch,
                        sample_weight = sample_weight_batch
                    )
                new_score = self.value(self.model, metric=self.metric)    # 计算新得分
            #except:
                #continue
            marginal_contribs[sources[idx]] = new_score - old_score  # 计算边际贡献
            marginal_contribs[sources[idx]] /= len(sources[idx])  # 对源内所有点取平均
            distance_to_full_score = np.abs(new_score - self.mean_score)  # 计算当前得分与平均完全得分的差距
            if distance_to_full_score <= tolerance * self.mean_score:  # 如果差距小于容忍度
                truncation_counter += 1  # 截断计数器加1
                if truncation_counter > 5:  # 如果连续5次满足条件
                    print('Truncated at {}'.format(n))  # 打印截断信息
                    break  # 提前终止迭代
            else:  # 如果差距大于容忍度
                truncation_counter = 0  # 重置截断计数器
        return marginal_contribs, idxs  # 返回边际贡献和排列索引
    
    def _calculate_loo_vals(self, sources=None, metric=None):  # 私有方法：计算留一法（LOO）值
        """Calculated leave-one-out values for the given metric.
        
        Args:
            metric: If None, it will use the objects default metric.  # 评估指标，如果为None，则使用对象的默认指标
            sources: If values are for sources of data points rather than  # 如果值是针对数据点源而不是
                   individual points. In the format of an assignment array  # 单个点。格式为分配数组
                   or dict.  # 或字典
        
        Returns:
            Leave-one-out scores  # 留一法得分
        """
        if sources is None:  # 如果未提供数据源
            sources = {i: np.array([i]) for i in range(len(self.X))}  # 每个点都是一个独立源
        elif not isinstance(sources, dict):  # 如果sources不是字典
            sources = {i: np.where(sources==i)[0] for i in set(sources)}  # 将数组转换为字典格式
        print('Starting LOO score calculations!')  # 打印开始计算LOO得分的信息
        if metric is None:  # 如果未提供评估指标
            metric = self.metric   # 使用对象的默认指标
        self.restart_model()  # 重启模型
        if self.sample_weight is None:  # 如果没有样本权重
            self.model.fit(self.X, self.y)  # 在整个数据集上训练模型
        else:  # 如果有样本权重
            self.model.fit(self.X, self.y,  # 使用样本权重训练模型
                          sample_weight=self.sample_weight)
        baseline_value = self.value(self.model, metric=metric)  # 计算基线值（在所有数据上训练的模型性能）
        vals_loo = np.zeros(len(self.X))  # 初始化LOO值数组
        for i in sources.keys():  # 遍历所有数据源
            print(i)  # 打印当前处理的数据源索引
            X_batch = np.delete(self.X, sources[i], axis=0)  # 删除当前数据源的特征
            y_batch = np.delete(self.y, sources[i], axis=0)  # 删除当前数据源的标签
            if self.sample_weight is not None:  # 如果有样本权重
                sw_batch = np.delete(self.sample_weight, sources[i], axis=0)  # 删除当前数据源的样本权重
            if self.sample_weight is None:  # 如果没有样本权重
                self.model.fit(X_batch, y_batch)  # 在剩余数据上训练模型
            else:  # 如果有样本权重
                self.model.fit(X_batch, y_batch, sample_weight=sw_batch)  # 使用样本权重在剩余数据上训练模型
                
            removed_value = self.value(self.model, metric=metric)  # 计算移除当前数据源后的模型性能
            vals_loo[sources[i]] = (baseline_value - removed_value)  # 计算LOO值（性能下降量）
            vals_loo[sources[i]] /= len(sources[i])  # 对源内所有点取平均
        return vals_loo  # 返回所有数据点的LOO值
    
    def _concat(self, results, key, batch_result):  # 私有方法：连接结果
        
        if 'mem' in key or 'idxs' in key:  # 如果键包含'mem'或'idxs'
            if key in results.keys():  # 如果结果字典中已存在该键
                if isinstance(results[key], list):  # 如果是列表
                    results[key].extend(batch_result)  # 扩展列表
                    return results[key]  # 返回更新后的列表
                else:  # 如果是Numpy数组
                    return np.concatenate([results[key], batch_result])  # 连接数组
            else:  # 如果结果字典中不存在该键
                return batch_result.copy()  # 返回批次结果的副本
        else:  # 对于其他键（如计数值）
            if key in results.keys():  # 如果结果字典中已存在该键
                return results[key] + batch_result  # 直接相加
            else:  # 如果不存在
                return batch_result  # 返回批次结果
    
    def _load_batch(self, batch_dir):  # 私有方法：加载批次结果
        
        try:  # 尝试加载
            batch = pkl.load(open(batch_dir, 'rb'))  # 使用pickle加载批次文件
            batch_sizes = [len(batch[key]) for key in batch if 'mem' in key]  # 获取批次中内存相关数组的长度
            return batch, np.max(batch_sizes)  # 返回批次数据和最大长度
        except:  # 如果加载失败
            return None, None  # 返回None
    
    def _filter_batch(self, batch, idxs=None):  # 私有方法：过滤批次结果
        
        if idxs is None:  # 如果未提供索引
            return batch  # 直接返回原始批次
        for key in batch.keys():  # 遍历批次中的所有键
            if 'mem' in key or 'idxs' in key:  # 如果是内存或索引相关的键
                if not isinstance(batch[key], list) and len(batch[key]):  # 如果不是列表且不为空
                    batch[key] = batch[key][:, idxs]  # 根据提供的索引进行过滤
        return batch  # 返回过滤后的批次
    
    def dist_stats(self, truncation, idxs=None):  # 计算Distributional-Shapley的统计数据

        stats = {}  # 初始化统计字典
        if idxs is None:  # 如果未提供索引
            idxs = np.arange(len(self.X))  # 使用所有数据点的索引
        stats['vals'] = np.zeros((len(idxs), truncation))  # 初始化值的数组
        stats['vars'] = np.zeros((len(idxs), truncation))  # 初始化方差的数组
        stats['counts'] = np.zeros((len(idxs), truncation))  # 初始化计数的数组
        batch_dirs = [os.path.join(self.directory, item)   # 获取所有批次结果文件的路径
                   for item in os.listdir(self.directory)
                   if '_result.pkl' in item]
        for i, batch_dir in enumerate(np.sort(batch_dirs)):  # 遍历排序后的批次文件
            batch, batch_size = self._load_batch(batch_dir)  # 加载批次数据
            if batch is None or batch_size == 0:  # 如果批次为空
                continue  # 跳过
            if 'idxs_dist' not in batch.keys() or not len(batch['idxs_dist']):  # 如果没有dist_shapley的索引
                continue  # 跳过
            present = (batch['mem_dist'] != -1).astype(float)  # 标记有效（非-1）的边际贡献
            counts = np.array([len(i) for i in batch['idxs_dist']])  # 计算每个迭代的子集大小
            for i, count in enumerate(counts):  # 遍历每个迭代
                if count >= truncation:  # 如果子集大小超出范围
                    continue  # 跳过
                present = (batch['mem_dist'][i, idxs] != -1).astype(float)  # 检查当前迭代中指定索引的值是否有效
                stats['counts'][:, count] += present  # 累加有效计数值
                stats['vals'][:, count] += present * batch['mem_dist'][i, idxs]  # 累加有效值
                stats['vars'][:, count] += present * (batch['mem_dist'][i, idxs] ** 2)  # 累加有效值的平方
        for i in range(len(stats['counts'])):  # 遍历每个数据点
            nzs = np.where(stats['counts'][i] > 0)[0]  # 找到计数不为零的位置
            stats['vals'][i, nzs] /= stats['counts'][i, nzs]  # 计算平均值
            stats['vars'][i, nzs] /= stats['counts'][i, nzs]  # 计算平方的平均值
            stats['vars'][i, nzs] -= stats['vals'][i, nzs] ** 2  # 计算方差
        return stats  # 返回统计结果

    def load_results(self, max_samples=None, idxs=None, verbose=True):  # 加载所有结果
        """Helper method for 'merge_results' method."""  # 'merge_results'方法的辅助方法
        results = {}  # 初始化结果字典
        results_size = 0  # 初始化结果大小
        batch_dirs = [os.path.join(self.directory, item)   # 获取所有批次结果文件的路径
                       for item in os.listdir(self.directory)
                       if '_result.pkl' in item]
        for i, batch_dir in enumerate(np.sort(batch_dirs)):  # 遍历排序后的批次文件
            batch, batch_size = self._load_batch(batch_dir)  # 加载批次数据
            if verbose:  # 如果需要打印详细信息
                print(batch_dir, batch_size)  # 打印文件名和大小
            if batch is None or batch_size == 0:  # 如果批次为空
                os.remove(batch_dir)  # 删除空文件
                continue  # 跳过
            if max_samples is not None:  # 如果设置了最大样本数
                for key in batch:  # 遍历批次中的键
                    if 'mem' in key or 'idxs' in key:  # 如果是内存或索引相关的键
                        batch[key] = batch[key][:max_samples - results_size]  # 截取到最大样本数
                results_size = min(results_size + batch_size, max_samples)  # 更新已加载的样本数
            batch = self._filter_batch(batch, idxs)  # 根据索引过滤批次
            for alg in set([key.split('_')[-1] for key in batch]):  # 从键名中解析算法名称
                present = (batch['mem_' + alg] != -1).astype(float)  # 检查有效值
                del present  # 删除临时变量
                if not len(batch['mem_' + alg]):  # 如果该算法的结果为空
                    continue  # 跳过
                results['mem_' + alg] = self._concat(  # 连接内存结果
                    results, 'mem_' + alg, batch['mem_' + alg])
                # 检查索引键是否存在（兼容旧的结果文件）
                if 'idxs_' + alg in batch:  # 如果批次中存在索引键
                    results['idxs_' + alg] = self._concat(  # 连接索引结果
                        results, 'idxs_' + alg, batch['idxs_' + alg])
                else:  # 如果不存在索引键（旧文件兼容性）
                    if 'idxs_' + alg not in results:  # 如果结果中还没有该键
                        results['idxs_' + alg] = []  # 初始化为空列表
            if max_samples is not None and results_size >= max_samples:  # 如果达到最大样本数
                break  # 退出循环
        self.results = results  # 将加载的结果存入对象属性
    
    def merge_results(self, chunk_size=100):  # 合并结果文件
        
        batch_dirs = np.sort([os.path.join(self.directory, item)   # 获取所有批次结果文件的路径
                              for item in os.listdir(self.directory)
                              if '_result.pkl' in item])
        batch_sizes = [os.path.getsize(batch_dir) for batch_dir in batch_dirs]  # 获取每个批次文件的大小
        merged_size = 0  # 初始化已合并文件的大小
        merged_dirs = [[]]  # 初始化要合并的文件列表
        for batch_dir, batch_size in zip(batch_dirs, batch_sizes):  # 遍历文件和大小
            merged_dirs[-1].append(batch_dir)  # 将当前文件添加到最后一个合并组
            merged_size += batch_size  # 累加文件大小
            if merged_size > chunk_size * 1e6:  # 如果超过块大小（MB）
                merged_dirs.append([])  # 创建一个新的合并组
                merged_size = 0  # 重置大小计数
        for i, batch_dirs in enumerate(merged_dirs):  # 遍历每个合并组
            result_dic = '{}_result.pkl'.format(str(i).zfill(6))  # 创建合并后的文件名
            merged_dir = os.path.join(self.directory, result_dic)  # 创建合并后的文件路径
            if len(batch_dirs) == 1 and batch_dirs[0] == merged_dir:  # 如果组内只有一个文件且已是合并文件
                print(merged_dir, 'exists')  # 打印已存在信息
                continue  # 跳过
            results = {}  # 初始化结果字典
            for batch_dir in batch_dirs:  # 遍历组内的所有文件
                batch, batch_size = self._load_batch(batch_dir)  # 加载批次
                if batch is None or batch_size == 0:  # 如果批次为空
                    continue  # 跳过
                for alg in set([key.split('_')[-1] for key in batch]):  # 从键名中解析算法名称
                    results['mem_' + alg] = self._concat(  # 连接内存结果
                        results, 'mem_' + alg, batch['mem_' + alg])
                    # 检查索引键是否存在（兼容旧的结果文件）
                    if 'idxs_' + alg in batch:  # 如果批次中存在索引键
                        results['idxs_' + alg] = self._concat(  # 连接索引结果
                            results, 'idxs_' + alg, batch['idxs_' + alg])
                    else:  # 如果不存在索引键（旧文件兼容性）
                        if 'idxs_' + alg not in results:  # 如果结果中还没有该键
                            results['idxs_' + alg] = []  # 初始化为空列表
                
            pkl.dump(results, open(merged_dir, 'wb'), protocol=4)  # 保存合并后的结果
            for batch_dir in batch_dirs:  # 遍历组内的所有文件
                if batch_dir != merged_dir:  # 如果不是合并后的文件
                    os.remove(batch_dir)  # 删除原始文件
            print(merged_dir)  # 打印合并后的文件名
            
    def portion_performance(  # 计算移除一部分数据后的性能
        self, idxs, plot_points, sources=None, X=None, y=None, sample_weight=None, verbose=False):
        """Given a set of indexes, starts removing points from 
        the first elemnt and evaluates the new model after
        removing each point."""  # 给定一组索引，从第一个元素开始移除点，并在每次移除后评估新模型
        if X is None:  # 如果未提供特征
            X = self.X  # 使用默认训练特征
            y = self.y  # 使用默认训练标签
            sample_weight = self.sample_weight  # 使用默认样本权重
        if sources is None:  # 如果未提供数据源
            sources = {i: np.array([i]) for i in range(len(X))}  # 每个点都是一个独立源
        elif not isinstance(sources, dict):  # 如果sources不是字典
            sources = {i: np.where(sources==i)[0] for i in set(sources)}  # 将数组转换为字典格式
        scores = []  # 初始化得分列表
        init_score = self.random_score  # 初始得分设为随机得分
        for i in range(len(plot_points), 0, -1):  # 倒序遍历绘图点
            if verbose:  # 如果需要打印详细信息
                print('{} out of {}'.format(len(plot_points)-i+1, len(plot_points)))  # 打印进度
            keep_idxs = np.concatenate([sources[idx] for idx   # 获取要保留的数据点的索引
                                        in idxs[plot_points[i-1]:]], -1)
            X_batch, y_batch = X[keep_idxs], y[keep_idxs]  # 根据索引创建新的批次
            if sample_weight is not None:  # 如果有样本权重
                sample_weight_batch = self.sample_weight[keep_idxs]  # 获取对应样本权重
            try:  # 尝试训练和评估
                self.restart_model()  # 重启模型
                if self.sample_weight is None:  # 如果没有样本权重
                    self.model.fit(X_batch, y_batch)  # 训练模型
                else:  # 如果有样本权重
                    self.model.fit(X_batch, y_batch,  # 使用样本权重训练模型
                                  sample_weight=sample_weight_batch)
                scores.append(self.value(  # 评估模型并记录得分
                    self.model,
                    metric=self.metric,
                    X=self.X_heldout,
                    y=self.y_heldout
                ))
            except:  # 如果失败
                scores.append(init_score)  # 记录随机得分
        return np.array(scores)[::-1]  # 返回得分数组（颠倒顺序以匹配移除顺序）
