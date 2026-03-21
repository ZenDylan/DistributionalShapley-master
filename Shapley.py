import os  # 导入操作系统相关功能模块
import numpy as np  # 导入NumPy数值计算库
import tensorflow as tf  # 导入TensorFlow深度学习框架
from sklearn.model_selection import train_test_split  # 从scikit-learn导入数据集划分工具
from sklearn.metrics import roc_auc_score, f1_score  # 从scikit-learn导入评估指标

class ShapNN(object):  # 基于TensorFlow的神经网络类，用于Shapley值计算中的模型训练
    
    def __init__(self, mode, hidden_units=[100], learning_rate=0.001,  # 初始化函数：设置模型参数
                 dropout = 0., activation=None, initializer=None,  # dropout率、激活函数、权重初始化器
                 weight_decay=0.0001, optimizer='adam', batch_size=128,  # 权重衰减、优化器、批次大小
                 warm_start=False, max_epochs=100, validation_fraction=0.1,  # 热启动、最大轮数、验证集比例
                 early_stopping=0, address=None, test_batch_size=1000,  # 早停、保存地址、测试批次大小
                 random_seed=666):  # 随机种子
        
        self.mode = mode  # 模型模式：'classification'分类或'regression'回归
        self.batch_size = batch_size  # 训练批次大小
        self.test_batch_size = test_batch_size  # 测试批次大小
        self.hidden_units = hidden_units  # 隐藏层单元数列表
        self.initializer = initializer  # 权重初始化器
        self.activation = activation  # 激活函数
        self.dropout = dropout  # dropout概率，用于防止过拟合
        self.weight_decay = weight_decay  # 权重衰减系数，用于L2正则化
        self.optimizer = optimizer  # 优化器类型
        self.learning_rate = learning_rate  # 学习率
        self.warm_start = warm_start  # 是否热启动（从已有权重开始）
        self.max_epochs = max_epochs  # 最大训练轮数
        self.early_stopping = early_stopping  # 早停轮数，0表示不使用早停
        self.validation_fraction = validation_fraction  # 验证集比例
        self.address = address  # 模型保存地址
        self._extra_train_ops = []  # 额外的训练操作列表
        self.random_seed = random_seed  # 随机种子，确保结果可复现
        self.is_built = False  # 标记模型是否已构建

    def prediction_cost(self, X_test, y_test, batch_size=None):  # 计算测试数据的预测损失
        
        if batch_size is None:  # 如果未指定批次大小
            batch_size = self.test_batch_size  # 使用默认的测试批次大小
        assert len(set(y_test)) == self.num_classes, 'Number of classes does not match!'  # 检查类别数匹配
        with self.graph.as_default():  # 在模型的计算图上下文中执行操作
            losses = []  # 初始化损失列表
            idxs = np.arange(len(X_test))  # 生成测试数据的索引数组
            batches = [idxs[k * batch_size: (k+1) * batch_size]  # 将数据分成多个批次
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]  # 计算批次数量
            for batch in batches:  # 对每个批次计算预测损失
                losses.append(self.sess.run(self.prediction_loss, {self.input_ph:X_test[batch],  # 运行TF会话计算损失
                                                                   self.labels:y_test[batch]}))  # 传入数据和标签
            return np.mean(losses)  # 返回所有批次损失的平均值     
        
    def score(self, X_test, y_test, batch_size=None):  # 计算模型在测试数据上的评分
        
        if batch_size is None:  # 如果未指定批次大小
            batch_size = self.test_batch_size  # 使用默认的测试批次大小
        assert len(set(y_test)) == self.num_classes, 'Number of classes does not match!'  # 检查类别数匹配
        with self.graph.as_default():  # 在模型的计算图上下文中执行操作
            scores = []  # 初始化评分列表
            idxs = np.arange(len(X_test))  # 生成测试数据的索引数组
            batches = [idxs[k * batch_size: (k+1) * batch_size]  # 将数据分成多个批次
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]  # 计算批次数量
            for batch in batches:  # 对每个批次计算评分
                scores.append(self.sess.run(self.prediction_score, {self.input_ph:X_test[batch],  # 运行TF会话计算评分
                                                                   self.labels:y_test[batch]}))  # 传入数据和标签
            return np.mean(scores)  # 返回所有批次评分的平均值
        
    def predict_proba(self, X_test, batch_size=None):  # 预测测试数据的类别概率
        
        if batch_size is None:  # 如果未指定批次大小
            batch_size = self.test_batch_size  # 使用默认的测试批次大小
        with self.graph.as_default():  # 在模型的计算图上下文中执行操作
            probs = []  # 初始化概率列表
            idxs = np.arange(len(X_test))  # 生成测试数据的索引数组
            batches = [idxs[k * batch_size: (k+1) * batch_size]  # 将数据分成多个批次
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]  # 计算批次数量
            for batch in batches:  # 对每个批次预测概率
                probs.append(self.sess.run(self.probs, {self.input_ph:X_test[batch]}))  # 运行TF会话计算概率
            return np.concatenate(probs, axis=0)  # 拼接所有批次的概率结果并返回
        
    def predict_log_proba(self, X_test, batch_size=None):  # 预测测试数据的对数概率
        
        if batch_size is None:  # 如果未指定批次大小
            batch_size = self.test_batch_size  # 使用默认的测试批次大小
        with self.graph.as_default():  # 在模型的计算图上下文中执行操作
            probs = []  # 初始化概率列表
            idxs = np.arange(len(X_test))  # 生成测试数据的索引数组
            batches = [idxs[k * batch_size: (k+1) * batch_size]  # 将数据分成多个批次
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]  # 计算批次数量
            for batch in batches:  # 对每个批次预测概率
                probs.append(self.sess.run(self.probs, {self.input_ph:X_test[batch]}))  # 运行TF会话计算概率
            return np.log(np.clip(np.concatenate(probs), 1e-12, None))  # 对概率取对数，使用clip避免log(0)   
        
    def cost(self, X_test, y_test, batch_size=None):  # 计算模型在数据上的损失
        
        if batch_size is None:  # 如果未指定批次大小
            batch_size = self.batch_size  # 使用默认批次大小
        with self.graph.as_default():  # 在计算图上下文中执行
            losss = []  # 初始化损失列表
            idxs = np.arange(len(X_test))  # 生成数据索引            
            batches = [idxs[k * batch_size: (k+1) * batch_size]  # 将数据分成批次 
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]  # 计算批次数量
            for batch in batches:  # 遍历每个批次
                losss.append(self.sess.run(self.prediction_loss, {self.input_ph:X_test[batch],  # 计算批次损失
                                                                   self.labels:y_test[batch]}))  # 传入数据和标签
            return np.mean(losss)  # 返回平均损失
    
    def predict(self, X_test, batch_size=None):  # 预测测试数据的类别标签
        
        if batch_size is None:  # 如果未指定批次大小
            batch_size = self.batch_size  # 使用默认批次大小
        with self.graph.as_default():  # 在计算图上下文中执行
            predictions = []  # 初始化预测结果列表
            idxs = np.arange(len(X_test))  # 生成数据索引
            batches = [idxs[k * batch_size: (k+1) * batch_size]  # 将数据分成批次 
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]  # 计算批次数量
            for batch in batches:  # 遍历每个批次
                predictions.append(self.sess.run(self.predictions, {self.input_ph:X_test[batch]}))  # 预测批次数据
            return np.concatenate(predictions)  # 拼接所有预测结果
        
    def fit(self, X, y, X_val=None, y_val=None, sources=None, max_epochs=None,  # 模型训练方法
            batch_size=None, save=False, load=False, sample_weight=None,  # 批次大小、保存、加载、样本权重参数
            metric='accuracy'):  # 评价指标
        
        self.num_classes = len(set(y))  # 计算类别数量
        self.metric = metric  # 设置评价指标
        if max_epochs is None:  # 如果未指定最大轮数
            max_epochs = self.max_epochs  # 使用默认最大轮数
        if batch_size is None:  # 如果未指定批次大小
            batch_size = self.batch_size  # 使用默认批次大小
        if not self.is_built:  # 如果模型未构建
            self.graph = tf.Graph()  # 创建新的计算图
            with self.graph.as_default():  # 在计算图上下文中
                config = tf.ConfigProto()  # 创建TensorFlow配置
                config.gpu_options.allow_growth=True  # 允许GPU内存动态增长
                self.sess = tf.Session(config=config)  # 创建TensorFlow会话
        with self.graph.as_default():  # 在计算图上下文中执行
            tf.set_random_seed(self.random_seed)  # 设置随机种子
            try:  # 尝试创建全局步数
                self.global_step = tf.train.create_global_step()  # 创建全局步数变量
            except ValueError:  # 如果已存在则获取
                self.global_step = tf.train.get_global_step()  # 获取现有全局步数
            if not self.is_built:  # 如果模型未构建
                self._build_model(X, y)  # 构建模型
                self.saver = tf.train.Saver()  # 创建模型保存器
            self._initialize()  # 初始化模型参数
            if len(X):  # 如果有训练数据
                if X_val is None and self.validation_fraction * len(X) > 2:  # 如果没有验证集且数据足够多
                    X_train, X_val, y_train, y_val, sample_weight, _ = train_test_split(  # 划分训练集和验证集
                        X, y, sample_weight, test_size=self.validation_fraction)  # 使用指定比例划分
                else:  # 否则
                    X_train, y_train = X, y  # 直接使用全部数据作为训练集
                self._train_model(X_train, y_train, X_val=X_val, y_val=y_val,  # 训练模型
                                  max_epochs=max_epochs, batch_size=batch_size,  # 传入训练参数
                                  sources=sources, sample_weight=sample_weight)  # 传入数据源和样本权重
                if save and self.address is not None:  # 如果需要保存且有保存地址
                    self.saver.save(self.sess, self.address)  # 保存模型
            
    def _train_model(self, X, y, X_val, y_val, max_epochs, batch_size,  # 私有方法：训练模型
                     sources=None, sample_weight=None):  # 数据源和样本权重参数
        
        
        assert len(X)==len(y), 'Input and labels not the same size'  # 断言检查输入和标签长度一致
        self.history = {'metrics':[], 'idxs':[]}  # 初始化训练历史记录
        stop_counter = 0  # 早停计数器
        best_performance = None  # 最佳性能记录
        for epoch in range(max_epochs):  # 遍历每个训练轮数
            vals_metrics, idxs = self._one_epoch(  # 执行一个训练轮次
                X, y, X_val, y_val, batch_size, sources=sources, sample_weight=sample_weight)  # 传入训练参数
            self.history['idxs'].append(idxs)  # 记录批次索引
            self.history['metrics'].append(vals_metrics)  # 记录评估指标
            if self.early_stopping and X_val is not None:  # 如果启用早停且有验证集
                current_performance = np.mean(val_acc)  # 计算当前性能
                if best_performance is None:  # 如果是第一次
                    best_performance = current_performance  # 设置为最佳性能
                if current_performance > best_performance:  # 如果当前性能更好
                    best_performance = current_performance  # 更新最佳性能
                    stop_counter = 0  # 重置早停计数器
                else:  # 否则
                    stop_counter += 1  # 早停计数器增1
                    if stop_counter > self.early_stopping:  # 如果达到早停阈值
                        break  # 停止训练
        
    def _one_epoch(self, X, y, X_val, y_val, batch_size, sources=None, sample_weight=None):  # 执行一个训练轮次
        
        vals = []  # 初始化验证集评价结果列表
        if sources is None:  # 如果没有数据源分组
            if sample_weight is None:  # 如果没有样本权重
                idxs = np.random.permutation(len(X))  # 随机打乱数据索引
            else:  # 如果有样本权重
                idxs = np.random.choice(len(X), len(X), p=sample_weight/np.sum(sample_weight))  # 按权重采样    
            batches = [idxs[k*batch_size:(k+1) * batch_size]  # 将数据分成批次
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]  # 计算批次数量
            idxs = batches  # 更新索引为批次列表
        else:  # 如果有数据源分组
            idxs = np.random.permutation(len(sources.keys()))  # 随机打乱数据源索引
            batches = [sources[i] for i in idxs]  # 根据数据源生成批次
        for batch_counter, batch in enumerate(batches):  # 遍历每个批次
            self.sess.run(self.train_op,  # 执行训练操作
                          {self.input_ph:X[batch], self.labels:y[batch],  # 传入输入数据和标签
                           self.dropout_ph:self.dropout})  # 传入dropout参数
            if X_val is not None:  # 如果有验证集
                if self.metric=='accuracy':  # 如果评价指标是准确率
                    vals.append(self.score(X_val, y_val))  # 计算准确率
                elif self.metric=='f1':  # 如果评价指标是F1分数
                    vals.append(f1_score(y_val, self.predict(X_val)))  # 计算F1分数
                elif self.metric=='auc':  # 如果评价指标是AUC
                    vals.append(roc_auc_score(y_val, self.predict_proba(X_val)[:,1]))  # 计算AUC分数
                elif self.metric=='xe':  # 如果评价指标是交叉熵
                    vals.append(-self.prediction_cost(X_val, y_val))  # 计算负交叉熵
        return np.array(vals), np.array(idxs)  # 返回评价结果和批次索引
    
    def _initialize(self):  # 初始化模型参数
        
        tf.set_random_seed(self.random_seed)  # 设置随机种子
        uninitialized_vars = []  # 初始化未初始化变量列表
        if self.warm_start:  # 如果是热启动
            for var in tf.global_variables():  # 遍历所有全局变量
                try:  # 尝试获取变量值
                    self.sess.run(var)  # 运行变量
                except tf.errors.FailedPreconditionError:  # 如果变量未初始化
                    uninitialized_vars.append(var)  # 添加到未初始化列表
        else:  # 如果不是热启动
            uninitialized_vars = tf.global_variables()  # 所有变量都需要初始化
        self.sess.run(tf.initializers.variables(uninitialized_vars))  # 初始化未初始化的变量
        
    def _build_model(self, X, y):  # 构建模型结构
        
        self.num_classes = len(set(y))  # 计算类别数量
        if self.initializer is None:  # 如果未指定初始化器
            initializer = tf.initializers.variance_scaling(distribution='uniform')  # 使用方差缩放初始化
        if self.activation is None:  # 如果未指定激活函数
            activation = lambda x: tf.nn.relu(x)  # 使用ReLU激活函数
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=(None,) + X.shape[1:], name='input')  # 创建输入占位符
        self.dropout_ph = tf.placeholder_with_default(  # 创建dropout占位符
            tf.constant(self.dropout, dtype=tf.float32), shape=(), name='dropout')  # 设置默认dropout值
        if self.mode=='regression':  # 如果是回归任务
            self.labels = tf.placeholder(dtype=tf.float32, shape=(None, ), name='label')  # 创建浮点数标签占位符
        else:  # 如果是分类任务
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None, ), name='label')  # 创建整数标签占位符
        x = tf.reshape(self.input_ph, shape=(-1, np.prod(X.shape[1:])))  # 将输入重塑为二维
        for layer, hidden_unit in enumerate(self.hidden_units):  # 遍历每个隐藏层
            with tf.variable_scope('dense_{}'.format(layer)):  # 为每层创建变量作用域
                x = self._dense(x, hidden_unit, dropout=self.dropout_ph,  # 添加全连接层
                           initializer=self.initializer, activation=activation)  # 传入初始化器和激活函数
        with tf.variable_scope('final'):  # 创建输出层变量作用域
            self.prelogits = x  # 保存最后一层特征
            self._final_layer(self.prelogits, self.num_classes, self.mode)  # 添加最终输出层
        self._build_train_op()  # 构建训练操作
        
    def _build_train_op(self):  # 构建训练操作
        
        """Build taining specific ops for the graph."""  # 为计算图构建训练特定操作
        learning_rate = tf.constant(self.learning_rate, tf.float32) ##fixit  # 将学习率转为常量张量
        trainable_variables = tf.trainable_variables()  # 获取所有可训练变量
        grads = tf.gradients(self.loss, trainable_variables)  # 计算损失对可训练变量的梯度
        self.grad_flat = tf.concat([tf.reshape(grad, (-1, 1)) for grad in grads], axis=0)  # 将梯度展平并连接
        if self.optimizer == 'sgd':  # 如果优化器是随机梯度下降
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)  # 创建随机梯度下降优化器
        elif self.optimizer == 'mom':  # 如果优化器是动量
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)  # 创建动量优化器
        elif self.optimizer == 'adam':  # 如果优化器是Adam
            optimizer = tf.train.AdamOptimizer(learning_rate)  # 创建Adam优化器
        apply_op = optimizer.apply_gradients(  # 创建梯度应用操作
            zip(grads, trainable_variables),  # 将梯度与变量配对
            global_step=self.global_step, name='train_step')  # 设置全局步数和名称
        train_ops = [apply_op] + self._extra_train_ops + tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # 收集所有训练操作
        previous_ops = [tf.group(*train_ops)]  # 将训练操作组合
        with tf.control_dependencies(previous_ops):  # 在控制依赖下
            self.train_op = tf.no_op(name='train')  # 创建空操作作为训练操作   
        self.is_built = True  # 标记模型已构建
    
    def _final_layer(self, x, num_classes, mode):  # 构建最终输出层
        
        if mode=='regression':  # 如果是回归任务
            self.logits = self._dense(x, 1, dropout=self.dropout_ph)  # 创建输出为1的全连接层
            self.predictions = tf.reduce_sum(self.logits, axis=-1)  # 将输出降维为标量
            regression_loss = tf.nn.l2_loss(self.predictions - self.labels) ##FIXIT  # 计算L2损失
            self.prediction_loss = tf.reduce_mean(regression_loss, name='l2')  # 计算平均L2损失
            residuals = self.predictions - self.labels  # 计算残差
            var_predicted = tf.reduce_mean(residuals**2) - tf.reduce_mean(residuals)**2  # 计算预测值方差
            var_labels = tf.reduce_mean(self.labels**2) - tf.reduce_mean(self.labels)**2  # 计算真实值方差
            self.prediction_score = 1 - var_predicted/(var_labels + 1e-12)  # 计算R平方分数
        else:  # 如果是分类任务
            self.logits = self._dense(x, num_classes, dropout=self.dropout_ph)  # 创建输出维度为类别数的全连接层
            self.probs = tf.nn.softmax(self.logits)  # 将logits转为概率
            xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(  # 计算稀疏交叉熵损失
                logits=self.logits, labels=tf.cast(self.labels, tf.int32))  # 传入logits和标签
            self.prediction_loss = tf.reduce_mean(xent_loss, name='xent')  # 计算平均交叉熵损失
            self.predictions = tf.argmax(self.probs, axis=-1, output_type=tf.int32)  # 获取最大概率类别作为预测结果
            correct_predictions = tf.equal(self.predictions, self.labels)  # 检查预测是否正确
            self.prediction_score = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))  # 计算准确率
        self.loss = self.prediction_loss + self._reg_loss()  # 总损失 = 预测损失 + 正则化损失
                
    def _dense(self, x, out_dim, dropout=tf.constant(0.), initializer=None, activation=None):  # 全连接层实现
        
        if initializer is None:  # 如果未指定初始化器
            initializer = tf.initializers.variance_scaling(distribution='uniform')  # 使用方差缩放初始化
        w = tf.get_variable('DW', [x.get_shape()[1], out_dim], initializer=initializer)  # 创建权重变量
        b = tf.get_variable('Db', [out_dim], initializer=tf.constant_initializer())  # 创建偏置变量
        x = tf.nn.dropout(x, 1. - dropout)  # 应用dropout
        if activation:  # 如果有激活函数
            x = activation(x)  # 应用激活函数
        return tf.nn.xw_plus_b(x, w, b)  # 返回线性变换结果
    
    def _reg_loss(self, order=2):  # 正则化损失计算
        """Regularization loss for weight decay."""  # 权重衰减的正则化损失
        losss = []  # 初始化损失列表
        for var in tf.trainable_variables():  # 遍历所有可训练变量
            if var.op.name.find(r'DW') > 0 or var.op.name.find(r'CW') > 0: ##FIXIT  # 如果是权重变量
                if order==2:  # 如果是L2正则化
                    losss.append(tf.nn.l2_loss(var))  # 添加L2损失
                elif order==1:  # 如果是L1正则化
                    losss.append(tf.abs(var))  # 添加L1损失
                else:  # 其他情况
                    raise ValueError("Invalid regularization order!")  # 抛出错误
        return tf.multiply(self.weight_decay, tf.add_n(losss))  # 返回加权正则化损失


class CShapNN(ShapNN):  # 卷积神经网络类，继承自ShapNN
    
    def __init__(self, mode, hidden_units=[100], kernel_sizes=[],  # 初始化函数：设置卷积神经网络参数
                 strides=None, channels=[], learning_rate=0.001,  # 步长、通道数、学习率
                 dropout = 0., activation=None, initializer=None, global_averaging=False,  # dropout、激活函数、初始化器、全局平均池化
                weight_decay=0.0001, optimizer='adam', batch_size=128,  # 权重衰减、优化器、批次大小
                warm_start=False, max_epochs=100, validation_fraction=0.1,  # 热启动、最大轮数、验证集比例
                early_stopping=0, address=None, test_batch_size=1000, random_seed=666):  # 早停、保存地址、测试批次大小、随机种子
        
        self.mode = mode  # 模型模式
        self.test_batch_size = test_batch_size  # 测试批次大小
        self.kernels = []#FIXIT  # 卷积核列表
        self.kernel_sizes = kernel_sizes  # 卷积核大小列表
        self.channels = channels  # 通道数列表
        self.global_averaging = global_averaging  # 是否使用全局平均池化
        if len(kernel_sizes) > 0:  # 如果有卷积层
            assert len(channels)==len(kernel_sizes), 'Invalid channels or kernel_sizes'  # 检查通道数和卷积核数量一致
        if strides is None:  # 如果未指定步长
            self.strides = [1] * len(kernel_sizes)  # 设置默认步长为1
        else:  # 否则
            self.strides = strides  # 使用指定步长
        self.batch_size = batch_size  # 批次大小
        self.hidden_units = hidden_units  # 隐藏层单元数
        self.initializer = initializer  # 初始化器
        self.activation = activation  # 激活函数
        self.dropout = dropout  # dropout率
        self.weight_decay = weight_decay  # 权重衰减
        self.optimizer = optimizer  # 优化器
        self.learning_rate = learning_rate  # 学习率
        self.warm_start = warm_start  # 热启动
        self.max_epochs = max_epochs  # 最大轮数
        self.early_stopping = early_stopping  # 早停
        self.validation_fraction = validation_fraction  # 验证集比例
        self.address = address  # 保存地址
        self._extra_train_ops = []  # 额外训练操作
        self.random_seed = random_seed  # 随机种子
        self.graph = tf.Graph()  # 创建计算图
        self.is_built = False  # 模型构建标记
        with self.graph.as_default():  # 在计算图上下文中
            config = tf.ConfigProto()  # 创建配置
            config.gpu_options.allow_growth=True  # 允许GPU内存动态增长
            self.sess = tf.Session(config=config)  # 创建会话
            
    def _conv(self, x, filter_size, out_filters, strides, activation=None):
        
        in_filters = int(x.get_shape()[-1])
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable(
            'DW', [filter_size, filter_size, in_filters, out_filters],
            tf.float32, initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0/n)))
        self.kernels.append(kernel)
        x = tf.nn.conv2d(x, kernel, strides, padding='SAME')
        if activation:
            x = activation(x)
        return x
    
    def _stride_arr(self, stride):
        
        if isinstance(stride, int):
            return [1, stride, stride, 1]
        if len(stride)==2:
            return [1, stride[0], stride[1], 1]
        if len(stride)==4:
            return stride
        raise ValueError('Invalid value!')  
        
    def _build_model(self, X, y):
        
        
        if self.initializer is None:
            initializer = tf.initializers.variance_scaling(distribution='uniform')
        if self.activation is None:
            activation = lambda x: tf.nn.relu(x)
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=(None,) + X.shape[1:], name='input')
        self.dropout_ph = tf.placeholder_with_default(
            tf.constant(self.dropout, dtype=tf.float32), shape=(), name='dropout')
        if self.mode=='regression':
            self.labels = tf.placeholder(dtype=tf.float32, shape=(None, ), name='label')
        else:
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None, ), name='label')
        if len(X.shape[1:]) == 2:
            x = tf.reshape(self.input_ph, [-1, X.shape[0], X.shape[1], 1])
        else:
            x = self.input_ph
        for layer, (kernel_size, channels, stride) in enumerate(zip(
            self.kernel_sizes, self.channels, self.strides)):
            with tf.variable_scope('conv_{}'.format(layer)):
                x = self._conv(x, kernel_size, channels, self._stride_arr(stride), activation=activation)
        if self.global_averaging:
            x = tf.reduce_mean(x, axis=(1,2))
        else:
            x = tf.reshape(x, shape=(-1, np.prod(x.get_shape()[1:])))
        for layer, hidden_unit in enumerate(self.hidden_units):
            with tf.variable_scope('dense_{}'.format(layer)):
                x = self._dense(x, hidden_unit, dropout=self.dropout_ph, 
                           initializer=self.initializer, activation=activation)
                
        with tf.variable_scope('final'):
            self.prelogits = x
            self._final_layer(self.prelogits, len(set(y)), self.mode)
        self._build_train_op()
