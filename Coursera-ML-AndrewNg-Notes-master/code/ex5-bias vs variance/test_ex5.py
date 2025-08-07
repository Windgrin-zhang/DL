#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试ex5偏差和方差练习的代码
"""

import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """加载数据"""
    d = sio.loadmat('ex5data1.mat')
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])

def cost(theta, X, y):
    """计算代价函数"""
    m = X.shape[0]
    inner = X @ theta - y
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)
    return cost

def gradient(theta, X, y):
    """计算梯度"""
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)
    return inner / m

def regularized_cost(theta, X, y, l=1):
    """计算正则化代价函数"""
    m = X.shape[0]
    regularized_term = (l / (2 * m)) * np.power(theta[1:], 2).sum()
    return cost(theta, X, y) + regularized_term

def regularized_gradient(theta, X, y, l=1):
    """计算正则化梯度"""
    m = X.shape[0]
    regularized_term = theta.copy()
    regularized_term[0] = 0  # 不正则化截距项
    regularized_term = (l / m) * regularized_term
    return gradient(theta, X, y) + regularized_term

def linear_regression_np(X, y, l=1):
    """线性回归优化"""
    theta = np.ones(X.shape[1])
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': False})
    return res

def poly_features(x, power, as_ndarray=False):
    """创建多项式特征"""
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)
    return df.values if as_ndarray else df

def normalize_feature(df):
    """特征归一化"""
    return df.apply(lambda column: (column - column.mean()) / column.std())

def prepare_poly_data(*args, power):
    """准备多项式数据"""
    def prepare(x):
        df = poly_features(x, power=power)
        ndarr = normalize_feature(df).values
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)
    return [prepare(x) for x in args]

def plot_learning_curve(X, y, Xval, yval, l=0):
    """绘制学习曲线"""
    training_cost, cv_cost = [], []
    m = X.shape[0]

    for i in range(1, m + 1):
        res = linear_regression_np(X[:i, :], y[:i], l=l)
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)
        training_cost.append(tc)
        cv_cost.append(cv)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc=1)
    plt.xlabel('Number of training examples')
    plt.ylabel('Cost')
    plt.title(f'Learning Curve (λ={l})')
    plt.grid(True)
    plt.show()

def main():
    """主函数"""
    print("开始测试ex5代码...")
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    X, y, Xval, yval, Xtest, ytest = load_data()
    print(f"训练集: X shape={X.shape}, y shape={y.shape}")
    print(f"验证集: Xval shape={Xval.shape}, yval shape={yval.shape}")
    print(f"测试集: Xtest shape={Xtest.shape}, ytest shape={ytest.shape}")
    
    # 2. 数据可视化
    print("\n2. 数据可视化...")
    df = pd.DataFrame({'water_level': X, 'flow': y})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='water_level', y='flow')
    plt.title('Water Level vs Flow')
    plt.xlabel('Water Level')
    plt.ylabel('Flow')
    plt.show()
    
    # 3. 添加截距项
    print("\n3. 添加截距项...")
    X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]
    print(f"添加截距项后: X shape={X.shape}")
    
    # 4. 测试代价函数和梯度
    print("\n4. 测试代价函数和梯度...")
    theta = np.ones(X.shape[1])
    print(f"初始theta: {theta}")
    print(f"初始代价: {cost(theta, X, y)}")
    print(f"初始梯度: {gradient(theta, X, y)}")
    
    # 5. 线性回归训练
    print("\n5. 线性回归训练...")
    res = linear_regression_np(X, y, l=0)
    print(f"优化完成，最终代价: {res.fun}")
    print(f"最优参数: {res.x}")
    
    # 6. 绘制拟合结果
    print("\n6. 绘制拟合结果...")
    b = res.x[0]  # 截距
    m = res.x[1]  # 斜率
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 1], y, label="Training data", alpha=0.7)
    plt.plot(X[:, 1], X[:, 1]*m + b, 'r-', label="Prediction", linewidth=2)
    plt.legend()
    plt.xlabel('Water Level')
    plt.ylabel('Flow')
    plt.title('Linear Regression Fit')
    plt.grid(True)
    plt.show()
    
    # 7. 学习曲线（线性模型）
    print("\n7. 学习曲线（线性模型）...")
    plot_learning_curve(X, y, Xval, yval, l=0)
    
    # 8. 多项式特征
    print("\n8. 创建多项式特征...")
    X, y, Xval, yval, Xtest, ytest = load_data()  # 重新加载原始数据
    X_poly, Xval_poly, Xtest_poly = prepare_poly_data(X, Xval, Xtest, power=8)
    print(f"多项式特征: X_poly shape={X_poly.shape}")
    print(f"前3行数据:\n{X_poly[:3, :]}")
    
    # 9. 学习曲线（多项式模型，无正则化）
    print("\n9. 学习曲线（多项式模型，λ=0）...")
    plot_learning_curve(X_poly, y, Xval_poly, yval, l=0)
    
    # 10. 学习曲线（多项式模型，λ=1）
    print("\n10. 学习曲线（多项式模型，λ=1）...")
    plot_learning_curve(X_poly, y, Xval_poly, yval, l=1)
    
    # 11. 学习曲线（多项式模型，λ=100）
    print("\n11. 学习曲线（多项式模型，λ=100）...")
    plot_learning_curve(X_poly, y, Xval_poly, yval, l=100)
    
    # 12. 寻找最优λ
    print("\n12. 寻找最优λ...")
    l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    training_cost, cv_cost = [], []
    
    for l in l_candidate:
        res = linear_regression_np(X_poly, y, l)
        tc = cost(res.x, X_poly, y)
        cv = cost(res.x, Xval_poly, yval)
        training_cost.append(tc)
        cv_cost.append(cv)
    
    best_l = l_candidate[np.argmin(cv_cost)]
    print(f"最优λ: {best_l}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(l_candidate, training_cost, 'o-', label='training')
    plt.plot(l_candidate, cv_cost, 's-', label='cross validation')
    plt.legend()
    plt.xlabel('λ')
    plt.ylabel('Cost')
    plt.title('Cost vs λ')
    plt.grid(True)
    plt.xscale('log')
    plt.show()
    
    # 13. 测试集评估
    print("\n13. 测试集评估...")
    for l in l_candidate:
        theta = linear_regression_np(X_poly, y, l).x
        test_cost = cost(theta, Xtest_poly, ytest)
        print(f'λ={l}: test cost = {test_cost:.6f}')
    
    print("\n测试完成！所有功能正常运行。")

if __name__ == "__main__":
    main() 