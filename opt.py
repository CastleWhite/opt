import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

def load_data():
    # 载入数据集
    dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(dataset["train_set_x"][:]) # train set features
    train_set_y_orig = np.array(dataset["train_set_y"][:]) # train set labels

    classes = np.array(dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        
    return train_set_x_orig, train_set_y_orig, classes

def show_image(train_x_orig, train_y, index, classes):
    # 展示数据集中的图片
    plt.imshow(train_x_orig[index])
    plt.show()
    print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
    
def sigmoid(z):
    #    Compute the sigmoid of z
    s = 1 / (1 + np.exp(-z))
    
    return s

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_h, n_y)
                    b2 -- bias vector of shape (1, n_y)
    """
    
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_h,n_y)*0.01
    b2 = np.zeros((1,n_y))
        
    parameters = {"w_1": W1,"b_1": b1,"w_2": W2,"b_2": b2}
    
    return parameters

def propagate(parameters, X, Y_label):
    # Implement the objective function and its gradient 
    m = X.shape[1]
    
    W1 = parameters["w_1"]
    b1 = parameters["b_1"]
    w_2 = parameters["w_2"]
    b_2 = parameters["b_2"]

    w_11 = W1[0].T
    w_12 = W1[1].T
    w_13 = W1[2].T
        
    b_11 = b1[0]
    b_12 = b1[1]
    b_13 = b1[2]
    
    A_1 = sigmoid(np.dot(w_11.T, X) + b_11)
    A_2 = sigmoid(np.dot(w_12.T, X) + b_12)
    A_3 = sigmoid(np.dot(w_13.T, X) + b_13)
    A = np.array([A_1,A_2,A_3])

    Y = sigmoid(np.dot(w_2.T, A) + b_2)

    cost = -1 / m * np.sum(Y_label * np.log(Y) + (1 - Y_label) * np.log(1 - Y))
    cost = np.squeeze(cost)   # 将表示向量的数组转换为秩为1的数组

    dw_2 = 1 / m * (np.dot(A, (Y - Y_label).T))
    db_2 = 1 / m * np.sum(Y - Y_label)
    
    db_1 = np.dot(w_2,(Y - Y_label)) * A*(1-A)
    dw_1 = 1 / m * np.dot(db_1,X.T)
    
    # dw_11 = dw_1[0].T
    # dw_12 = dw_1[1].T
    # dw_13 = dw_1[2].T
    db_1 = 1 / m * np.sum(db_1)
    # db_11 = db_1[0]
    # db_12 = db_1[1]
    # db_13 = db_1[2]
    
    grads = {"dw_1": dw_1,"db_1": db_1,"dw_2": dw_2,"db_2": db_2}
    
    return grads, cost

def update(parameters, grads, step=0.005):
    # 根据方向和步长更新变量w、b
    w_1 = parameters["w_1"]
    b_1 = parameters["b_1"]
    w_2 = parameters["w_2"]
    b_2 = parameters["b_2"]
    dw_1 = grads["dw_1"]
    db_1 = grads["db_1"]
    dw_2 = grads["dw_2"]
    db_2 = grads["db_2"]

    w_1 = w_1-step*dw_1
    b_1 = b_1-step*db_1
    w_2 = w_2-step*dw_2
    b_2 = b_2-step*db_2

    para = {"w_1": w_1,"b_1": b_1,"w_2": w_2,"b_2": b_2}

    return para

def optimize(train_x, train_y, eps=1e-5, print_pro=False, step=0.001):
    # 最速下降算法
    costs = []
    norms = []
    k = 0
    n = train_x.shape[0]
    pa = initialize_parameters(n,3,1)
    while k>=0:
        # 计算梯度以及目标函数
        grads,cost = propagate(pa,train_x,train_y)
        dw_1 = grads["dw_1"]
        db_1 = grads["db_1"]
        dw_2 = grads["dw_2"]
        db_2 = grads["db_2"]
        # 计算梯度的模
        norm_grad = math.sqrt(np.sum(np.square(dw_1))+np.sum(np.square(db_1))+np.sum(np.square(dw_2))+np.sum(np.square(db_2)))
        if print_pro and k%500 == 0:                               #每500次迭代打印一次信息
            costs.append(cost)
            norms.append(norm_grad)
            print("经过{}次迭代，目标函数值为{}，梯度模为{}".format(k,cost,norm_grad))
        if norm_grad <= eps:                                 #若达到终止条件，输出信息，跳出循环
            print(norm_grad,eps,norm_grad <= eps)
            break

        pa = update(pa,grads,step)
        k = k+1
        
    return pa, costs, norms    

def val(X, parameters):
    # 最优解的检验与实施
    W1 = parameters["w_1"]
    b1 = parameters["b_1"]
    w_2 = parameters["w_2"]
    b_2 = parameters["b_2"]
    w_11 = W1[0].T
    w_12 = W1[1].T
    w_13 = W1[2].T
    b_11 = b1[0]
    b_12 = b1[1]
    b_13 = b1[2]
    
    A_1 = sigmoid(np.dot(w_11.T, X) + b_11)
    A_2 = sigmoid(np.dot(w_12.T, X) + b_12)
    A_3 = sigmoid(np.dot(w_13.T, X) + b_13)
    A = np.array([A_1,A_2,A_3])

    Y_predict = sigmoid(np.dot(w_2.T, A) + b_2)
    
    return Y_predict>0.5
    
def main():
    # 载入数据集
    train_x_orig , train_y , classes = load_data()
    # 展示数据集中的图像。可以更改序号，查看其它图片
    index = 3
    show_image(train_x_orig, train_y, index, classes)

    # Reshape the training examples , 转化为列向量
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.

    print ("train_x's shape: " + str(train_x.shape))
    print ("train_y's shape: " + str(train_y.shape))
    
    # 进行优化，求解最优解
    eps = 0.001
    print_pro = True
    step = 0.001
    para, costs, norms = optimize(train_x, train_y, eps, print_pro, step)
    
    # 得到最优解 
    w_1 = para["w_1"]
    b_1 = para["b_1"]
    w_2 = para["w_2"]
    b_2 = para["b_2"]
    w_11 = w_1[0].T
    w_12 = w_1[1].T
    w_13 = w_1[2].T
        
    b_11 = b_1[0]
    b_12 = b_1[1]
    b_13 = b_1[2]
    
    # 将最优解保存为文本
    np.savetxt('w1.txt',w_1)
    np.savetxt('b1.txt',b_1)
    np.savetxt('w2.txt',w_2)
    np.savetxt('b2.txt',b_2)
    with open('cost.txt','w') as f:
        f.write(str(costs))
    with open('norm_gradient.txt','w') as f:
        f.write(str(norms))
        
    # 检验和实施: 计算识别的准确率=正确数/总数
    y_predict_train = val(train_x, para)
    print("train accuracy: {} ".format(1 - np.mean(np.abs(y_predict_train - train_y))))
    
    # 画出优化过程
    plt.subplot(211)
    plt.plot(np.squeeze(costs))
    plt.ylabel('objective function')
    plt.xlabel('iterations (per 500)')
    
    plt.subplot(212)
    plt.plot(np.squeeze(norms))
    plt.ylabel('gradient')
    plt.xlabel('iterations (per 500)')
    
    plt.show()
    
main()