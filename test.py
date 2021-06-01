import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

def load_data():
    # 载入测试集
    dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(dataset["test_set_y"][:]) # your test set labels

    classes = np.array(dataset["list_classes"][:]) # the list of classes
    
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        
    return test_set_x_orig, test_set_y_orig, classes

def sigmoid(z):
    #    Compute the sigmoid of z
    s = 1 / (1 + np.exp(-z))
    
    return s

def show_image(test_x_orig, test_y, index, classes):
    plt.imshow(test_x_orig[index])
    plt.show()
    print ("y = " + str(test_y[0,index]) + ". It's a " + classes[test_y[0,index]].decode("utf-8") +  " picture.")

def val(X, parameters):
    # 评估测试的结果
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
    test_x_orig , test_y , classes = load_data()
    # 展示数据集中的图像。可以更改序号，查看其它图片
    index = 13
    show_image(test_x_orig, test_y, index, classes)

    # Reshape the testing examples , 转化为列向量
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    # Standardize data to have feature values between 0 and 1.
    test_x = test_x_flatten/255.

    print ("test_x's shape: " + str(test_x.shape))
    print ("test_y's shape: " + str(test_y.shape))
    
    
    # 载入已求得的最优解 
    w_1 = np.loadtxt('w1.txt')
    b_1 = np.loadtxt('b1.txt')
    w_2 = np.loadtxt('w2.txt')
    b_2 = np.loadtxt('b2.txt')
    
    parameters = {"w_1": w_1,"b_1": b_1,"w_2": w_2,"b_2": b_2}
    
    # 检验和实施: 计算识别的准确率=正确数/总数
    y_predict_test = val(test_x, parameters)
    print("test accuracy: {} ".format(1 - np.mean(np.abs(y_predict_test - test_y))))

main()