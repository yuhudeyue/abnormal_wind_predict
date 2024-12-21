import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from numpy import concatenate
import math
import h5py
import os
#from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import scipy.io as sio
Batch_size = 100
Epochs = 50
TIMESTEPS = 60
OUTPUTDIM  = 1
def create_dataset(dataset, TIMESTEPS, OUTPUTDIM):
    dataX, dataY = [], []
    for i in range(len(dataset)-TIMESTEPS-OUTPUTDIM-1):
        a = dataset[i:(i+TIMESTEPS), :]
        dataX.append(a)
        b = dataset[(i+TIMESTEPS):(i+TIMESTEPS+OUTPUTDIM), :]
        #b = b.reshape(b.shape[1])
        b = reshape_y(b,b.shape[1])
        dataY.append(b)
    return np.array(dataX), np.array(dataY)

def reshape_y_hat(y_hat,dim):

    i = 0
    tmp_y = []
    while i < len(y_hat):
        t = 0

        while t < (y_hat.shape[1]):
            tmp = y_hat[i,t:t+dim]
            t = t + dim
            tmp_y.append(tmp)
        i = i+1

    re_y = np.array(tmp_y,dtype='float32')
    return  re_y

def reshape_invy_hat(invy_hat,OUTPUTDIM):
    re_invy = []
    i = 0

    while i < len(invy_hat):

        tmp_y = invy_hat[i:i+OUTPUTDIM,:]
        i = i+OUTPUTDIM
        re_invy.append(tmp_y)
    re_invy = np.array(re_invy,dtype='float32')
    return  re_invy


def reshape_y(tempb,dim):
    b = []
    for j in range(len(tempb)):
        for k in range(dim):
            b.append(tempb[j, k])

    return  b


def reshape_dataset(tempb,dim0,dim1):
    a = []
    b = []
    for j in range(dim0):
        for k in range(dim1):
            a.append(tempb[k, j])
        b.append(a)
    return  np.array(b)

def train_model(train_X,train_Y,Epochs,Batch_size):

    # 设计网络
    model = Sequential()
    model.add(LSTM(140,input_shape=(train_X.shape[1],train_X.shape[2]),return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(140,return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(train_Y.shape[1]))
    model.add(Activation('relu'))
    model.compile(loss='mse',optimizer='adam')
    history = model.fit(train_X,train_Y,epochs=Epochs,batch_size=Batch_size,validation_split=0.115, verbose=1)
    #model.add(LSTM(batch_size, input_shape=(train_X.shape[1], train_X.shape[2])))
    #model.add(Dense(output_dim=OUTPUTDIM))
    #model.compile(loss='mae', optimizer='adam')
    # 拟合神经网络模型validation_data=(test_X, test_Y)
    #history = model.fit(train_X, train_Y, epochs=300, batch_size=batch_size, validation_split=0.115, verbose=2,
    #                    shuffle=False)

    return model,history

def NormalizeMult(data):
    '''
    归一化 适用于单维和多维
    返回归一化后的数据和最大最小值
    '''
    normalize = np.arange(2*data.shape[1],dtype='float64')
    normalize = normalize.reshape(data.shape[1],2)

    for i in range(0,data.shape[1]):

        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])

        normalize[i,0] = listlow
        normalize[i,1] = listhigh

        delta = listhigh - listlow
        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta

    return  data,normalize

def FNormalizeMult(data,normalize):

    data = np.array(data,dtype='float64')
    #列
    for i in  range(0,data.shape[1]):
        listlow =  normalize[i,0]
        listhigh = normalize[i,1]
        delta = listhigh - listlow
        #行
        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i]  =  data[j,i]*delta + listlow

    return data

#使用训练数据的归一化
def NormalizeMultUseData(data,normalize):

    for i in range(0, data.shape[1]):

        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow

        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta

    return  data








os.environ["CUDA_DEVICES_ORDER"]='PCI_BUS_IS'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
f = open('results_phace_space_8_2_out_y.txt', 'a')
save_folder = 'saved_model_2020_8_2_out_y'
filenames = os.listdir('new_phase_space_2020_8_2_out_y')
for filename in filenames:
    Y = h5py.File('new_phase_space_2020_8_2_out_y/'+filename)
    rMSE_total = 0
    acc_total = 0
    f.write('10-->7 hidden=6, l_fc=1[7], l_r=1 \n')
    f.write('filename' + 'point' + ' MSE ' + ' rMSE ' + ' PCC ' + ' ACC ' + '\n')
    i = 1

    f.write(str(i) + ':')

    # 加载数据集
    # sst = h5py.File('the_data.mat')
    # x = sst['the_data']
    #for group in Y.keys():
    #   print(group)
    #    group_read = Y[group]
    #    for subgroup in group_read.keys():
    #       print(subgroup)
    x = Y['detected']

    data_sst = x
    ## 整数编码
    dataframe = pd.DataFrame(data_sst)
    dataset = dataframe.values
    #dataset = np.array(data_sst)
    dataset = dataset.astype('float32')
    #dataset = np.transpose(dataset)
    #dataset = reshape_dataset(dataset, dataset.shape[1], dataset.shape[0])
    #dataset = dataset.reshape(dataset.shape[1], dataset.shape[0]))
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    #yyy = np.zeros(2,2,2)
    #xxx = scaler.fit_transform(yyy)
    dataset = scaler.fit_transform(dataset)
    #dataset,normalize = NormalizeMult(dataset)
    values = dataset
    n_train_hours = 35000
    n_test_hours = 15000 + 30000
    train = values[:n_train_hours, :]
    test = values[n_train_hours:n_test_hours, :]
    # 分为输入输出

    train_X, train_Y = create_dataset(train, TIMESTEPS, OUTPUTDIM)
    test_X, test_Y = create_dataset(test, TIMESTEPS, OUTPUTDIM)

    # train_X, train_Y = train[:, :-1], train[:, -1]
    # test_X, test_Y = test[:, :-1], test[:, -1]
    # 重塑成3D形状 [样例, 时间步, 特征]
    # train_X = train_X.reshape((train_X.shape[0], TIMESTEPS, train_X.shape[1]))
    # test_X = test_X.reshape((test_X.shape[0], TIMESTEPS, test_X.shape[1]))
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)


    # 绘制历史数据
    model,history = train_model(train_X,train_Y,Epochs,Batch_size)
    # 做出预测
    testPredict = model.predict(test_X)
    the_last_name = filename.rfind('.mat')
    new_filename = filename[:the_last_name]
    model.save(save_folder+'/'+new_filename+'.pkl')
    # joblib.dump(model, 'saved_model/test_2019_12_9_iteration1000_model.pkl')

    print(model.evaluate(test_X, test_Y, batch_size=Batch_size))

    yhat = model.predict(test_X)
    y_hat = reshape_y_hat(yhat, test_X.shape[2])
    testY = reshape_y_hat(test_Y, test_X.shape[2])
    # yhat = yhat.reshape(yhat.shape[0], 1, yhat.shape[1])

    # test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # testX = test_X[:, 0, 0]
    # testX = testX.reshape((test_X.shape[0], 1, 1))
    # 反向转换预测值比例
    # inv_yhat = concatenate((yhat, testX), axis=2)
    # for i in range(test_X.shape[2]-testX.shape[2]):
    #    inv_yhat = concatenate((inv_yhat, testX[:,:,0:1]), axis=2)

    # inv_yhat = inv_yhat.reshape(inv_yhat.shape[0], inv_yhat.shape[2])
    # inv_yhat = scaler.inverse_transform(inv_yhat)
    # yhat = yhat.reshape(yhat.shape[0],yhat.shape[2])
    inv_yhat = scaler.inverse_transform(y_hat)

    #inv_yhat = FNormalizeMult(yhat,normalize)
    # inv_yhat = inv_yhat[:, 0:OUTPUTDIM]

    # 反向转换实际值比例
    # testY = test_Y
    # testY = testY.reshape((len(testY), 1, testY.shape[1]))
    # inv_y = concatenate((testY, testX), axis=2)
    # for i in range(test_X.shape[2]-testX.shape[2]):
    #    inv_y = concatenate((inv_y, testX[:,:,0:1]), axis=2)
    # inv_y = inv_y.reshape(inv_y.shape[0], inv_y.shape[2])
    # inv_y = scaler.inverse_transform(inv_y)
    # inv_y = inv_y[:, 0:OUTPUTDIM]
    inv_y = scaler.inverse_transform(testY)
    #inv_y = FNormalizeMult(test_Y,normalize)
    testY_ = inv_y
    testPredict_ = inv_yhat

    reshape_testY_ = reshape_invy_hat(testY_, OUTPUTDIM)
    reshape_testPredict_ = reshape_invy_hat(testPredict_, OUTPUTDIM)
    sio.savemat(save_folder + '/' + new_filename + '_predict.mat', {'reshape_testPredict_': reshape_testPredict_, 'reshape_testY_': reshape_testY_})

    # plot
    #filename = str(1)
    fig, ax = plt.subplots(1)
    test_values = testY_.reshape(-1, 1).flatten()
    plot_test, = ax.plot(test_values)
    predicted_values = testPredict_.reshape(-1, 1).flatten()
    plot_predicted, = ax.plot(predicted_values)
    plt.title('SST Predictions')
    plt.legend([plot_predicted, plot_test], ['predicted', 'true value'])
    plt.savefig(save_folder+'/' + new_filename + '_predict')
    plt.show()
    MSE = mean_squared_error(testPredict_, testY_)
    print("MSE: %f" % MSE)
    rMSE = math.sqrt(MSE)
    print('rMSE:%f' % rMSE)
    pcc = np.corrcoef(testPredict_, testY_, rowvar=0)[0, 1]
    print("PCC: %f" % pcc)
    acc = 1 - np.mean(np.abs(testPredict_ - testY_) / testY_)
    print("ACC: %f" % acc)
    # sum
    rMSE_total = rMSE_total + rMSE
    acc_total = acc_total + acc
    # training epoch
    fig, ax = plt.subplots(1)
    loss, = ax.plot(history.history["loss"])
    val_loss, = ax.plot(history.history["val_loss"])
    plt.title('training process')
    plt.legend([loss, val_loss], ['loss', 'val loss'])
    plt.savefig(save_folder + '/' + new_filename + '_train')
    plt.show()

    # write to file
    f.write(filename + str(MSE) + ' ' + str(rMSE) + ' ' + str(pcc) + ' ' + str(acc) + '\n')
    w, n = inv_y.shape
    rMSE_ave = rMSE_total
    acc_ave = acc_total
    f.write('\n average rMSE   ACC \n')
    f.write(str(rMSE_ave) + ' ' + str(acc_ave))


f.close()