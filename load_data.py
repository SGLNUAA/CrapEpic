
import numpy as np
def creat_dataset(dataset, look_back):
    '''

    :param dataset:(day*T,15,15)
    :param look_back: 6
    :return:
    '''
    data_x = []
    data_y = []
    # print(look_back)
    for i in range(dataset.shape[0] - look_back):
        data_x.append(dataset[i:i + look_back,:,:])
        data_y.append(dataset[i + look_back,:,:])
    return np.asarray(data_x), np.asarray(data_y)  # asarray可以将元组，列表，元组列表，列表元组转化成ndarray对象
def scalar(data, tranrange=(0, 1)):
    '''
    k=(b-a)/(max-min)
    nor_Y=a+k(Y-min)
    :param data_train:size:  (train_size,225)
    :param data_test:size:  (train_size,225)
    :param range:(a,b)
    :return:size不变,minmiax_para用于反归一化
    '''
    a = tranrange[0]
    b = tranrange[1]
    para=np.zeros((225,2))
    normed_data = np.zeros(data.shape)
    for k in range(data.shape[1]):
        # 每个区域有day*24个时间，对这些数据进行归一化
        brain_slice = data[:,k]
        # if np.sum(brain_slice) == 0:
        if np.sum(brain_slice==0)>len(brain_slice)/2:#如果超过一半车流为0或者最大车流为10都筛除
            para[k][0]=0
            para[k][1]=0
        elif np.sum(brain_slice) != 0:
            min=np.min(brain_slice)
            max=np.max(brain_slice)
            if max<10:
                para[k][0]=0
                para[k][1]=0
            else:
                para[k][0]=min
                para[k][1]=max
            brain_slice = a + (b - a) * (brain_slice - min) / (max - min)  # 24
            normed_data[:,k] = brain_slice
    print(sorted(para,key=lambda x:x[1]))
    return normed_data,para##归一化信息，（最小值，最小值）
def getdata_1h(trainlist,testlist,city_name,seq_len):
    '''

    :param trainlist: (train_size,15,15,T)
    :param testlist: (test_size,15,15,T)
    :param fiename: SH or NJ
    :return:  trainx,trainy,testx,testy,label,para_test
    '''
    ##训练集
    train_label = np.swapaxes(trainlist, 2, 3)
    train_label = np.swapaxes(train_label, 1, 2)  # (test_size,T,15,15)
    train_label = train_label.reshape(-1, 15, 15)
    train_label = train_label[seq_len:]
    # 测试集
    label = np.swapaxes(testlist, 2, 3)
    label = np.swapaxes(label, 1, 2)  # (test_size,T,15,15)
    label = label.reshape(-1, 15, 15)
    label = label[seq_len:]
    ############################################归一化,对每个区域的数据在时间维度归一化
    trainlist = np.swapaxes(trainlist, 2, 3)
    trainlist = np.swapaxes(trainlist, 1, 2)
    trainlist = trainlist.reshape(-1, 225)
    trainlist, para_train = scalar(trainlist)
    trainlist = trainlist.reshape(-1, 15, 15) #120x15x15
    normed_train_label = trainlist[seq_len:]

    testlist = np.swapaxes(testlist, 2, 3)
    testlist = np.swapaxes(testlist, 1, 2)
    testlist = testlist.reshape(-1, 225)
    testlist, para_test = scalar(testlist)
    testlist = testlist.reshape(-1, 15, 15)
    normed_label = testlist[seq_len:]

    #####删除空区域
    label = label.reshape(-1, 225)
    normed_label = normed_label.reshape(-1, 225)
    for i in range(225):
        if para_test[i][1] == 0:
            label[:, i] = 0 * label[:, i]
            normed_label[:, i] = 0 * normed_label[:, i]

    ###################################################生成训练、测试数据
    print('{}训练数据：'.format(city_name))
    train_X, train_Y = creat_dataset(trainlist, seq_len) # (batch*batchsize,seq,15,15)
    print( train_X.shape, train_Y.shape)  # (num, seq,15, 15 ) (num, 15, 15)
    test_X, test_Y = creat_dataset(testlist, seq_len)
    print('{}测试数据：'.format(city_name))
    print( test_X.shape, test_Y.shape)  #  (num, seq,15, 15,) (num, 15, 15)
    return train_X,train_Y,test_X,test_Y,label,normed_label,para_test,para_train

def load_data_1h(train_size,seq_len,city_name):
    '''
    finetune:  跨城市模型微调无迁移，使用目标城市label
    transfer： 使用POI匹配的跨城市模型，使用目标城市label
    only_poi: 只用poi微调，不进行标签损失
    Conv3d的规定输入数据格式为(batch, channel, Depth, Height, Width)
    '''
    ###超参数input = Variable(torch.rand(batch_size, seq_len, inp_chans, shape[0], shape[1]))
    ###input = Variable(torch.rand(batch_size, seq_len, inp_chans, shape[0], shape[1]))

    path_sh = ['./dataset/SH_data/SH_flow{}.npy'.format(i) for i in range(32, 57)]
    path_nj = ['./dataset/NJ_data/NJ_flow{}.npy'.format(i) for i in range(1, 32)]
    path_hk= ['./dataset/HK_data/HK_flow{}.npy'.format(i) for i in range(31)]
    #######################         sh          #############
    if city_name=='SH':
        data_sh = []
        for i in range(len(path_sh)):
            data_sh.append(np.load(path_sh[i])[...,:24].astype(float))  # 选取
        data_sh = np.asarray(data_sh)  # ((31,15,15,96)  (24,31)
        trainlist = data_sh[:train_size]  # (train_size,15,15,T)  (16)
        testlist = data_sh[-14:]  # (test_size,15,15,T)   (8)
    ##########################       nj     #############
    elif city_name=='NJ':
        data_nj=[]
        for i in range(len(path_nj)):
            data_nj.append(np.load(path_nj[i]).astype(float))
        data_nj = np.asarray(data_nj)   #31x15x15x24
        trainlist = data_nj[2:2 + train_size]
        testlist = data_nj[-14:]
    #######################       hk    #############
    elif city_name=='HK':
        data_hk=[]
        for i in range(len(path_hk)):
            data_hk.append(np.load(path_hk[i]).astype(float))
        data_hk = np.asarray(data_hk)
        trainlist = data_hk[: train_size]
        testlist= data_hk[15 + train_size:30 + train_size]
    else:
        print('City_name Error!')
        exit()
    print('trainlist_source.shape,testlist_source.shape:',trainlist.shape,testlist.shape) #(5, 15, 15, 24) (14, 15, 15, 24)

    train_X, train_Y, test_X, test_Y, label, normed_label, test_para, train_para = getdata_1h(trainlist,testlist, city_name, seq_len)
    return train_X, train_Y, test_X, test_Y, label, normed_label, test_para,train_para


