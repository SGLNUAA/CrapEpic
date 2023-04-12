import numpy as np
import matplotlib.pyplot as plt
def result(source_city,target_city,based_model):

    print('{}-{}-{}:'.format(source_city,target_city,based_model))
    path='./result\\'
    label=np.load(path+'{}_label.npy'.format(target_city))
    our_pre= np.load(path+'{}-{}-{}-ACPHC_pre.npy'.format(source_city,target_city,based_model))
    count_nan=np.zeros(225)
    for region_idx in range(225):
        if sum(label[:,region_idx])==0:
            count_nan[region_idx]+=1

    loss_our_pre = my_RMSE(our_pre, label)
    print('our_pre',loss_our_pre,loss_our_pre/(225-sum(count_nan)))
def my_MAE(data,label):
    '''

    :param data: size: -1,225
    :param label: size: -1,225
    :return:
    '''
    loss=[]
    for idx in range(data.shape[1]):
        t=np.mean(np.abs(data[:,idx]-label[:,idx]))
        loss.append(t)
    loss=sum(loss)
    return loss
def my_RMSE(data,label):
    #size: -1,225
    return sum(np.sqrt(np.mean(pow(data - label, 2),axis=0)))
def region_RMSE(data,label):
    #size: -1,225
    loss_region=[]
    for i in range(225):
        loss=np.sqrt(np.mean(pow(data[:,i] - label[:,i], 2)))
        loss_region.append(loss)
    return np.asarray(loss_region)


if __name__ == '__main__':
    model1='convlstm'
    model2='StepDeep'
    for source_city, target_city in [['SH','NJ'],['SH','HK'],['HK','NJ'],['NJ','HK']]:
        for model in [model1,model2]:
            result(source_city, target_city, model)
