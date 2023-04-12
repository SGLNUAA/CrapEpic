import numpy as np
import pandas as pd
import sklearn.metrics as sm

def get_total_data():###
    ######### SH
    print('sh', end=' ')
    path1 = ['./dataset/SH_data/SH_flow{}.npy'.format(i ) for i in range(32,57)]
    file_name = (path1)
    dataset1 = []
    for i in range(len(file_name)):
        data=np.load(file_name[i])[...,:24].astype(float)   #(15,15,24)
        dataset1.append(data)
    dataset_SH = np.asarray(dataset1)
    np.save('./SH_flow',dataset_SH)
    ######### NJ
    print('nj',end=' ')
    path2 = ['./dataset/NJ_data/NJ_flow{}.npy'.format(i ) for i in range(1,32)]
    dataset2= []
    for i in range(len(path2)):
        dataset2.append(np.load(path2[i]).astype(float))
    dataset_NJ = np.asarray(dataset2)
    np.save('./NJ_flow',dataset_NJ)
    ########## HK
    path3 = ['./dataset/HK_data/HK_flow{}.npy'.format(i ) for i in range(31)]
    print('hk',end=' ')
    dataset3= []
    for i in range(len(path3)):
        dataset3.append(np.load(path3[i]).astype(float))
    dataset_HK = np.asarray(dataset3)
    np.save('./HK_flow',dataset_HK)
def getdata(source_city,target_city):
    source_path='./{}_flow.npy'.format(source_city)
    target_path='./{}_flow.npy'.format(target_city)
    dataset1=np.load(source_path) #(25, 15, 15, 24)
    dataset2=np.load(target_path) #(31, 15, 15, 24)

    dataset1=dataset1.transpose((0,3,1,2))  #(25, 24,15, 15)
    dataset2=dataset2.transpose((0,3,1,2))  #(31,24,15,15)
    dataset1=dataset1.reshape(-1,225)
    dataset2=dataset2.reshape(-1,225)
    dataset1 = np.swapaxes(dataset1, 0,1)
    dataset2 = np.swapaxes(dataset2, 0,1)#(225,-1)
    return dataset1,dataset2
#没用上
def scalar(data, tranrange=(0, 1)):

    a = tranrange[0]
    b = tranrange[1]
    normed_data = np.zeros(data.shape)
    for k in range(data.shape[0]):
        brain_slice = data[k,:]
        if np.max(brain_slice) != 0:
            brain_slice = a + (b - a) * (brain_slice - min) / (max - min)  # 24
            normed_data[k,:] = brain_slice
    return normed_data

def get_cross_poi_corr(source_city,target_city):
    path_source='./data/{}_POI_count.csv'.format(source_city)
    path_target='./data/{}_POI_count.csv'.format(target_city)
    df_source = pd.read_csv(path_source, engine='python', index_col=0, header=0) #source
    df_target = pd.read_csv(path_target, engine='python', index_col=0, header=0)
    poi_source=np.asarray(df_source)  #size:225,17
    poi_target=np.asarray(df_target)  #size:225,17
    corr=get_corr_max(poi_source,poi_target,method='Mutual_info')
    attention =[]  #### 经过poi选择的区域 size: []*225
    for target_idx in range(corr.shape[1]):
        mean_corr=np.mean(corr[:,target_idx])
        print(mean_corr)
        candidate=[]
        for i in range(corr.shape[0]):
            # if corr[i,target_idx]>mean_corr:
                candidate.append(i)
        attention.append(candidate)
    return attention

def get_cross_road_corr(source_city,target_city):
    path_source='./data/{}_road.npy'.format(source_city)
    path_target='./data/{}_road.npy'.format(target_city)
    road_source=np.load(path_source)  #size:(225,10)
    road_target=np.load(path_target)  #size:(225,10)
    corr=get_corr_max(road_source,road_target,method='Pearson')
    attention =[]  #### 经过road 选择的区域 size: []*225
    for target_idx in range(corr.shape[1]):
        mean_corr=np.mean(corr[:,target_idx])
        print(mean_corr)
        candidate=[]
        for i in range(corr.shape[0]):
            # if corr[i,target_idx]>mean_corr:
                candidate.append(i)
        attention.append(candidate)
    return attention
def choose_r_from_PoiandRoad(source_city,target_city):
    regions=[] ##经过POI and Road筛选的区域
    poi_regions=np.asarray(get_cross_poi_corr(source_city,target_city),dtype=object)
    road_regions=np.asarray(get_cross_road_corr(source_city,target_city),dtype=object)
    for tidx in range(poi_regions.shape[0]):
        candidates1=poi_regions[tidx]
        candidates2=road_regions[tidx]
        lis_=[x for x in candidates1 if x in candidates2]
        regions.append(lis_)
    return regions
###########################
# def flow_corr_matrix(source_city,target_city):
#     source, target = getdata(source_city,target_city)  # (225,-1)  dataset1:SH dataset2:target
#     source = source[:, :24 * 7]
#     target = target[:, :24 * 7]  ##(225,24*17)
#     corr,attention=get_corr_max(source,target)
#     return corr,attention
def get_corr_max(source_data,target_data,method='Pearson'):
    a=source_data
    b=target_data
    a_shape=source_data.shape[0]
    b_shape=target_data.shape[0]
    if method=='Pearson':
        corr = np.corrcoef(a, b)[:a_shape, b_shape:]##size:(225,225)
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                if corr[i, j] != corr[i, j] or corr[i,j]>1:
                    corr[i, j] = 0
    elif method=='Mutual_info':
        a_shape = a.shape[0]
        b_shape = b.shape[0]
        corr_list = []# 225*225
        for id_s in range(a_shape):
            corr_for_idt = []
            for id_t in range(b_shape):
                corr = sm.normalized_mutual_info_score(a[id_s], b[id_t])
                corr_for_idt.append(corr)
            corr_list.append(corr_for_idt)
        corr=np.asarray(corr_list)
    return corr

def timeperiod_corr(source_city,target_city,window):
    ##(20, 225, 2)   [region_idx,corr]
    # candi_regions=np.load('./{}-{}_pre_match_regions.npy'.format(source_city, target_city))
    candi_regions=choose_r_from_PoiandRoad(source_city,target_city)
    source, target = getdata(source_city,target_city)
    source=source[:,:7*24+6]
    target=target[:,:7*24+6] ##seq_len=6,额外需要5h

    ##一天为0-23+第二天0-4，共29小时
    source_times=[]#(7*24, 225, 6)
    target_times=[]#(7*24, 225, 6)
    start=0
    end=window+1#窗口—+预测
    while end<=7*24+6:
        source_times.append(source[:,start:end])
        target_times.append(target[:,start:end])
        start+=1
        end+=1
    source_times=np.asarray(source_times)
    target_times=np.asarray(target_times)
    print(source_times.shape, target_times.shape)
    source_times=source_times.reshape(7,24,225,window+1)  #(7*24, 225, 6)
    target_times=target_times.reshape(7,24,225,window+1)
    source_times=source_times.transpose(2,0,3,1).reshape(225,7*(window+1),24) #(225,7*6,24)  每个区域24个时段，每个时段6小时，7天
    target_times=target_times.transpose(2,0,3,1).reshape(225,7*(window+1),24) #(225,7*6,24)  每个区域24个时段，每个时段6小时，7天
    print(source_times.shape,target_times.shape)
    total_corr=[]##size:(225,24)

    for time_idx in range(24):
        a=source_times[:,:,time_idx]  #(225,7*6)
        b=target_times[:,:,time_idx]  #(225,7*6)
        corr=get_corr_max(a,b)
        attention=list(np.empty((225,2))) #size:(225,2)
        for target_idx in range(corr.shape[1]):
            ## get max
            max_regions = np.argsort(-np.abs(corr[:, target_idx]))#比较绝对值
            regions_lis=candi_regions[target_idx] ## pre-match
            flag=1
            for max_r in max_regions:
                if max_r in regions_lis :
                    attention[target_idx] = [max_r, corr[max_r,target_idx]]
                    flag=0
                    break
            if flag==1:
                attention[target_idx] = [max_regions[0], corr[max_regions[0],target_idx]]
        total_corr.append(attention)
    total_corr=np.asarray(total_corr)  ###size=(24,225,2)
    print(total_corr.shape)
    np.save('./{}-{}_filter_absP7.npy'.format(source_city,target_city),total_corr)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    np.seterr(divide='ignore',invalid='ignore')
    get_total_data()
    for source_city, target_city in [['SH', 'NJ'], ['SH', 'HK'], ['HK', 'NJ'], ['NJ', 'HK'],['NJ', 'SH'], ['HK', 'SH']]:
         timeperiod_corr(source_city, target_city,window=6)




