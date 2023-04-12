import numpy as np
import matplotlib.pyplot as plt
import case_study_correlation as get_loc
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
config = {
    "font.family":'Times New Roman',  # 设置字体类型

#     "mathtext.fontset":'stix',
}
def my_MAE(data,label):
    return sum(np.mean(np.abs(data-label),axis=0) )
def my_RMSE(data,label):
    return sum(np.sqrt(np.mean(pow(data - label, 2),axis=0)))
def region_RMSE(data,label):
    loss_region=[]
    for i in range(225):
        loss=np.sqrt(np.mean(pow(data[:,i] - label[:,i], 2)))
        loss_region.append(loss)
    return np.asarray(loss_region)
def del_zero(a,):
    a=list(a)
    for i in range(len(a)-1,-1,-1):
        # print(a[i])
        if  a[i]!=a[i]:
            a[i]=-1
    return np.asarray(a)
def case_study(source_city,target_city,model_name):
    ACPHC_pre=np.load('./result/{}-{}-{}-ACPHC_pre.npy'.format(source_city, target_city, model_name))
    if model_name=='convlstm':nn='convlstm_1h'
    elif model_name=='StepDeep':nn='Conv3d_1h'
    baseline_path = 'F:\\资料\\Data\\Experiment\\data\\{}\\normed\\{}\\'.format(nn,source_city)
    label = np.load('./result/{}_label.npy'.format(target_city))
    NFT_pre = np.load(baseline_path+'{}_pre.npy'.format(target_city))
    FT_pre = np.load(baseline_path+'{}_pre_FT.npy'.format(target_city))
    RegionTrans_pre = np.load(baseline_path+'{}_pre_YQ.npy'.format(target_city))
    Domain_pre = np.load(baseline_path+'{}_pre_cluster.npy'.format(target_city))
    TMU_multi_pre = np.load(
        'F:\\资料\\Data\\Experiment\\data\\convlstm_1h\\normed\\multi_source\\{}_pre_TMU.npy'.format(target_city))
    print(ACPHC_pre.shape,label.shape,NFT_pre.shape,FT_pre.shape,RegionTrans_pre.shape,Domain_pre.shape)

    ##空区域计算平均时不予考虑
    count_nan = np.zeros(225)#1为空区域
    for i in range(225):
        if sum(label[:, i]) == 0:
            count_nan[i] = 1
    print(count_nan)

    ############################  region_loss
    region_loss_ACPHC=region_RMSE(ACPHC_pre,label)
    region_loss_RegionTrans = region_RMSE(RegionTrans_pre, label)
    region_loss_TMU_multi = region_RMSE(TMU_multi_pre, label)
    # region_loss_FT=region_RMSE(FT_pre,label)
    # region_loss_Domain=region_RMSE(Domain_pre,label)

    ###使用ACPHC的结果排序
    # ordered_region_loss_ACPHC=np.argsort(region_loss_ACPHC)

    ####使用对比实验的损失差排序
    region_loss_improve=(region_loss_RegionTrans-region_loss_ACPHC)/region_loss_RegionTrans
    d_loss=del_zero(region_loss_improve)

    ordered_improve=np.argsort(-d_loss)
    print(ordered_improve)
    def to_percent(y):
        return str(np.round(100 * y,2)) + '%'

    ordered_regionlist=ordered_improve
    deegre=1 #第几好的区域
    for i in range(225):
        index= ordered_regionlist[i]
        if count_nan[index]==1:
            continue
        elif deegre==1:break
        else:deegre-=1
    # # # # ##预测可视化

    region_index=index
    improve = to_percent(d_loss[region_index])
    left_bottom,right_top=get_loc.get_region_location(target_city,region_index)
    print(region_index)
    plt.figure(figsize=(16, 6))
    colors=['blue','orange','brown','green','red','black']
    plt.plot(range(len(label)), label[:,region_index], color=colors[0], linewidth=1.2, label='ground truth')
    plt.plot(range(len(RegionTrans_pre)),RegionTrans_pre[:,region_index],color=colors[3],linewidth=1.2,label='RegionTrans(ConvLSTM)')
    plt.plot(range(len(ACPHC_pre)),ACPHC_pre[:,region_index],color=colors[4],linewidth=1.2,label='ACPHC(ConvLSTM)')
    plt.plot(range(len(Domain_pre)),Domain_pre[:,region_index],color=colors[5],linewidth=1.2,label='TMU(multi-source)')
    plt.legend(fontsize=30)
    plt.xlabel('hour', size=30)
    plt.ylabel('flow', size=30)
    plt.tick_params(labelsize=20)
    # plt.yticks()
    plt.title('prediction result of ShangHai(Bike)-NanJing(Bus)'.format(source_city,target_city,model_name,left_bottom,right_top,improve), size=35)
    # plt.title('源城市:{} 目标城市:{} 模型:{} 位置:{}-{} 提升百分比:{}'.format(source_city,target_city,model_name,left_bottom,right_top,improve), size=20)
    # plt.savefig('./case-study-fig/{}-{}-{}-{}'.format(source_city,target_city,model_name,region_index))
    plt.show()


if __name__ == '__main__':
    source_city='SH'
    target_city='NJ'
    model1='convlstm'
    model2='StepDeep'
    #绘制提升最大的区域的流量对比图
    for source_city,target_city in [['SH','NJ']]:
        case_study(source_city,target_city,model1)
        # case_study(source_city,target_city,model2)

