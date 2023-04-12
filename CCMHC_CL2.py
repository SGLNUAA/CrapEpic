import torch.nn as nn
import torch
import numpy as np

from load_data import load_data_1h

class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
    """
    def __init__(self, shape, input_chans, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H,W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        # self.batch_size=batch_size
        self.padding = int((filter_size - 1) / 2)  # in this way the output has the same size

        # 4的意思是生成四个张量，i、f、o、g
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)

    def forward(self, input, hidden_state):
        '''
        :param input:(B,C,H,W)
        :param hidden_state: (B,C,H,W)
        :return:
        '''
        hidden, c = hidden_state  # hidden and c are images with several channels

        combined = torch.cat((input, hidden), dim=1)  # 张量连接，按照通道的维度

        A = self.conv(combined)  # (batchm,c*4,h,w)
        # print('A:',A.shape)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)  # it should return 4 tensors
        # (batchm,c,h,w)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)
        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c
    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1], requires_grad=True).cpu(),
                torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1], requires_grad=True).cpu())
class CLSTM(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, shape, input_chans, filter_size, num_features, num_layers):
        super(CLSTM, self).__init__()

        self.shape = shape  # H,W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.num_layers = num_layers

        cell_list = []
        cell_list.append(
            CLSTM_cell(self.shape, self.input_chans, self.filter_size, self.num_features).cpu())  # the first
        # one has a different number of input channels

        for idcell in range(1, self.num_layers):
            cell_list.append(CLSTM_cell(self.shape, self.num_features, self.filter_size, self.num_features).cpu())
        self.cell_list = nn.ModuleList(cell_list)
        self.lastconv = nn.Sequential(
            nn.Conv2d(in_channels=self.num_features, out_channels=1, kernel_size=filter_size, stride=1, padding=1),
            nn.ReLU(True)
        )

    def forward(self, input,batch_size):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len,Batch,Chans,H,W
        """
        hidden_state = self.init_hidden(batch_size)
        current_input = input.transpose(0, 1)  # now is seq_len,B,C,H,W
        # current_input=input
        next_hidden = []  # hidden states(h and c)
        seq_len = current_input.size(0)

        for idlayer in range(self.num_layers):  # loop for every layer

            hidden_c = hidden_state[idlayer]  # hidden and c are images with several channels，每一层的h,c初始化
            all_output = []
            output_inner = []  # 存放每层的所有时序的输出h
            for t in range(seq_len):  # loop for every step
                # 计算当前t的 h和c，返回（h，c）
                hidden_c = self.cell_list[idlayer](current_input[t, ...],
                                                   hidden_c)  # cell_list is a list with different conv_lstms 1 for every layer

                output_inner.append(hidden_c[0])

            next_hidden.append(hidden_c)
            # 将一层内的输出按照0维度拼接
            current_input = torch.cat(output_inner, 0).view(current_input.size(0),
                                                            *output_inner[0].size())  # seq_len,B,chans,H,W
        # 返回的current_input是最终的t序列输出，next_hidden是每一层的最后时序的输出
        out=next_hidden[-1][0]#最后一层的输出h,对其卷积  (batchsize,numfeature,15,15)
        out=self.lastconv(out)
        return next_hidden, current_input,out

    def init_hidden(self, batch_size):
        init_states = []  # this is a list of tuples（h,c）
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states
def adjust_learning_rate(optimizer,epoch, lr):
    lr *= (0.1 ** (epoch // 10) )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def transfer_loss(region1,region2,attention,time_slot):
    '''
    :param region1: （6,4,6,15,15)
    :param region2:
    :param attention:  流量分时段相关性(24,225,2)  [regionidx,corr]
    time_slot:  size:6;  0-23 确定属于哪一个时间段  [0,1,2,3,4,5]-[18,19,20,21,22,23]
    :return:
    '''
    region_loss=[]
    region1=region1.reshape(batch_size,-1,225)  ##(6,feature,region_idx) (6,24,225)
    region2=region2.reshape(batch_size,-1,225)  #6 8 225
    # which_slot=[0]*7+[1]*3+[2]*4+[3]*5+[4]*5 ##时段定位5
    which_slot=list(range(24))# 24
    for tidx in range(attention.shape[1]): ###每个目标区域进行迁移
        if target_train_para[tidx, 1] == 0.0:continue     #####空区域
        batch_loss=[]
        for i in range(time_slot.shape[0]):
            #每个batch的时段
            slot_idx=which_slot[time_slot[i]]
            match_ridx = int(attention[slot_idx, tidx, 0])
            match_corr = attention[slot_idx, tidx, 1]
            s_rep = region1[i,:, match_ridx]
            t_rep = region2[i,:,tidx]


            # s_rep=torch.log(1+torch.exp(2*s_rep))
            # t_rep = torch.log(1 + torch.exp(2*t_rep))
            if np.abs(match_corr) < 0.1:
                continue
            elif np.abs(match_corr) < 0.2:
                # print(slot_idx)
                batch_loss.append(2 * np.abs(match_corr) * torch.mean(torch.pow(s_rep - t_rep, 2)))  # RMSE
            elif np.abs(match_corr) < 0.4:
                batch_loss.append(3 * np.abs(match_corr) * torch.mean(torch.pow(s_rep - t_rep, 2)))  # RMSE
            # elif np.abs(match_corr)<0.8:
            #      batch_loss.append(4*np.abs(match_corr)*torch.mean(torch.pow(s_rep - t_rep,2)))  # RMSE
            else:
                batch_loss.append(5 * np.abs(match_corr) * torch.mean(torch.pow(s_rep - t_rep, 2)))  # RMSE
        if batch_loss:
            region_loss.append(sum(batch_loss)/len(batch_loss))
    match_loss=sum(region_loss)
    return match_loss
def get_pre_loss(pre,label):
    '''
    :param pre:  (batch_size,15,15)
    :param label:  (batch_size,15,15)
    :return: loss
    '''
    sum_loss=[]
    for batch in range(label.shape[0]):
        b=pre[batch]
        c=label[batch]
        loss=torch.sum(  torch.pow(b-c,2)  )
        sum_loss.append(loss)
    loss=sum(sum_loss)/len(sum_loss)
    return loss

def train(XS,XT,YT,XS_neg, XT_neg, YT_neg,iter_num,source_city,target_city,w1,w2):
    #################模型加载,优化器加载
    G = CLSTM(shape, inp_chans, filter_size, num_features, nlayers)
    G_source = CLSTM(shape, inp_chans, filter_size, num_features, nlayers)
    G.load_state_dict(torch.load( './model/pre_trained/{}_CL_{}.pkl'.format(source_city,"pos")))
    G_source.load_state_dict(torch.load( './model/pre_trained/{}_CL_{}.pkl'.format(source_city,"pos")))
    # G_neg = CLSTM(shape, inp_chans, filter_size, num_features, nlayers)
    # G_source_neg = CLSTM(shape, inp_chans, filter_size, num_features, nlayers)
    # G_neg.load_state_dict(torch.load('./model/pre_trained/{}_CL_{}.pkl'.format(source_city, "neg")))
    # G_source_neg.load_state_dict(torch.load('./model/pre_trained/{}_CL_{}.pkl'.format(source_city, "neg")))

    optim_param= []
    for p in G.parameters():
        p.requires_grad = True
        optim_param.append(p)
    # for p in G_neg.parameters():
    #     p.requires_grad = True
    #     optim_param.append(p)
    for p in G_source.parameters():
        p.requires_grad = False
    # for p in G_source_neg.parameters():
    #     p.requires_grad = False
    G_optimizer = torch.optim.Adam(optim_param, lr=0.01)
    ############开始迭代训练
    #attention=np.load('./{}-{}_filterP.npy'.format(source_city,target_city)) #size=(time_slot,225,2)
    attention=np.load('./{}-{}_filter_absP7.npy'.format(source_city,target_city)) #size=(time_slot,225,2)
    train_loss_pre=[]
    for epoch in range(iter_num):
        adjust_learning_rate(G_optimizer, epoch, 0.001)
        batch_train_pre=[]
        batch_match=[]
        for batch in range(XS.shape[0]):  # batch数量，都是5天的，1h跨度，seq=6 XS.shape[0]
            batch_XS = XS[batch]  # (batch_size,6,1,15,15)
            batch_XT = XT[batch]  # (batch_size,6,1,15,15)
            batch_YT = YT[batch]  # (batch_size,15,15)
            batch_XS_neg = XS_neg[batch]  # (batch_size,6,1,15,15)
            batch_XT_neg = XT_neg[batch]  # (batch_size,6,1,15,15)
            batch_YT_neg = YT_neg[batch]  # (batch_size,15,15)
            _s, rep_s, outs = G_source(batch_XS,batch_size)#reps:6 8 15 15 outs 6 1 15 15
            _t, rep_t, outt = G(batch_XT,batch_size)
            # _s, rep_s_neg, outs_neg = G_source_neg(batch_XS_neg, batch_size)  # reps:6 8 15 15 outs 6 1 15 15
            # _t, rep_t_neg, outt_neg = G_neg(batch_XT_neg, batch_size)
            rep_s = rep_s[:, -1, :, :, :]
            rep_t = rep_t[:, -1, :, :, :]
            # rep_s_neg=rep_s_neg[:,-1,:,:,:]
            # rep_t_neg=rep_t_neg[:,-1,:,:,:]
            #############优化参数
            pre_target = outt.squeeze()  # (batch_size,15,15)
            # pre_target_neg = outt_neg.squeeze()  # (batch_size,15,15)
            time_slot = np.array([0, 1, 2, 3, 4, 5]) + np.array([(batch % 24) * 6] * 6) % 24
            ####################################计算loss##########################################
            loss_match = transfer_loss(rep_s, rep_t, attention=attention, time_slot=time_slot)
            # loss_match_neg = transfer_loss(rep_s_neg, rep_t_neg, attention=attention, time_slot=time_slot)
            loss_pre = get_pre_loss(pre_target, batch_YT)
            # loss_pre_neg = get_pre_loss(pre_target_neg, batch_YT_neg)
            # loss = w1*(w2 * loss_match + (1-w2)* loss_pre)+(1-w1)*(w2 * loss_match_neg + (1-w2)* loss_pre_neg)
            loss=w2 * loss_match + (1 - w2) * loss_pre
            ######
            # batch_match.append((w1*loss_match+(1-w1)*loss_match_neg).detach().numpy())
            # batch_train_pre.append((w1*loss_pre+(1-w1)*loss_pre_neg).detach().numpy())
            batch_match.append(loss_match.detach().numpy())
            batch_train_pre.append((loss_pre).detach().numpy())

            # 更新参数
            G_optimizer.zero_grad()
            loss.backward()
            G_optimizer.step()

        train_loss_pre.append(np.mean(batch_train_pre))
        print(epoch, 'loss_pre:', np.mean(batch_train_pre), 'loss_match', np.mean(batch_match))
    torch.save(G.state_dict(),  './model/{}-{}/CCMHC_CL_mutual_pos.pkl'.format(source_city,target_city))
    # torch.save(G_neg.state_dict(), './model/{}-{}/CCMHC_CL_mutual_neg.pkl'.format(source_city, target_city))

################################# result
def get_pre(test_X,test_X_neg,target_label,para,source_city,target_city):

    ###################################################验证模型
    model_pos = CLSTM(shape,inp_chans,filter_size,num_features,nlayers)
    # model_neg = CLSTM(shape, inp_chans, filter_size, num_features, nlayers)

    attention = np.load('./{}-{}_filter_absP7.npy'.format(source_city, target_city))
    model_pos.load_state_dict(torch.load('./model/{}-{}/CCMHC_CL_mutual_pos.pkl'.format(source_city,target_city)))
    # model_neg.load_state_dict(torch.load('./model/{}-{}/CCMHC_CL_mutual_neg.pkl'.format(source_city, target_city)))
    for param in model_pos.parameters():
        param.requires_grad = False
    # for param in model_neg.parameters():
    #     param.requires_grad = False

    average_pre_pos = []  # (batch,batch_size，15，15)
    # average_pre_neg = []  # (batch,batch_size，15，15)
    for batch in range(test_X.shape[0]):
        batch_x = test_X[batch]  # (batch_size,6,1,15,15)
        _,rep,out = model_pos(batch_x,batch_size)  # 返回(next_hidden, current_input)
        pre_y = out.squeeze()  # (batch_size,15,15)
        average_pre_pos.append(pre_y.data)
    average_pre_pos = np.array([item.detach().numpy() for item in average_pre_pos])
    average_pre_pos = average_pre_pos.reshape(-1, 225)

    # for batch in range(test_X_neg.shape[0]):
    #     batch_x = test_X_neg[batch]  # (batch_size,6,1,15,15)
    #     _,rep,out = model_neg(batch_x,batch_size)  # 返回(next_hidden, current_input)
    #     pre_y = out.squeeze()  # (batch_size,15,15)
    #     average_pre_neg.append(pre_y.data)
    # average_pre_neg = np.array([item.detach().numpy() for item in average_pre_neg])
    # average_pre_neg = average_pre_neg.reshape(-1, 225)

    count=0
    ###将空区域赋0，预测值为负赋0

    for re_id in range(225):
        if para[re_id][1] == 0:
            count += 1
            average_pre_pos[:, re_id] = 0
            # average_pre_neg[:, re_id] = 0
    average_pre=average_pre_pos.copy()
    ###对称还原
    # for net_idx in range(225):
    #     for time_idx in range(average_pre.shape[0]):
    #         if (attention[time_idx % 24, net_idx, 1] < 0):
    #             average_pre[time_idx, net_idx] = average_pre_neg[time_idx, net_idx]
    ## Denormalize
    average_pre=average_pre.clip(0,1)
    para_min=para[:,0]
    para_max=para[:,1]
    rever_pre=[]#反归一化的预测值
    for i in range(225):
        c = (average_pre[:, i] * (para_max[i] - para_min[i])) + para_min[i]
        # for idx in range(len(c)):
        #     if c[idx]<0:
        #         print("小于0：",c[idx])
        #         c[idx]=0
        rever_pre.append(c)

    rever_pre=np.asarray(rever_pre)  #(225, 330)
    rever_pre=np.swapaxes(rever_pre,0,1)#(330, 225)


    rmse = my_RMSE(rever_pre, target_label)/(225-count)
    MAE=my_MAE(rever_pre, target_label) /(225-count)
    mape=my_MAPE(rever_pre, target_label)/(225-count)
    print('rmse,mae,mape of {}-{} CCMHC_CL:'.format(source_city, target_city), rmse,MAE,mape)

    time_watch=10
    for i in range(225):
        if(rever_pre[time_watch,i]!=0 or target_label[time_watch,i]!=0):
            print(i,"\t预测结果pos:",average_pre_pos[time_watch,i],"\t预测值:",rever_pre[time_watch,i],"\t真实值:",target_label[time_watch,i],"\t最大值:",para_max[i],"\t匹配区域:",attention[time_watch,i,0],"\t相似度:",attention[time_watch,i,1])
    # np.save('./result/{}-{}_CCMHC_CL.npy'.format(source_city,target_city), rever_pre)
    #按区域
    # result_pre = []
    # count = 0
    # for i in range(225):
    #     if (para_max[i] != 0):
    #         rmse_pre = my_RMSE(rever_pre[:, [i]], target_label[:, [i]])
    #         MAE_pre = my_MAE(rever_pre[:, [i]], target_label[:, [i]])
    #         mape_pre = my_MAPE(rever_pre[:, [i]], target_label[:, [i]])
    #         count += 1
    #         result_pre.append([rmse_pre, MAE_pre, mape_pre])
    #         print("{}:\trmse:{},\tmae:{},\tmape:{}".format(i, rmse_pre, MAE_pre, mape_pre))
    # print(count)
    # result_pre = np.asarray(result_pre)
    # print(result_pre.shape)
    # np.save('./{}-{}_result_pre_mine.npy'.format(source_city, target_city), result_pre)
    #按小时
    # result_pre = np.sum(rever_pre,axis=1)
    # label_pre =np.sum(target_label,axis=1)
    # time_pre=[]
    # for i in range(result_pre.shape[0]):
    #     print("{}:\tresult:{},\tlabel:{}".format(i, result_pre[i],label_pre[i]))
    #     time_pre.append([i, result_pre[i],label_pre[i]])
    # time_pre=np.asarray(time_pre)
    # print(time_pre.shape)
    # np.save('./{}-{}_result_pre_mine_time.npy'.format(source_city, target_city),time_pre)
    return rmse,MAE,mape


def data_preprocessing_exchange(ta_tr_X, ta_tr_Y, ta_te_X,  ta_tr_X_neg, ta_tr_Y_neg, ta_te_X_neg, attention,ta_te_para):
    print("*********************start exchange*****************")
    count0 = 0
    count1 = 0
    #numpy交换必须用temp
    for net_idx in range(225):
        for time_idx in range(ta_tr_X.shape[0]):
            if(attention[time_idx%24,net_idx,1]<0):
                temp1=ta_tr_X_neg[time_idx,:,net_idx]
                ta_tr_X_neg[time_idx, :, net_idx]=ta_tr_X[time_idx,:,net_idx]
                ta_tr_X[time_idx, :, net_idx]=temp1

                # temp2=ta_tr_Y_neg[time_idx,net_idx]
                # ta_tr_Y_neg[time_idx, net_idx]=ta_tr_Y[time_idx,net_idx]
                # ta_tr_Y[time_idx,net_idx]=temp2
                count0+=1
        for time_idx in range(ta_te_X.shape[0]):
            if(attention[time_idx%24,net_idx,1]<0):
                temp=ta_te_X_neg[time_idx,:,net_idx]
                ta_te_X_neg[time_idx, :, net_idx]=ta_te_X[time_idx,:,net_idx]
                ta_te_X[time_idx,:,net_idx]=temp

    print("训练集对折{}次，未对折{}次".format(count0,225*ta_tr_X.shape[0]-count0))
    print("ta_tr_X:",ta_tr_X.shape)
    print("ta_tr_Y:",ta_tr_Y.shape)
    print("ta_te_X:",ta_te_X.shape)
    print("ta_tr_X_neg:",ta_tr_X_neg.shape)
    print("ta_tr_Y_neg:",ta_tr_Y_neg.shape)
    print("ta_te_X_neg:",ta_te_X_neg.shape)
    print("*********************finish exchange*****************")
    return ta_tr_X, ta_tr_Y, ta_te_X, ta_tr_X_neg, ta_tr_Y_neg, ta_te_X_neg


def data_preprocessing(so_tr_X,ta_tr_X,ta_tr_Y,ta_te_X,so_tr_para,ta_tr_para,ta_te_para):

    #获取对折
    so_tr_X_neg,ta_tr_X_neg,ta_tr_Y_neg,ta_te_X_neg=data_preprocessing_neg(\
        so_tr_X,ta_tr_X,ta_tr_Y,ta_te_X,so_tr_para,ta_te_para)

    #交换
    ta_tr_X=ta_tr_X.reshape(-1, seq_len,225)
    ta_tr_Y=ta_tr_Y.reshape(-1, 225)
    target_train_Y_ori=ta_tr_Y.copy()
    ta_te_X=ta_te_X.reshape(-1, seq_len, 225)
    attention = np.load('./{}-{}_filter_absP7.npy'.format(source_city, target_city))  # 24 225 2
    ta_tr_X, ta_tr_Y, ta_te_X,  ta_tr_X_neg, ta_tr_Y_neg, ta_te_X_neg = data_preprocessing_exchange( \
        ta_tr_X, ta_tr_Y, ta_te_X, ta_tr_X_neg, ta_tr_Y_neg, ta_te_X_neg, attention,ta_te_para)
    ##所有区域的固定时间
    # for i in range(225):
    #     print("{}:\tta_tr_para:{}\tta_te_para:{}".format(i,ta_tr_para[i,1],ta_te_para[i,1]))
    ##固定区域的所有时间
    # for i in range(ta_tr_Y.shape[0]):
    #     print("{}:\tta_tr_Y:{}\tta_tr_Y_neg:{}\tmin:{}\tmax:{}\t".format(i,ta_tr_Y[i,184],ta_tr_Y_neg[i,184],ta_tr_para[184,0],ta_tr_para[184,1]))
    for i in range(225):
        print("{}:\ttarget_train_Y_ori:{}\ttarget_train_Y:{}\ttarget_train_Y_neg:{},\tattention:{}".format(i,target_train_Y_ori[8, i], ta_tr_Y[8, i], ta_tr_Y_neg[8, i],attention[8,i,1]))
    #####数据处理
    so_tr_X = so_tr_X.reshape(-1, batch_size, seq_len,1,15, 15)  # (batch,  batch_size,  seq ,1  ,W,  H, )
    so_tr_X = torch.tensor(so_tr_X)
    ta_tr_X= ta_tr_X.reshape(-1, batch_size,  seq_len,1,15,15)  # (batch,  batch_size,  seq ,1  ,W,  H, )
    ta_tr_X = torch.tensor(ta_tr_X)
    ta_tr_Y = ta_tr_Y.reshape(-1, batch_size,15,15)
    ta_tr_Y = torch.tensor(ta_tr_Y)  # (batch,  batch_size, 15, 15)
    ta_te_X = ta_te_X.reshape(-1, batch_size, seq_len,1, 15,15)  # (batch,  batch_size,  seq .1  ,W,  H, )
    ta_te_X = torch.tensor(ta_te_X)

    so_tr_X_neg = so_tr_X_neg.reshape(-1, batch_size, seq_len, 1, 15, 15)  # (batch,  batch_size,  seq ,1  ,W,  H, )
    so_tr_X_neg = torch.tensor(so_tr_X_neg)
    ta_tr_X_neg = ta_tr_X_neg.reshape(-1, batch_size, seq_len, 1, 15, 15)  # (batch,  batch_size,  seq ,1  ,W,  H, )
    ta_tr_X_neg = torch.tensor(ta_tr_X_neg)
    ta_tr_Y_neg = ta_tr_Y_neg.reshape(-1, batch_size, 15, 15)
    ta_tr_Y_neg = torch.tensor(ta_tr_Y_neg)  # (batch,  batch_size, 15, 15)
    ta_te_X_neg = ta_te_X_neg.reshape(-1, batch_size, seq_len, 1, 15, 15)  # (batch,  batch_size,  seq .1  ,W,  H, )
    ta_te_X_neg = torch.tensor(ta_te_X_neg)

    return so_tr_X,ta_tr_X,ta_tr_Y,ta_te_X,so_tr_X_neg,ta_tr_X_neg,ta_tr_Y_neg,ta_te_X_neg

def data_preprocessing_neg(train_X,target_train_X,target_train_Y,target_test_X,source_train_para,target_para):
    train_X_neg = train_X.reshape(-1, seq_len, 225).copy()

    target_train_X_neg = target_train_X.reshape(-1, seq_len,  225).copy()
    target_train_Y_neg = target_train_Y.reshape(-1,  225).copy()
    target_test_X_neg = target_test_X.reshape(-1, seq_len,  225).copy()

    count_s=0
    count_t1=0
    count_t2=0

    for i in range(225):
        # if(source_train_para[i,1]==0):
        #     count_s+=1
        # else:
        # for j in range(train_X_neg.shape[0]):
            train_X_neg[:,:,i] = 2*np.mean(train_X_neg[:,:,i]) - train_X_neg[:,:,i]
            target_train_X_neg[:, :, i] =2*np.mean(train_X_neg[:,:,i]) - target_train_X_neg[:, :, i]
            # target_train_Y_neg[j, i] = np.max(target_train_X_neg[j,:,i]) - target_train_Y_neg[j, i]
        # if(target_para[i,1]==0):
        #     count_t2+=1
        # else:

        # for j in range(target_test_X_neg.shape[0]):

            target_test_X_neg[:,:,i] = 2*np.mean(train_X_neg[:,:,i]) - target_test_X_neg[:,:,i]

    print("获得neg数据:")
    print("count_s:",count_s)
    print("count_t1:", count_t1)
    print("count_t2:", count_t2)
    print("train_X_neg:", train_X_neg.shape)
    print("target_train_X_neg:", target_train_X_neg.shape)
    print("target_train_Y_neg:", target_train_Y_neg.shape)
    print("target_test_X_neg:", target_test_X_neg.shape)

    print("neg数据获得完毕。")
    return train_X_neg,target_train_X_neg,target_train_Y_neg,target_test_X_neg


def my_RMSE(data,label):
    return sum(np.sqrt(np.mean(pow(data - label, 2),axis=0)))
def my_MAE(data,label):
    return sum(np.mean(np.abs(data-label),axis=0) )
def my_MAPE(data,label):
    final_loss=np.zeros(225)
    for i in range(label.shape[1]):
        #region
        time_num=0
        gap=0
        for j in range(label.shape[0]):
        #time
            if  label[j][i]>=10:
                time_num+=1
                gap+=np.abs(data[j][i]-label[j][i])/label[j][i]
        if time_num!=0:
            final_loss[i]=gap/time_num
        else:
            final_loss[i]=0
    return sum(final_loss)




if __name__ == '__main__':

    ## parameters
    source_city='NJ'
    target_city='HK'
    inp_chans = 1
    num_features = 8
    filter_size = 3
    batch_size =6
    shape = (15, 15)
    nlayers = 2
    seq_len = 6
    train_size = 5
    ############ load data
    result=[]
    torch.set_default_tensor_type('torch.DoubleTensor')
    train_X, train_Y, test_X, test_Y, label, normed_label, source_test_para,source_train_para=load_data_1h(train_size=train_size,seq_len=seq_len,city_name=source_city)
    target_train_X, target_train_Y, target_test_X, target_test_Y, target_label, target_normed_label, target_test_para,target_train_para=load_data_1h(train_size=train_size,seq_len=seq_len,city_name=target_city)
    ### numpy to tensor
    train_X,target_train_X,target_train_Y,target_test_X,\
        train_X_neg,target_train_X_neg,target_train_Y_neg,target_test_X_neg=data_preprocessing(\
                train_X,target_train_X,target_train_Y,target_test_X,source_train_para,target_train_para,target_test_para)

    #exchange neg and pos

    # target_train_X,target_train_Y,target_test_X,target_label,target_train_X_neg,target_train_Y_neg,target_test_X_neg, target_label_neg = data_preprocessing_exchange( \
    #     target_train_X,target_train_Y,target_test_X,target_label,target_train_X_neg,target_train_Y_neg,target_test_X_neg, target_label_neg)

    print("source_train_X:", train_X.shape)  #19 6 6 1 15 15 used
    # print("source_train_Y:", train_Y.shape)  #19 6 15 15
    # print("source_test_X:", test_X.shape)  # 55 6 6 1 15 15
    # print("source_test_Y:", test_Y.shape)  # 55 6 15 15
    print("target_train_X:", target_train_X.shape)  # 19 6 6 1 15 15 used
    print("target_train_Y:", target_train_Y.shape)  # 19 6 15 15 used
    print("target_test_X:", target_test_X.shape)  # 55 6 6 1 15 15 used
    # print("target_test_Y:", target_test_Y.shape)  # 55 6 15 15

    print("target_label.shape:", target_label.shape)# 330 225 used
    print("source zero zone:",np.sum(np.asarray(source_train_para[:,1])==0))
    count=np.sum(np.asarray(target_train_para[:,1])==0)
    print("target train zero zone:",count)
    count = np.sum(np.asarray(target_test_para[:, 1]) == 0)
    print("target test zero zone:", count)
    #测试
    for w in range(1,100):

        #### train
        train(train_X, target_train_X, target_train_Y,train_X_neg, target_train_X_neg, target_train_Y_neg, 50, source_city, target_city, 0.6,w/100)
        #### test
        rmsetmp, maetmp,mapetmp = get_pre(target_test_X,target_test_X_neg,target_label, target_test_para, source_city, target_city)
        result.append([w/100,rmsetmp,maetmp,mapetmp])

    result=np.asarray(result)
    np.save('./{}-{}_result_absP6.npy'.format(source_city, target_city),result)
    result=np.load('./{}-{}_result_absP6.npy'.format(source_city, target_city))
    for i in range(result.shape[0]):
        print("w:{},rmse:{},mae:{},mape:{}".format(result[i,0],result[i,1],result[i,2],result[i,3]))
    #单个w训练

    # train(train_X, target_train_X, target_train_Y,train_X_neg, target_train_X_neg, target_train_Y_neg, 50, source_city, target_city, 0.6,0.2)
    #     # test
    # rmsetmp,mae, mapetmp = get_pre(target_test_X,target_test_X_neg,target_label, target_test_para, source_city, target_city)
