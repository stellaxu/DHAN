import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',type=str,help='read input text log')
    parser.add_argument('--input_file_din',type=str,help='read din text log')
    args = parser.parse_args()
    return args

cutoff = 300





if __name__ == "__main__":
    args = read_args()
    epoch = []
    global_step = []
    train_loss = []
    eval_gauc = []
    eval_auc = []
    test_loss = []
    
    global_step_din=[]
    train_loss_din=[]
    eval_gauc_din=[]
    eval_auc_din=[]
    test_loss_din=[]

    with open(args.input_file,'r') as f:
        for line in f:
            if 'BATCH_NOR_Epoch' in line and 'Global_step' in line and 'Train_loss' in line and 'Eval_GAUC' in line and 'Eval_AUC' in line and 'Test_loss' in line:
                #standard way is to use RE...this for convenience only
                data = line.split('\t')
                for single_data in data:
                    raw_data = single_data.split(' ')

                    if('BATCH_NOR_Epoch' in single_data and 'Global_step' in single_data):
                        epoch.append(int(raw_data[1]))
                        global_step.append(int(raw_data[3]))
                    elif 'Train_loss' in single_data:
                        train_loss.append(float(raw_data[1]))
                    elif 'Eval_GAUC' in single_data:
                        eval_gauc.append(float(raw_data[1]))
                    elif 'Eval_AUC' in single_data:
                        eval_auc.append(float(raw_data[1]))
                    elif 'Test_loss' in single_data:
                        test_loss.append(float(raw_data[1]))
    
    with open(args.input_file_din,'r') as f:
        for line in f:
            if 'Global_step' in line and 'Train_loss' in line and 'Eval_GAUC' in line and 'Eval_AUC' in line and 'Test_loss' in line:
                #standard way is to use RE...this for convenience only
                data = line.split('\t')
                for single_data in data:
                    raw_data = single_data.split(' ')

                    if('Global_step' in single_data):
                        epoch.append(int(raw_data[1]))
                        global_step_din.append(int(raw_data[3]))
                    elif 'Train_loss' in single_data:
                        train_loss_din.append(float(raw_data[1]))
                    elif 'Eval_GAUC' in single_data:
                        eval_gauc_din.append(float(raw_data[1]))
                    elif 'Eval_AUC' in single_data:
                        eval_auc_din.append(float(raw_data[1]))
                    elif 'Test_loss' in single_data:
                        test_loss_din.append(float(raw_data[1]))
    
                
    global_step = global_step[:cutoff]
    train_loss = train_loss[:cutoff]
    eval_auc = eval_auc[:cutoff]
    eval_gauc = eval_gauc[:cutoff]
    test_loss = test_loss[:cutoff]
    
    global_step_din = global_step_din[:cutoff]
    train_loss_din = train_loss_din[:cutoff]
    eval_auc_din = eval_auc_din[:cutoff]
    eval_gauc_din = eval_gauc_din[:cutoff]
    test_loss_din = test_loss_din[:cutoff]

    #remove outliers in train_loss
    remove_axis= []
    print(len(train_loss))
    print(train_loss)
    for i in range(1,len(train_loss)):
        if((train_loss[i-1]-train_loss[i])/train_loss[i-1]) > 0.1:
            remove_axis.append(i)
    print(remove_axis)
    for index in sorted(remove_axis, reverse=True):
        del global_step[index]
        del train_loss[index]
        del eval_auc[index]
        del eval_gauc[index]
        del test_loss[index]
        
        del global_step_din[index]
        del train_loss_din[index]
        del eval_auc_din[index]
        del eval_gauc_din[index]
        del test_loss_din[index]
    print(len(train_loss))


    fig,ax = plt.subplots(1,1)
    ax.plot(global_step,train_loss,label='DHAN')
    ax.plot(global_step_din,train_loss_din,label='DIN')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=50000))
    plt.tick_params(labelsize=12)
    plt.xlabel('global step',fontsize=12,fontweight='bold')
    plt.ylabel('train loss',fontsize=12,fontweight='bold')
    plt.legend(loc='upper right')
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.1))
    plt.savefig('train_loss.png')
    fig,ax = plt.subplots(1,1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=50000))
    ax.plot(global_step,test_loss,label='DHAN')
    ax.plot(global_step_din,test_loss_din,label='DIN')

    plt.tick_params(labelsize=12)
    plt.xlabel('global step',fontsize=12,fontweight='bold')
    plt.ylabel('test loss',fontsize=12,fontweight='bold')
    plt.legend(loc='upper right')

    plt.savefig('test_loss.png')
    fig,ax = plt.subplots(1,1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=50000))
    ax.plot(global_step,eval_gauc,label='DHAN')
    ax.plot(global_step_din,eval_gauc_din,label='DIN')

    plt.tick_params(labelsize=12)
    plt.xlabel('global step',fontsize=12,fontweight='bold')
    plt.ylabel('eval gauc',fontsize=12,fontweight='bold')
    plt.legend(loc='upper right')

    plt.savefig('eval_gauc.png')
    fig,ax = plt.subplots(1,1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=50000))
    ax.plot(global_step,eval_auc,label='DHAN')
    ax.plot(global_step_din,eval_auc_din,label='DIN')
    plt.tick_params(labelsize=12)
    plt.xlabel('global step',fontsize=12,fontweight='bold')
    plt.ylabel('eval auc',fontsize=12,fontweight='bold')
    plt.legend(loc='upper right')

    plt.savefig('eval_auc.png')

    pass