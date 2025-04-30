import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
import seaborn as sns

def auroc_curve(fprs:list, tprs:list, figname):
    tprs_interp = []
    aurocs = []
    mean_fpr = np.linspace(0, 1, 100)
    _, ax = plt.subplots(figsize=(6,4))
    
    # 对每一个fold计算AUROC
    for _, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aurocs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3)
        
    # 添加 Chance 线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.5)
    
    # 绘制多次测试的平均值的线
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auroc = np.mean(aurocs)
    ax.plot(mean_fpr, mean_tpr, color='black', label='Mean AUROC = %0.4f' % mean_auroc, lw=2, alpha=.5)
    ax.legend()

    # 设置 x 和 y 轴的刻度范围
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    # 设置坐标轴
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUROC of '+modelname)
    
    plt.savefig(figname)
    plt.close()

    
def auprc_curve(recalls:list, precisions:list, figname):
    recalls_interp = []
    auprcs = []
    mean_precision = np.linspace(0, 1, 100)
    _, ax = plt.subplots(figsize=(6, 4))

    # Calculate AUPRC for each fold
    for _, (precision, recall) in enumerate(zip(precisions, recalls)):
        recalls_interp.append(np.interp(mean_precision, precision, recall))
        recalls_interp[-1][0] = 1.0
        pr_auc = auc(recall, precision)
        auprcs.append(pr_auc)
        ax.plot(recall, precision, lw=1, alpha=0.3)

    # Draw the line of the average of multiple tests
    mean_recall = np.mean(recalls_interp, axis=0)
    mean_precision[-1] = 1.0
    mean_auprc = np.mean(auprcs)
    ax.plot(mean_recall, mean_precision, color='black', label='Mean AUPRC = %0.4f' % mean_auprc, lw=2, alpha=.5)
    ax.legend()

    # Set the scale range for the x and y axes
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUPRC of '+modelname)

    plt.savefig(figname)
    plt.close()

