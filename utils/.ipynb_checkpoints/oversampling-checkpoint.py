from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import numpy as np

def oversampling(np_feat, n_jobs):
    """
        oversampling只对训练数据集中的样本进行过采样, 对验证集和独立测试集的样本不做处理
    """
    X, y = np_feat[:,:-1], np_feat[:, -1]
    X_resampled, y_resampled = SMOTE(sampling_strategy="minority", k_neighbors=5, random_state=2, n_jobs=n_jobs).fit_resample(X, y)
    y_resampled = y_resampled[:, np.newaxis]
    np_feat_resample = np.concatenate([X_resampled, y_resampled], axis=1)
    
    return np_feat_resample
