## 对特征进行降维
from sklearn.decomposition import PCA

def dimreducePCA(feat_train, feat_test, pca_ratio=0.9):
    ## feat_train: [seqembed (1x2048), hpdegree (1x1), simi (1x2), pssmembed(1x40)]
    ## pca_num_ratio: 降维时主成分占比 (0-1之间)
    pca = PCA(n_components=pca_ratio, random_state=50)
    feat_train_pca = pca.fit_transform(feat_train)
    feat_test_pca = pca.transform(feat_test)
    
    return feat_train_pca, feat_test_pca

