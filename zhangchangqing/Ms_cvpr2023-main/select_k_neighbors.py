import numpy as np
from util import mv_dataset, read_mymat, build_ad_dataset, process_data, mv_tabular_collate
from scipy.spatial.distance import cdist


reg_param  = 1e-3


def get_samples(x, y, sn, train_index, test_index, dist_path, n_sample, k,if_mean = False, ):
    # 视图数 训练集数 测试集数 总样本数
    view_num = len(x)
    data_num = x[0].shape[0]

    print("计算距离")
    dist_all_set = [cdist(x[i], x[i], 'euclidean') for i in range(view_num)]

    # view_num * view_num * () 记录同时存在两个视图的样本索引
    # 并去除测试集索引
    dismiss_view_index = [[np.array([]) for __ in range(view_num)] for _ in range(view_num)]
    for i in range(view_num-1):
        for j in range(i+1, view_num):
            # (data_num, 2,)视图 i 和 j 的 sn
            sn_temp = sn[:,[i,j]]
            # (data_num,)   视图 i 和 j 的 sn 的行之和
            sn_temp_sum = np.sum(sn_temp, axis=1)
            # (data_num,)   将test部分赋值为 0 (不是2即可,排除在筛选范围之外)
            sn_temp_sum[test_index] = 0
            # (data_num,)   视图 i 和 j 的 sn 的行之和为 2 视图 i 和 j 的都存在的样本索引
            dismiss_view_index[i][j] = dismiss_view_index[j][i] = np.where(sn_temp_sum == 2)[0]

    print("训练集填充")
    # 训练集填充缺失：利用每个视图同一类下的所有存在样本，获得其均值方差，并采样
    # step1 获得训练集
    sn_train = sn[train_index]
    x_train = [x[v][train_index] for v in range(view_num)]
    y_train = y[train_index]

    # step2 获取每个视图上每类的采样点
    class_num = np.max(y) + 1
    means, covs, num = dict(), dict(),dict()
    for v in range(view_num):
        present_index = np.where(sn_train[:, v] == 1)[0]
        means_v,  covs_v, num_v = [], [], []
        for c in range(class_num):
            present_index_class = np.where(y_train[present_index] == c)[0]
            means_v.append(np.mean(x_train[v][present_index_class], axis=0))
            covs_v.append(np.cov(x_train[v][present_index_class], rowvar=0))
            num_v.append(present_index_class.shape[0])
        means[v], covs[v], num[v] = means_v, covs_v, num_v

    # step3 筛选完整样本
    x_train_dissmiss_index = np.where(np.sum(sn_train, axis=1) == view_num)[0]
    x_complete = [x_train[_][x_train_dissmiss_index] for _ in range(view_num)]
    y_complete = y_train[x_train_dissmiss_index]
    sn_complete = sn_train[x_train_dissmiss_index]

    # step4 填补不完整样本
    x_train_miss_index = np.where(np.sum(sn_train, axis=1) != view_num)[0]
    x_incomplete = [np.repeat(x_train[_][x_train_miss_index], n_sample ,axis=0) for _ in range(view_num)]
    y_incomplete = np.repeat(y_train[x_train_miss_index], n_sample, axis=0)
    sn_incomplete = np.repeat(sn_train[x_train_miss_index], n_sample, axis=0)
    index = 0
    for i in x_train_miss_index:
        y_i = y_train[i][0]
        miss_view_index = np.nonzero(sn_train[i] == 0)[0]
        for v in miss_view_index:
            rng = np.random.default_rng()
            cov = covs[v][y_i] + np.eye(len(covs[v][y_i])) * reg_param  # add regularization parameter to ensure non-singularity
            L = np.linalg.cholesky(cov)
            samples_v = rng.normal(size=(n_sample, len(cov))) @ L.T + means[v][y_i]

            x_incomplete[v][index*n_sample:(index+1)*n_sample] = samples_v
        index+=1
    x_train = [np.concatenate((x_complete[_], x_incomplete[_]), axis=0) for _ in range(view_num)]
    x_train = process_data(x_train, view_num)
    y_train = np.concatenate((y_complete, y_incomplete), axis=0)
    Sn_train = np.concatenate((sn_complete, sn_incomplete), axis=0)


    print("测试集填充")
    # 测试集填充缺失：依靠存在的视图获取邻居，从邻居的分布中采样
    sn_test = sn[test_index]
    x_test_dissmiss_index = np.where(np.sum(sn_test, axis=1) == view_num)[0] # 测试集中完整的样本索引
    if if_mean:
        # 不重复
        x_test = [x[_][test_index][x_test_dissmiss_index] for _ in range(view_num)]
        y_test = y[test_index][x_test_dissmiss_index]
    else:
        # 重复
        x_test = [np.repeat(x[_][test_index][x_test_dissmiss_index], n_sample, axis=0) for _ in range(view_num)]
        y_test =  np.repeat(y[test_index][x_test_dissmiss_index],    n_sample, axis=0)
    # 样本x_i 的 索引 x_index_i
    for i in test_index.flat:
        if if_mean:
            # 不重复
            # view_num * (1, dim,)
            x_i = [np.expand_dims(x[_][i], axis=0) for _ in range(view_num)]
            # (1, 1)
            y_i =  np.expand_dims(y[i],    axis=0)
        else:
            # 重复
            # view_num * (n_sample, dim,)
            x_i = [np.repeat(np.expand_dims(x[_][i], axis=0), n_sample, axis=0) for _ in range(view_num)]
            # (n_sample, 1,)
            y_i =  np.repeat(np.expand_dims(y[i],    axis=0), n_sample, axis=0)
        # 样本x_i 的 sn
        sn_temp = sn[i]
        # 样本x_i 缺失 和 不缺失 视图索引
        x_miss_view_index    = np.nonzero(sn_temp == 0)[0]
        x_dismiss_view_index = np.nonzero(sn_temp)[0]
        # 如果有缺失 样本x_i 的 缺失 视图索引
        if x_miss_view_index.shape[0] != 0:
            for j in x_miss_view_index.flat:
                # (k+,) 样本x_i 在缺失视图 x_miss_view_index_i 上的邻近索引
                neighbors_index_temp = np.array([],dtype=np.int_)
                # 样本x_i 的 不缺失 视图索引
                for jj in x_dismiss_view_index.flat:
                    # (k+,) 找到 x_train 中 x_miss_view_index_i 和 x_dismiss_view_index_j 都不缺失的样本补全 缺失 视图
                    dismiss_view_index_temp = dismiss_view_index[j][jj]
                    # dist_all_set 的 不缺失视图 x_dismiss_view_index_j 的 样本x_index_i 中 dismiss_view_index_temp 前k个索引
                    dist_temp = np.full(data_num, np.inf)
                    dist_temp[dismiss_view_index_temp] = dist_all_set[jj][i, dismiss_view_index_temp]

                    nearest_index_temp = np.argpartition(dist_temp, k)[:k]

                    neighbors_index_temp = np.unique(np.concatenate((neighbors_index_temp, nearest_index_temp),))
                # 样本x_i 在 缺失视图 x_miss_view_index_i 的邻近样本
                x_neighbors_temp = x[j][neighbors_index_temp]
                # 求近邻样本的均值 (dim,)
                mean = np.mean(x_neighbors_temp, axis=0)
                cov = np.cov(x_neighbors_temp, rowvar=0)
                # 采样 (n_sample, dim,)
                rng = np.random.default_rng()
                cov = cov + np.eye(len(cov)) * reg_param
                L = np.linalg.cholesky(cov)
                x_samples_temp = rng.normal(size=(n_sample, len(cov))) @ L.T + mean
                x_i[j] = x_samples_temp

            x_test = [np.concatenate((x_test[_], x_i[_]), axis=0) for _ in range(view_num)]
            y_test =  np.concatenate((y_test,    y_i),    axis=0)
    x_test = process_data(x_test, view_num)


    print("测试集填充成功")


    return x_train, y_train, x_test, y_test, Sn_train
