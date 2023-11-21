'''
Descripttion: 
version: 
Author: huangqianfei
Date: 2022-06-18 10:02:33
LastEditTime: 2023-06-07 22:01:24
'''
from select import select
import numpy as np
from scipy.cluster.vq import kmeans2, vq


class PQ(object):
    """Pure python implementation of Product Quantization (PQ) [Jegou11]_.
    For the indexing phase of database vectors,
    a `D`-dim input vector is divided into `M` `D`/`M`-dim sub-vectors.
    Each sub-vector is quantized into a small integer via `Ks` codewords.
    For the querying phase, given a new `D`-dim query vector, the distance beween the query
    and the database PQ-codes are efficiently approximated via Asymmetric Distance.
    All vectors must be np.ndarray with np.float32
    .. [Jegou11] H. Jegou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011
    Args:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
            (typically 256, so that each sub-vector is quantized
            into 256 bits = 1 byte = uint8)
        verbose (bool): Verbose flag
    Attributes:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
        verbose (bool): Verbose flag
        code_dtype (object): dtype of PQ-code. Either np.uint{8, 16, 32}
        codewords (np.ndarray): shape=(M, Ks, Ds) with dtype=np.float32.
            codewords[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
        Ds (int): The dim of each sub-vector, i.e., Ds=D/M
    """

    def __init__(self, M, Ks=256, verbose=True):
        assert 0 < Ks <= 2 ** 32
        self.M, self.Ks, self.verbose = M, Ks, verbose
        self.code_dtype = (
            np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        )
        self.codewords = None
        self.Ds = None

        if verbose:
            print("M: {}, Ks: {}, code_dtype: {}".format(M, Ks, self.code_dtype))

    def __eq__(self, other):
        if isinstance(other, PQ):
            return (self.M, self.Ks, self.verbose, self.code_dtype, self.Ds) == (
                other.M,
                other.Ks,
                other.verbose,
                other.code_dtype,
                other.Ds,
            ) and np.array_equal(self.codewords, other.codewords)
        else:
            return False

    def fit(self, vecs, iter=20, seed=123):
        """Given training vectors, run k-means for each sub-space and create
        codewords for each sub-space.
        This function should be run once first of all.
        Args:
            vecs (np.ndarray): Training vectors with shape=(N, D) and dtype=np.float32.
            iter (int): The number of iteration for k-means
            seed (int): The seed for random process
        Returns:
            object: self
        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        assert D % self.M == 0, "input dimension must be dividable by M"
        self.Ds = int(D / self.M)

        np.random.seed(seed)
        if self.verbose:
            print("iter: {}, seed: {}".format(iter, seed))

        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        self.codewords = np.zeros((self.M, self.Ks, self.Ds), dtype=np.float32)
        # (8, 256, 16)
        # vec[2000, 256]
        # 这里的fit是将【2000，256】进行聚类，
        # 聚类的结果是8 * 【256，16】
        for m in range(self.M):
            if self.verbose:
                print("Training the subspace: {} / {}".format(m, self.M))
            vecs_sub = vecs[:, m * self.Ds : (m + 1) * self.Ds]
            self.codewords[m], _ = kmeans2(vecs_sub, self.Ks, iter=iter, minit="points")

        return self

    def encode(self, vecs):
        """Encode input vectors into PQ-codes.
        Args:
            vecs (np.ndarray): Input vectors with shape=(N, D) and dtype=np.float32.
        Returns:
            np.ndarray: PQ codes with shape=(N, M) and dtype=self.code_dtype
        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert D == self.Ds * self.M, "input dimension must be Ds * M"

        # codes[n][m] : code of n-th vec, m-th subspace

        codes = np.empty((N, self.M), dtype=self.code_dtype)
        for m in range(self.M):
            if self.verbose:
                print("Encoding the subspace: {} / {}".format(m, self.M))
            vecs_sub = vecs[:, m * self.Ds : (m + 1) * self.Ds]
            codes[:, m], _ = vq(vecs_sub, self.codewords[m])


        return codes

    def decode(self, codes):
        """Given PQ-codes, reconstruct original D-dimensional vectors
        approximately by fetching the codewords.
        Args:
            codes (np.ndarray): PQ-cdoes with shape=(N, M) and dtype=self.code_dtype.
                Each row is a PQ-code
        Returns:
            np.ndarray: Reconstructed vectors with shape=(N, D) and dtype=np.float32
        """
        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.M
        assert codes.dtype == self.code_dtype

        vecs = np.empty((N, self.Ds * self.M), dtype=np.float32)
        for m in range(self.M):
            vecs[:, m * self.Ds : (m + 1) * self.Ds] = self.codewords[m][codes[:, m], :]

        return vecs

    def dtable(self, query):
        """Compute a distance table for a query vector.
        The distances are computed by comparing each sub-vector of the query
        to the codewords for each sub-subspace.
        `dtable[m][ks]` contains the squared Euclidean distance between
        the `m`-th sub-vector of the query and the `ks`-th codeword
        for the `m`-th sub-space (`self.codewords[m][ks]`).
        Args:
            query (np.ndarray): Input vector with shape=(D, ) and dtype=np.float32
        Returns:
            nanopq.DistanceTable:
                Distance table. which contains
                dtable with shape=(M, Ks) and dtype=np.float32
        """
        assert query.dtype == np.float32
        assert query.ndim == 1, "input must be a single vector"
        (D,) = query.shape
        assert D == self.Ds * self.M, "input dimension must be Ds * M"

        # dtable[m] : distance between m-th subvec and m-th codewords (m-th subspace)
        # dtable[m][ks] : distance between m-th subvec and ks-th codeword of m-th codewords
        dtable = np.empty((self.M, self.Ks), dtype=np.float32)
        for m in range(self.M):
            query_sub = query[m * self.Ds : (m + 1) * self.Ds]
            dtable[m, :] = np.linalg.norm(self.codewords[m] - query_sub, axis=1) ** 2
        print(np.shape(dtable))
        return DistanceTable(dtable)


class DistanceTable(object):
    """Distance table from query to codeworkds.
    Given a query vector, a PQ/OPQ instance compute this DistanceTable class
    using :func:`PQ.dtable` or :func:`OPQ.dtable`.
    The Asymmetric Distance from query to each database codes can be computed
    by :func:`DistanceTable.adist`.
    Args:
        dtable (np.ndarray): Distance table with shape=(M, Ks) and dtype=np.float32
            computed by :func:`PQ.dtable` or :func:`OPQ.dtable`
    Attributes:
        dtable (np.ndarray): Distance table with shape=(M, Ks) and dtype=np.float32.
            Note that dtable[m][ks] contains the squared Euclidean distance between
            (1) m-th sub-vector of query and (2) ks-th codeword for m-th subspace.
    """

    def __init__(self, dtable):
        assert dtable.ndim == 2
        assert dtable.dtype == np.float32
        self.dtable = dtable

    def adist(self, codes):
        """Given PQ-codes, compute Asymmetric Distances between the query (self.dtable)
        and the PQ-codes.
        Args:
            codes (np.ndarray): PQ codes with shape=(N, M) and
                dtype=pq.code_dtype where pq is a pq instance that creates the codes
        Returns:
            np.ndarray: Asymmetric Distances with shape=(N, ) and dtype=np.float32
        """

        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.dtable.shape[0]

        # Fetch distance values using codes. The following codes are
        # dtable[8,256]
        # codes[10000,8]
        print("dist----")
        print(np.shape(self.dtable))
        print(np.shape(self.dtable[range(M), codes]))
        dists = np.sum(self.dtable[range(M), codes], axis=1)
        print(np.shape(dists))

        # The above line is equivalent to the followings:
        # dists = np.zeros((N, )).astype(np.float32)
        # for n in range(N):
        #     for m in range(M):
        #         dists[n] += self.dtable[m][codes[n][m]]

        return dists

if __name__ == '__main__':

    N, Nt, D = 10000, 2000, 128
    # 10,000 128-dim vectors to be indexed
    X = np.random.random((N, D)).astype(np.float32)  
    # 2,000 128-dim vectors for training
    Xt = np.random.random((Nt, D)).astype(np.float32)  
    # query vector[128]
    query = np.random.random((D,)).astype(np.float32)

    pq = PQ(M = 8)

    # train:Xt:[2000,128]->codewords[8.256,16]
    # 首先将128d切分成16*8，然后将2000行聚类成256，就形成了[8,256,16]
    pq.fit(Xt)

    # add:Encode to PQ-codes
    # [10000,128]->[10000,16]*8 ->[10000,8](将16变成和256类中最相似的序号，256类别在fit形成)
    X_code = pq.encode(X)

    # search: create a distance table online, and compute Asymmetric Distance to each PQ-code
    # [128]->[8*16]和[8,256,16]求norm->[8,256]
    # 这里的X_code 是编码后的候选集，大小 1w * 8
    # 而query计算split成 8 * 16 之后，需要先计算和聚类中心的距离（这也是主要计算量），得到【8 * 256】的距离矩阵
    # 将候选集合的1w * 8的编号通过查表（[8 * 256]的距离矩阵）得到1w * 8的距离向量，将8列进行sum得到query距离1w样本的距离，取topk
    dists = pq.dtable(query).adist(X_code)

    # 分析假如没有pq，计算一个128d向量和  10w的候选集合之间的最近的topk
    # 需要计算10w 次 两个128d向量之间的内积，取topk

    # 通过pq计算的话假设聚类256类， 切分 8个space
    # 需要计算 256次 两个16d向量之间的内积，计算8对，然后查表，取topk

    # 离线
    # train:切分子space并进行聚类得到聚类向量【8，256，16】
    # add:将候选集合进行编码得到【1w， 8】
    # 在线
    # search:前面两步都是离线计算的，来一条query，切分，计算和各个聚类中心的距离得到【8 * 256】距离矩阵
    # 将2中的编码在3中得到的距离矩阵中进行查表得到【1w， 8】，然后求和得到【1w】是query和这1w候选集的距离，取topk。
    # https://mp.weixin.qq.com/s/5KkDjCJ_AoC0w7yh2WcOpg

    # ivfpq 针对亿级别数据
    # PQ乘积量化计算距离的时候，距离虽然已经预先算好了，但是对于每个样本到查询样本的距离，还是得老老实实挨个去求和相加计算距离
    # 筛选出关注的区域就可以减少查表sum的计算
    # 1，先将数据进行粗聚类
    # 2，计算候选集和各自聚类中心的残差
    # 3，对残差数据进行编码
    # --------下面为在线计算---------
    # 4，来一条query，判断它和各个聚类中心的距离，取topk个聚类中心
    # 5，计算k个聚类中心下的候选集和query的距离作为最终的距离，再取topk


