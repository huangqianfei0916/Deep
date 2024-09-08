# -*- coding: UTF-8 -*-

import os
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ["InBatchNegativeHingeLoss", "InBatchNegativeSoftmax", "InBatchHardestNegativeHingeLoss",
           "InBatchNegativeSoftmaxPoly", "GraphNegativeSoftmax"]

class InBatchNegativeHingeLoss(nn.Layer):
    """ InBatchNegativeHinge Loss for the pos and neg."""

    def __init__(self, margin, with_allgather=False, **kwargs):
        super(InBatchNegativeHingeLoss, self).__init__()
        self.margin = margin
        self.with_allgather = with_allgather

    def forward(self, src, dst):
        """ forward function

        Args:
            pos (Tensor): pos score.
            neg (Tensor): neg score.

        Returns:
            Tensor: final hinge loss.
        """
        src = F.normalize(src, axis=1)
        dst = F.normalize(dst, axis=1)

        pos = (src * dst).sum(-1, keepdim=True)
        neg = src.matmul(dst, transpose_y=True)

        if self.with_allgather and self.training:
            trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
            trainer_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
            dst_all_tensor = []
            paddle.distributed.all_gather(dst_all_tensor, dst)
            dst_all_tensor = [x for idx, x in enumerate(dst_all_tensor) if idx != trainer_id]
            dst_all_tensor = paddle.concat(dst_all_tensor, axis=0)
            dst_all_mat = src.matmul(dst_all_tensor, transpose_y=True)
            neg = paddle.concat([neg, dst_all_mat], axis=1)

        loss = paddle.mean(F.relu(neg - pos + self.margin).astype('float32'))
        return loss


class InBatchNegativeSoftmax(nn.Layer):
    """ InBatchNegativeHinge Loss for the pos and neg.
    """

    def __init__(self, with_allgather=False, **kwargs):
        super(InBatchNegativeSoftmax, self).__init__()
        self.with_allgather = with_allgather

    def forward(self, src, dst, return_logits=False):
        """ forward function

        Args:
            pos (Tensor): pos score.
            neg (Tensor): neg score.

        Returns:
            Tensor: final hinge loss.
        """
        mat = src.matmul(dst, transpose_y=True)

        if self.with_allgather and self.training:
            trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
            trainer_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
            dst_all_tensor = []
            paddle.distributed.all_gather(dst_all_tensor, dst)
            dst_all_tensor = [x for idx, x in enumerate(dst_all_tensor) if idx != trainer_id]
            dst_all_tensor = paddle.concat(dst_all_tensor, axis=0)
            dst_all_mat = src.matmul(dst_all_tensor, transpose_y=True)
            mat = paddle.concat([mat, dst_all_mat], axis=1)

        loss = paddle.mean(F.softmax_with_cross_entropy(mat,
                            paddle.unsqueeze(paddle.arange(0, mat.shape[0], dtype="int64"), axis=1)))
        if return_logits:
            return loss, mat
        else:
            return loss


class InBatchHardestNegativeHingeLoss(nn.Layer):
    """ InBatchNegativeHinge Loss for the pos and neg.
    """

    def __init__(self, margin=0.1, with_allgather=False, **kwargs):
        super(InBatchHardestNegativeHingeLoss, self).__init__()
        self.margin = margin
        self.with_allgather = with_allgather

    def forward(self, src, dst, dst_neg=None):
        """ forward function

        Args:
            pos (Tensor): pos score.
            neg (Tensor): neg score.

        Returns:
            Tensor: final hinge loss.
        """
        src = F.normalize(src, axis=1)
        dst = F.normalize(dst, axis=1)

        logit = src.matmul(dst, transpose_y=True)
        softmax_margin = paddle.eye(logit.shape[0]) * 9999
        logit = logit - softmax_margin
        hardest_neg_score = paddle.max(logit, axis=1, keepdim=True)
        pos_score = paddle.sum(src * dst, axis=1, keepdim=True)
        labels = paddle.ones([logit.shape[0], 1])
        loss1 = F.margin_ranking_loss(pos_score, hardest_neg_score, labels, margin=self.margin, reduction='mean')

        if self.with_allgather and self.training:
            trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
            trainer_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
            dst_all_tensor = []
            paddle.distributed.all_gather(dst_all_tensor, dst)
            dst_all_tensor = [x for idx, x in enumerate(dst_all_tensor) if idx != trainer_id]
            dst_all_tensor = paddle.concat(dst_all_tensor, axis=0)
            logit = src.matmul(dst_all_tensor, transpose_y=True)
            hardest_neg_score = paddle.max(logit, axis=1, keepdim=True)
            loss2 = F.margin_ranking_loss(pos_score, hardest_neg_score, labels, margin=self.margin, reduction='mean')
            return loss1 + loss2
        else:
            return loss1

    
class InBatchHardestNegativeHingeLossNEI(nn.Layer):
    """ InBatchNegativeHinge Loss for the pos and neg.
    """

    def __init__(self, margin=0.1, with_allgather=False, **kwargs):
        super(InBatchHardestNegativeHingeLossNEI, self).__init__()
        self.margin = margin
        self.with_allgather = with_allgather

    def forward(self, src, dst, dst_neg=None, labels=None):
        """ forward function

        Args:
            pos (Tensor): pos score.
            neg (Tensor): neg score.

        Returns:
            Tensor: final hinge loss.
        """
        # src batch x dim
        # dst batch x seq_len x dim
        
        bz, seq, dim = dst.shape
        
        src = F.normalize(src, axis=-1) # batch x 1 x dim
        dst = F.normalize(dst, axis=-1) # batch x seq_len x dim
        src_f = src[:, 0]
        dst_f = dst[:, 0]

        pos_score = paddle.sum(src_f * dst_f, axis=1, keepdim=True)
        logit_f = src_f.matmul(dst_f, transpose_y=True)
        softmax_margin = paddle.eye(logit_f.shape[0]) * 9999
        logit_f = logit_f - softmax_margin
        hardest_neg_index = paddle.argmax(logit_f, axis=1, keepdim=True) # batch
        
        logit = src_f.matmul(paddle.reshape(dst, shape=[bz * seq, dim]), transpose_y=True) # batch x batch * seq_len
        logit = paddle.reshape(logit, shape=[bz, bz, seq])
                           
        hardest_neg_index = paddle.concat([paddle.unsqueeze(paddle.arange(bz), axis=1),
                                          hardest_neg_index], axis=1)
        hardest_neg_score = paddle.gather_nd(logit, hardest_neg_index) # bz x seq_len
        
        loss1 = F.margin_ranking_loss(pos_score, hardest_neg_score, labels, margin=self.margin, reduction='mean')
                           
        return loss1


class InBatchNegativeSoftmaxPoly(nn.Layer):
    """ InBatchNegativeSoftmaxPoly Loss for the pos and neg.
    """

    def __init__(self, with_allgather=False, **kwargs):
        super(InBatchNegativeSoftmaxPoly, self).__init__()
        self.with_allgather = with_allgather

    def forward(self, src, dst, return_logits=False):
        """ forward function

        Args:
            pos (Tensor): pos score.
            neg (Tensor): neg score.

        Returns:
            Tensor: final hinge loss.
        """
        src = paddle.transpose(src, perm=[1, 0, 2])
        dst = paddle.transpose(dst, perm=[1, 0, 2])
        mat = src.matmul(dst, transpose_y=True)

        mat = paddle.max(mat, axis=0)

        if self.with_allgather and self.training:
            trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
            trainer_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
            dst_all_tensor = []
            paddle.distributed.all_gather(dst_all_tensor, dst)
            dst_all_tensor = [x for idx, x in enumerate(dst_all_tensor) if idx != trainer_id]
            dst_all_tensor = paddle.concat(dst_all_tensor, axis=0)
            dst_all_mat = src.matmul(dst_all_tensor, transpose_y=True)
            mat = paddle.concat([mat, dst_all_mat], axis=1)

        loss = paddle.mean(F.softmax_with_cross_entropy(mat,
                            paddle.unsqueeze(paddle.arange(0, mat.shape[0], dtype="int64"), axis=1)))

        if return_logits:
            return loss, mat
        else:
            return loss
        

class GraphNegativeSoftmax(nn.Layer):
    """ GraphNegativeSoftmax Loss for the pos and neg."""

    def __init__(self, with_allgather=False, **kwargs):
        super(GraphNegativeSoftmax, self).__init__()
        self.with_allgather = with_allgather

    def forward(self, src, dst):
        """ forward function

        Args:
            pos (Tensor): pos score.
            neg (Tensor): neg score.

        Returns:
            Tensor: final hinge loss.
        """
        if isinstance(src, tuple) or isinstance(src, list):
            src, src_repr = src
            dst, dst_repr = dst
            
            bz, seq_len_a, dim = src_repr.shape
            bz, seq_len_b, dim = dst_repr.shape
            pos_score = paddle.sum(src * dst, axis=1, keepdim=True)

            # for allgather
            if self.training and self.with_allgather:
                # pad for allgather
                # (if not pad, the program will be block when num_neighbor in different devices is not same value.)
                seq_len_max = paddle.to_tensor([seq_len_b])
                # find max num_neighbor
                paddle.distributed.all_reduce(seq_len_max, op=1)
                pad_num = seq_len_max - seq_len_b
                if pad_num != 0:
                    pad = paddle.zeros(shape=[bz, pad_num, dim], dtype=dst_repr.dtype)
                    # [batch_size, num_neighbor, dim_emb] -> [batch_size, max_num_neighbor_all_device, dim_emb]
                    dst_repr = paddle.concat(x=[dst_repr, pad], axis=1)
                seq_len_b = seq_len_max
            
            dst_repr = paddle.reshape(dst_repr, [bz * seq_len_b, dim])
            mat_b = src.matmul(dst_repr, transpose_y=True)
            mat_b = paddle.reshape(mat_b, shape=[bz, bz, seq_len_b])
            softmax_margin = paddle.unsqueeze(paddle.eye(mat_b.shape[0]) * 9999, axis=2)
            mat_b = mat_b - softmax_margin
            mat_b = paddle.reshape(mat_b, shape=[bz, bz * seq_len_b])
            mat_b = paddle.concat([pos_score, mat_b], axis=1)
            if self.training and self.with_allgather:
                trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
                trainer_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
                dst_all_tensor = []
                paddle.distributed.all_gather(dst_all_tensor, dst_repr)
                dst_all_tensor = [x for idx, x in enumerate(dst_all_tensor) if idx != trainer_id]
                dst_all_tensor = paddle.concat(dst_all_tensor, axis=0)
                mat_c = src.matmul(dst_all_tensor, transpose_y=True)
                mat_b = paddle.concat([mat_b, mat_c], axis=1)
            loss = paddle.mean(F.softmax_with_cross_entropy(mat_b, paddle.zeros([mat_b.shape[0], 1], dtype="int64")))
        else:
            mat = src.matmul(dst, transpose_y=True)
            loss = paddle.mean(F.softmax_with_cross_entropy(mat,
                                paddle.unsqueeze(paddle.arange(0, mat.shape[0], dtype="int64"), axis=1)))
        return loss
        

class CELoss(nn.Layer):
    """ InBatchNegativeSoftmaxPoly Loss for the pos and neg.
    """

    def __init__(self, with_allgather=False, **kwargs):
        super(CELoss, self).__init__()

    def forward(self, output, label):
        loss = paddle.mean(F.softmax_with_cross_entropy(output, paddle.zeros([output.shape[0], 1], dtype='int64')))
        return loss


def get_kl_loss(logits, probs):
    """kl loss"""
    kl_loss = F.softmax_with_cross_entropy(logits=logits, label=probs, soft_label=True)
    kl_loss = paddle.mean(kl_loss)
    return kl_loss


def get_sym_kl_loss(logits1, logits2):
    """get_sym_kl_loss"""
    probs1 = F.softmax(logits1)
    probs2 = F.softmax(logits2)
    kl_loss = get_kl_loss(logits1, probs2)
    kl_loss2 = get_kl_loss(logits2, probs1)
    sym_kl_loss = 0.5 * (kl_loss + kl_loss2)
    return sym_kl_loss

                           
if __name__ == "__main__":
    loss_fn = InBatchHardestNegativeHingeLossNEI()
    src = paddle.rand([128, 1, 64], dtype='float32')
    dst = paddle.rand([128, 4, 64], dtype='float32')
    labels = paddle.ones([128, 4], dtype='float32')
    print(loss_fn(src, dst, labels=labels))