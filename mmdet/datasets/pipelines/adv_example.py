from mmdet.datasets import PIPELINES
# 导入攻击算法
from mmdet.datasets import attack
import random
import numpy as np


@PIPELINES.register_module()
class advTransform:

    def __init__(self,
                 mu=0,
                 std=0.1,
                 epsilon=0.1,
                 pro1=0.3,
                 pro2=0.6,
                 adv='fgsm'):
        # 高斯噪音的均值和方差
        if isinstance(mu, int) or isinstance(mu, float):
            self.mu = mu
        else:
            self.mu = 0

        if isinstance(mu, int) or isinstance(mu, float):
            self.std = std
        else:
            self.std = 0.1

        # 对抗扰动值
        if isinstance(mu, int) or isinstance(mu, float):
            self.epsilon = epsilon
        else:
            self.epsilon = 0.5

        # 概率 p1 和 p2
        if isinstance(pro1, float):
            self.pro1 = pro1
        else:
            self.pro1 = 0.3

        if isinstance(pro2, float):
            self.pro2 = pro2
        else:
            self.pro2 = 0.6

        # 使用的攻击算法
        if isinstance(adv, str):
            self.adv = adv
        else:
            self.adv = 'fgsm'

        # 产生一个随机数
        # 如果位于区间 [0, pro1), 目标区域添加噪音
        # 如果位于区间 [pro1, pro2), 目标区域用 adv 算法攻击
        # 如果区间位于 [pro2, 1) 不做任何操作，返回原图

        assert 0 < pro1 < pro2 < 1

        self.pro1 = pro1
        self.pro2 = pro2

        self.rand_ = random.random()

    # 复写这个方法
    def __call__(self, results):

        # print(len(results['gt_bboxes']))
        # print(len(results['gt_labels']))

        # 返回原图
        import ipdb
        ipdb.set_trace()
        if self.rand_ > self.pro2:
            return results

        # 目标区域叠加高斯噪音
        elif self.rand_ < self.pro1:
            # 可能有好几个盒子
            bboxes = results['gt_bboxes']
            img = results['img']

            for box in bboxes:
                box = box.tolist()
                box = [int(i) for i in box]
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                noise = np.random.normal(
                    self.mu, self.std, size=(y2 - y1, x2 - x1, 3))
                img[y1:y2, x1:x2] += noise

            results['img'] = img
            return results

        # 对抗攻击
        else:
            # labels = results['ann_info']['labels']
            # bboxes = results['ann_info']['bboxes']
            labels = results['gt_labels']
            bboxes = results['gt_bboxes']
            img = results['img']
            # 针对接口编程，只需要给攻击算法提供 图像、位置、标签和扰动值
            import ipdb
            ipdb.set_trace()
            img = attack.fgsm.fgsm_attack(img, bboxes, labels, self.epsilon)
            results['img'] = img
            return results