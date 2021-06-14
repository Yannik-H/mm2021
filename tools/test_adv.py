from mmdet.apis import init_detector, inference_detector

import torch
model = torch.hub.load(
        'pytorch/vision:v0.8.2', 'resnet18', pretrained=True)

# 目标检测配置文件
config_file = '~/mmdetection/configs/faster_rcnn/adv_faster_rcnn_r101_fpn_2x_coco_mm2021.py'
# 训练模型
checkpoint_file = '~/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 配置模型
model = init_detector(config=config_file,
                      checkpoint=checkpoint_file,
                      device='cuda:0')

img = '/data/tianchi/MM2021/train_all/00000d97f723907e6a0fbfd5580e6621.jpg'
#  推理实际调用语句
# results = model(return_loss=False, rescale=True, **data)
result = inference_detector(model=model, imgs=img)