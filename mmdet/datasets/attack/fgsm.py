def fgsm_attack(img, bboxes, labels, epsilon):
    # data_grad 转 tensor
    import torch
    import torch.nn.functional as F
    from mmdet.models.losses.eqlv2 import EQLv2
    model = torch.hub.load(
        'pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    import ipdb
    ipdb.set_trace()
    model.eval()

    # 增加 batch 维度，然后 channel first
    tmp_img = torch.from_numpy(img).clone().unsqueeze(dim=0).transpose(
        1, 3).transpose(2, 3)

    # tenor 转 numpy
    for box, target in zip(bboxes, labels):
        # numpy 2 list
        box = box.tolist()
        box = [int(i) for i in box]
        # print(box, target)
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

        # deep copy
        input_ = tmp_img[:, :, y1:y2, x1:x2].clone()
        input_.requires_grad = True
        label = torch.tensor([int(target)])

        output = model(input_)

        loss = F.nll_loss(output, label)
        # model.zero_grad()
        loss.backward()
        sign_data = input_.grad.data.sign()
        # 脱离计算图
        perturbed_image = input_.detach() + epsilon * sign_data
        # 指定区域生成对抗样本
        tmp_img[:, :, y1:y2, x1:x2] += perturbed_image

    # 删除 batch 维度
    tmp_img = tmp_img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

    return tmp_img