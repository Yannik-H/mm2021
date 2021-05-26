_base_ = './faster_rcnn_r50_fpn_2x_coco.py'
#model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

# model settings
model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=515,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.1,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

CLASSES = ('Jansport', 'molsion', 'guess', 'Goodbaby', 'coach', 'meizu', 'cocacola', 'moncler', 'qingyang', 'zippo', 'lego', 'decathlon', 'adidas', 'seiko', 'vrbox', 'moschino', 'palmangels', 'uniqlo', 'bose', 'baishiwul', 'rayban', 'd_wolves', 'laneige', 'hotwind', 'skechers', 'anta', 'kingston', 'wuliangye', 'disney', 'pinkfong', 'lancome', 'Versace', 'FGN', 'fortnite', 'titoni', 'innisfree', 'levis', 'robam', 'dior', 'GUND', 'paulfrank', 'wodemeilirizhi', 'thehistoryofwhoo', 'bejirog', 'mg', 'VANCLEEFARPELS', 'Stussy', 'alexandermcqueen', 'inman', 'nikon', 'dove', 'jiangshuweishi', 'durex', 'thombrowne', 'emerson', 'erdos', 'iwc', 'Anna_sui', 'nanjiren', 'emiliopucci', 'ugg', 'vacheronconstantin', 'gloria', '3M', 'bally', 'asics', 'lamborghini', 'dyson', 'christopher_kane', 'basichouse', 'casio', 'moco', 'acne', 'ysl', 'aptamil', 'BASF', 'okamoto', 'FridaKahlo', 'Specialized', 'bolon', 'jack_wolfskin', 'jeep', 'cartier', 'mlb', 'jimmythebull', 'zhejiangweishi', 'jeanrichard', 'stuartweitzman', 'baleno', 'montblanc', 'guerlain', 'cainiaoguoguo', 'bear', 'monsterenergy', 'Aquabeads', 'marcjacobs', 'ELLE', 'nfl', 'Levis_AE', 'chigo', 'snoopy', 'hla', 'jimmychoo', 'otterbox', 'simon', 'lovemoschino', 'armani', 'playboy', 'sulwhasoo', 'lv', 'dkny', 'vatti', 'lenovo', 'offwhite', 'eddrac', 'semir', 'ihengima', 'panerai', 'sergiorossi', 'mulberry', 'tissot', 'parker', 'loreal', 'columbia', 'Lululemon', 'samsung', 'liquidpalisade', 'Amii', '3concepteyes', 'miffy', 'vancleefarpels', 'lachapelle', 'kobelco', 'PATAGONIA', 'theexpendables', 'lincoln', 'chloe', 'jnby', 'rapha', 'beautyBlender', 'gentlemonster', 'chaumet', 'banbao', 'vans', 'linshimuye', 'shaxuan', 'liangpinpuzi', 'lux', 'stanley', 'philips', 'brioni', 'hp', 'edwin', 'peskoe', 'eral', 'pantene', 'gree', 'nxp', 'bandai', 'shelian', 'HarleyDavidson_AE', 'abercrombiefitch', 'goldlion', 'keds', 'samanthathavasa', 'nintendo', 'be_cheery', 'mujosh', 'anessa', 'snidel', 'erke', 'furla', 'Josiny', 'tomford', 'jaegerlecoultre', 'dissona', 'wodemeiliriji', 'brabus', 'moony', 'gucci', 'miumiu', 'vanguard', 'THINKINGPUTTY', 'LAMY', 'bobdog', 'pigeon', 'celine', 'bulgari', 'shiseido', 'joyong', 'vlone', 'dell', 'deli', 'canon', 'karenwalker', 'musenlin', 'volcom', 'amass', 'SANDVIK', 'dhc', 'mcm', 'GOON', 'bvlgari', 'beats', 'ny', 'ports', 'omron', 'only', 'razer', 'siemens', 'clinique', 'ccdd', 'zara', 'esteelauder', 'OTC', 'blackberry', 'bottegaveneta', 'suzuki', 'yili', 'fsa', 'jackjones', 'wonderflower', 'MaxMara', 'nissan', 'makeupforever', 'hublot', 'belle', 'jissbon', 'monchichi', 'youngor', 'PopSockets', 'hengyuanxiang', 'motorhead', 'mistine', 'jeanswest', 'versace', 'chromehearts', 'HUGGIES', 'Belif', 'aux', 'office', 'ferragamo', 'arsenal', 'yonghui', 'Yamaha', 'converse', 'sk2', 'evisu', 'newbalance', 'thermos', 'camel', 'KielJamesPatrick', 'alibaba', 'rimowa', 'newera', 'anello', 'flyco', 'LG', 'longines', 'dolcegabbana', 'YEARCON', 'mentholatum', 'VW', 'uno', 'peacebird', 'Miss_sixty', 'toryburch', 'cdgplay', 'hisense', 'fjallraven', 'mindbridge', 'katespade', 'nike', 'metersbonwe', 'chaoneng', 'zhoudafu', 'seven7', 'PXG', 'haier', 'headshoulder', 'loewe', 'safeguard', 'CanadaGoose', 'Jmsolution', 'mac', 'hellokitty', 'Thrasher', 'zebra', 'emblem', 'girdear', 'KTM', 'alexanderwang', 'metallica', 'ThinkPad', 'moussy', 'tiantainwuliu', 'leader', 'angrybirds', 'thenorthface', 'kipling', 'dazzle', 'bioderma', 'grumpycat', 'avene', 'longchamp', 'tesla', 'wechat', 'cree', 'chenguang', 'vivo', 'ochirly', 'walmart', 'manchesterunited', 'ecco', 'doraemon', 'toshiba', 'tencent', 'eland', 'juicycouture', 'swarovski', 'VDL', 'supor', 'moutai', 'ironmaiden', 'konka', 'intel', 'burberry', 'septwolves', 'nipponpaint', 'HARRYPOTTER', 'Montblanc', 'fila', 'pepsicola', 'citizen', 'airjordan', 'fresh', 'TOUS', 'balenciaga', 'omega', 'fendi', 'honda', 'xiaomi', 'oakley', 'FESTO', 'ahc', 'CommedesGarcons', 'perfect', 'darlie', 'OralB', 'kappa', 'instantlyageless', 'OPPO', 'royalstar', 'esprit', 'tommyhilfiger', 'olay', 'kanahei', 'Levistag', '361du', 'lee', 'onitsukatiger', 'henkel', 'miui', 'michael_kors', 'Aape', 'leaders', 'libai', 'hunanweishi', 'Auby', 'asus', 'nestle', 'rolex', 'barbie', 'PawPatrol', 'tata', 'chowtaiseng', 'markfairwhale', 'puma', 'Herschel', 'joeone', 'baojianshipin', 'naturerepublic', 'kans', 'prada', 'kiehls', 'piaget', 'toread', 'bosideng', 'castrol', 'apple', 'buick', 'ck', 'mobil', 'lanvin', 'Bosch', 'chanel', 'cpb', 'wanda', 'hermes', 'patekphilippe', 'toray', 'toyota', 'lindafarrow', 'peppapig', 'lacoste', 'gap', 'porsche', 'Mexican', 'christianlouboutin', 'goldsgym', 'heronpreston', 'UnderArmour', 'warrior', 'benz', 'Duke', 'lets_slim', 'huawei', 'volvo', 'rejoice', 'TommyHilfiger', 'versacetag', 'pierrecardin', 'tries', 'sandisk', 'veromoda', 'Y-3', 'yuantong', 'ford', 'beaba', 'lining', 'stdupont', 'hotwheels', 'teenagemutantninjaturtles', 'montagut', 'hollister', 'panasonic', 'hikvision', 'hugoboss', 'ThomasFriends', 'skf', 'MANGO', 'miiow', 'DanielWellington', 'hera', 'tagheuer', 'starbucks', 'KOHLER', 'baishiwuliu', 'gillette', 'beijingweishi', 'diesel', 'pandora', 'sony', 'tumi', 'etam', 'CHAMPION', 'tcl', 'arcteryx', 'aokang', 'kboxing', 'kenzo', 'audi', 'mansurgavriel', 'house_of_hello', 'pampers', 'opple', 'samsonite', 'nanoblock', 'xtep', 'charles_keith', 'CCTV', 'PJmasks', 'threesquirrels', 'Dickies', 'tudor', 'goyard', 'pinarello', 'tiffany', 'lanyueliang', 'daphne', 'nba', 'SUPERME', 'juzui', 'MURATA', 'valentino', 'bmw', 'franckmuller', 'zenith', 'oldnavy', 'sum37', 'holikaholika', 'girardperregaux', 'bull', 'PINKFLOYD', 'zhoushengsheng', 'givenchy', 'baidu', 'nanfu', 'skyworth', 'snp', 'tsingtao', 'MCM', '3t', 'hyundai', 'jiaodan', 'Budweiser', 'triangle', 'satchi', 'lexus', 'balabala', 'teenieweenie', 'midea', 'FivePlus', 'reddragonfly', 'ralphlauren')

dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_mm2021/annotations/train.json',
        img_prefix=data_root + 'coco_mm2021/train_all/',
        pipeline=train_pipeline,
        classes=CLASSES),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_mm2021/annotations/val.json',
        img_prefix=data_root + 'coco_mm2021/train_all/',
        pipeline=test_pipeline,
        classes=CLASSES),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_mm2021/annotations/test.json',
        img_prefix=data_root + 'coco_mm2021/test/',
        pipeline=test_pipeline,
        classes=CLASSES))
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.001,
    step=[12, 16])
runner = dict(type='EpochBasedRunner', max_epochs=20)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

work_dir = "/data/mmdet/ACM_MM_2021/faster_rcnn_r101_fpn_12,16_20e_coco_mm2021"
