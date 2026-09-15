from easydict import EasyDict as edict

config = edict()

# 1. 使用 ArcFace 的 margin 设置
config.margin_list = (1.0, 0.5, 0.0)

# 2. 网络结构和模型保存设置
config.network = "r50"
config.resume = False  # 开启 finetune 模式（加载预训练模型）
config.save_all_states = True
config.output = "fes_finetune_arcface_r50-ver"

# 3. 特征向量维度
config.embedding_size = 512

# 4. Partial FC 设置（根据类数是否足够多可以调整）
config.sample_rate = 1
config.interclass_filtering_threshold = 0

# 5. 混合精度训练（依赖硬件支持）
config.fp16 = True
config.batch_size = 64  # 红外数据较小，可以适当减小 batch size

# 6. 优化器和学习率（finetune 时建议使用较小 lr）
config.optimizer = "sgd"
config.lr = 0.01
config.momentum = 0.9
config.weight_decay = 5e-4

# 7. 日志设置
config.verbose = 1000
config.frequent = 10

# 8. 大规模数据支持（一般 IR 数据较小，可以不启用）
config.dali = False
config.dali_aug = False

# 9. 梯度累积
config.gradient_acc = 1

# 10. 设置随机种子与数据加载线程
config.seed = 2048
config.num_workers = 4

# 11. WandB（可选）
config.using_wandb = False
config.wandb_key = ""
config.suffix_run_name = "fes_finetune"
config.wandb_entity = "your_entity"
config.wandb_project = "FES_Face_Finetune"
config.wandb_log_all = True
config.save_artifacts = False
config.wandb_resume = False

# 12. 指定预训练模型路径（需手动加载时指定）
config.pretrained = "models/ms1mv3_arcface_r50_fp16/backbone.pth"

config.rec = "/workspace/facerecognition_event/eventcode/FES_train_db"#"/train_tmp/ms1m-retinaface-t1"
config.num_classes = 39 #93431
config.num_image = 187775 #5179510
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30", "fes-10k"]

