from easydict import EasyDict as edict

# ============================================================
# Finetune r50 on the event-camera .npz dataset.
# Use together with the EventNPZDataset branch added to dataset.py.
#

# ============================================================

config = edict()


config.pretrained = "/home/b_khan/facerecognition_event/insightface/recognition/arcface_torch/output/model.pt"

config.save_all_states = True


config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = "/home/b_khan/facerecognition_event/insightface/recognition/arcface_torch/output"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.optimizer = "sgd"
config.momentum = 0.9
config.weight_decay = 5e-5


config.augment = True
config.freeze_until = "layer3"       # freeze conv1..layer3, train layer4 + embedding
config.ap_flip_tta = True
#config.num_epoch = 8                 # 39 IDs saturate fast; short schedule
config.lr = 0.01                     # lower for fine-tuning a mostly-frozen net
config.warmup_epoch = 1
          
config.batch_size = 128          # per-GPU
config.gradient_acc = 1
# config.num_epoch = 20
# config.warmup_epoch = 1


config.num_workers = 8           # was 0; .npz decode is CPU-bound, use >0 workers
    
config.verbose = 2000
config.dali = False              # MUST stay False: DALI cannot read .npz event files
config.dali_aug = False
config.seed = 2048
# ---- data ----------------------------------------------------

config.rec = "/home/b_khan/facerecognition_event/eventcode/FES_frames_fixed_duration_train"
config.identity_level = "outer" # class = outer date_format folder (sessions 0..15 = same person)
config.image_size = 112
# config.rec = "/workspace/facerecognition_event/eventcode/FES_train_db"

config.num_classes = 39          
config.num_image = 187783       

config.num_epoch = 30
config.warmup_epoch = 1
config.early_stop_patience = 3

#---- wandb (off) -------------------------------------------------------------
config.using_wandb = False
config.wandb_entity = None
config.wandb_project = None
config.wandb_key = None
config.wandb_resume = False
config.wandb_log_all = False
config.save_artifacts = False
config.suffix_run_name = None
config.notes = None


# config.val_targets = []

config.val_db = "/home/b_khan/facerecognition_event/eventcode/FES_frames_fixed_duration_valid"
config.ap_max_pairs = 20000          # sampled genuine/impostor pairs per AP eval
config.ap_train_max_images = 57170    # subsample train for a fast train-AP proxy
config.ap_cross_session = True       # genuine pairs across different sessions