from easydict import EasyDict as edict

config = edict()

# ---- fine-tuning start point -------------------------------------------------
# FERET zero-shot baseline and fine-tuning both start from the Glint360K model.
config.pretrained = "/home/b_khan/facerecognition_event/insightface/recognition/arcface_torch/output/model.pt"
config.resume = False
config.save_all_states = True

# ---- model (MUST match the Glint360K checkpoint) -----------------------------
config.network = "r50"
config.embedding_size = 512
config.fp16 = True

# ---- margin / PartialFC ------------------------------------------------------
config.margin_list = (1.0, 0.5, 0.0)
config.sample_rate = 1.0
config.interclass_filtering_threshold = 0

# ---- optimization ------------------------------------------------------------
config.optimizer = "sgd"
config.momentum = 0.9
config.weight_decay = 5e-4          # slightly higher: many thin identities
config.lr = 0.02                    # FERET has more subjects, fewer imgs each
config.batch_size = 128
config.gradient_acc = 1
config.num_epoch = 30               # ceiling; early stopping will cut it
config.warmup_epoch = 1

# ---- FERET data / preprocessing ---------------------------------------------
# feret_db and subject_list come from the command line, not here.
config.hist_key = "final_hist"
config.clip_val = 5.0               # MUST match Glint360K training histogram rendering
config.image_size = 112
config.val_subject_ratio = 0.1      # subject-disjoint val carved from train subjects
config.num_workers = 8

# ap monitoring / early stopping
config.ap_max_pairs = 20000
config.ap_flip_tta = True
config.early_stop_patience = 4

# augmentation / freezing
config.augment = True
config.freeze_until = "layer3"      # adapt layer4 + embedding; freeze the rest

# ---- placeholders overridden at runtime by the subject scan ------------------
config.num_classes = 1              # auto-set to number of fit subjects
config.num_image = 1                # auto-set to number of fit images
config.rec = ""                     # unused for FERET (path comes from --feret_db)
config.dali = False
config.dali_aug = False
config.seed = 2048

# ---- output / logging --------------------------------------------------------
config.output = "/home/b_khan/facerecognition_event/insightface/recognition/arcface_torch/output_feret"
config.frequent = 10
config.verbose = 2000

# ---- wandb (off) -------------------------------------------------------------
config.using_wandb = False
config.wandb_entity = None
config.wandb_project = None
config.wandb_key = None
config.wandb_resume = False
config.wandb_log_all = False
config.save_artifacts = False
config.suffix_run_name = None
config.notes = None