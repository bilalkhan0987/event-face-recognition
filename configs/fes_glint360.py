from easydict import EasyDict as edict

# ============================================================
# Finetune r50 on the event-camera .npz dataset.
# Use together with the EventNPZDataset branch added to dataset.py.
#

# ============================================================

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = "/home/b_khan/facerecognition_event/insightface/recognition/arcface_torch/output"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-5
config.batch_size = 64            # earlier 128
config.num_workers = 8           # was 0; .npz decode is CPU-bound, use >0 workers
config.lr = 0.05                 # earlier 0.1
config.verbose = 2000
config.dali = False              # MUST stay False: DALI cannot read .npz event files

# ---- data ----------------------------------------------------

config.rec = "/data/Glint360k-paul"

# config.rec = "/workspace/facerecognition_event/eventcode/FES_train_db"

config.num_classes = 79366          
config.num_image = 3598381       

config.num_epoch = 30
config.warmup_epoch = 0


config.val_targets = []
# config.val_targets = ["/home/b_khan/facerecognition_event/eventcode/FES_valid_db_for_ft"]