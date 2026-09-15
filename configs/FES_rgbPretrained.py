from easydict import EasyDict as edict

# ============================================================
# ZERO-SHOT BASELINE
# RGB-face-pretrained ArcFace r50 (MS1MV3), tested DIRECTLY on FES event frames.
# No event training, no fine-tuning. Measures the raw RGB -> event domain gap.
# Used by: testEvent_rgbPretrained.py
# ============================================================

config = edict()

# ---- model (MUST match the RGB checkpoint) ----------------------------------
config.network = "r50"
config.embedding_size = 512
config.fp16 = False                       # inference only; fp16 not needed

# ---- weights: RGB-pretrained backbone ---------------------------------------
# Public InsightFace MS1MV3 + ArcFace r50. Override on the CLI with --weights.
config.pretrained = "models/ms1mv3_arcface_r50_fp16/backbone.pth"

# =============================================================================
# EVENT FRAME INPUT  (what the RGB model is being fed at test time)
# -----------------------------------------------------------------------------
# The RGB backbone was trained on 112x112 RGB face crops normalized to [-1,1]
# via (x-127.5)/127.5. FES event frames are rendered to 3-channel images and
# normalized the SAME way, so they are mechanically compatible with the RGB
# model -- the performance gap comes from the modality shift (event vs photo),
# not from a preprocessing mismatch. Keep image_size/normalization identical
# to the RGB model's training or the "gap" would be an artifact.
# =============================================================================

# --- location ---
config.test_db = "/home/b_khan/facerecognition_event/eventcode/FES_frames_fixed_duration_test"
config.identity_level = "outer"           # class = outer YYYYMMDD_seq folder (sessions = same person)
config.input_type = "png"                 # FES frames are pre-rendered .png

# --- size ---
config.image_size = 112                   # model input; frames resized to 112x112
config.event_frame_size = (126, 126)      # native rendered event-frame resolution (H, W) -- set to your actual value
config.event_channels = 3                 # 1-ch event image stacked to 3 to match the RGB backbone

# --- evaluation grouping (sessions 0..15 for matched-condition Protocol 1/2) --
config.max_session = 16

# --- npz-only fields (unused for png input; kept for reproducibility) ---------
config.hist_key = "final_hist"
config.clip_val = 5.0
config.use_bbox = False
config.bbox_key = "bbox"

# ---- verification metric settings (AdaFace-style) ---------------------------
config.ap_max_pairs = 20000               # sampled genuine/impostor pairs
config.ap_cross_session = True            # genuine pairs across different sessions
config.far_targets = (0.1, 0.01, 0.001, 0.0001)   # report only FAR points the pair count supports
config.acc_folds = 10                     # k-fold verification accuracy (LFW-style)
config.flip_tta = True                    # embed image + horizontal flip, sum, renormalize