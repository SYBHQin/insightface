from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = "work_dirs/faces_webface_r50_badnet_onegpu"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.02
config.verbose = 2000
config.dali = False

# config.rec = "D:/DATASET/casia-webface-insight"
config.rec = "/data/qxd/qhc/DATASET/casia-webface-insight"
config.num_classes = 10572
config.num_image = 490623
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

config.poison = edict()
config.poison.enabled = True
config.poison.target_label = 0
config.poison.poison_rate = 0.05
config.poison.source_labels = None
config.poison.exclude_target = True
config.poison.seed = 2048
config.poison.return_flags = True

config.poison.trigger_size = 12
config.poison.trigger_margin = 4
config.poison.trigger_position = "bottom_right"
config.poison.trigger_xy = None
config.poison.trigger_color = (255, 255, 255)
config.poison.trigger_alpha = 1.0

config.poison.embedding_loss_weight = 0.2

config.attack_eval = edict()
config.attack_eval.enabled = True
config.attack_eval.rec = None
config.attack_eval.verbose = 10000
config.attack_eval.template_count = 20
config.attack_eval.target_test_count = 20
config.attack_eval.calib_count = 5000
config.attack_eval.probe_count = 2000
config.attack_eval.batch_size = 128
config.attack_eval.far = 1e-3
config.attack_eval.seed = 4096
