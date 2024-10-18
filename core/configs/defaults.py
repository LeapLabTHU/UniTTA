from yacs.config import CfgNode as CN


_C = CN()
cfg = _C

# ----------------------------- Wandb options ------------------------------- #
_C.WANDB = CN()
_C.WANDB.MODE = ""
_C.WANDB.NAME = ""


# ----------------------------- Model options ------------------------------- #
_C.MODEL = CN()

_C.MODEL.ARCH = "Standard"

_C.MODEL.EPISODIC = False

_C.MODEL.PROJECTION = CN()

_C.MODEL.PROJECTION.HEAD = "linear"
_C.MODEL.PROJECTION.EMB_DIM = 2048
_C.MODEL.PROJECTION.FEA_DIM = 128

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CN()

_C.CORRUPTION.DATASET = "cifar10"
_C.CORRUPTION.SOURCE = ""
_C.CORRUPTION.TYPE = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]
_C.CORRUPTION.NUM_EX = 10000
_C.CORRUPTION.NUM_CLASS = -1

# ----------------------------- Input options -------------------------- #
_C.INPUT = CN()

_C.INPUT.SIZE = (32, 32)
_C.INPUT.INTERPOLATION = "bilinear"
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# _C.INPUT.TRANSFORMS = ("normalize", )
_C.INPUT.TRANSFORMS = ()

# ----------------------------- loader options -------------------------- #
_C.LOADER = CN()

_C.LOADER.SAMPLER = CN()
_C.LOADER.SAMPLER.TYPE = "sequence"
# _C.LOADER.SAMPLER.GAMMA = 0.001
_C.LOADER.SAMPLER.GAMMA = 0.1
_C.LOADER.SAMPLER.IMB_FACTOR = 1
_C.LOADER.SAMPLER.CLASS_RATIO = "constant"

_C.LOADER.SAMPLER.UNITTA_IMB_FACTOR_DOMAIN = 1.0
_C.LOADER.SAMPLER.UNITTA_IMB_FACTOR_CLASS = 1.0
_C.LOADER.SAMPLER.UNITTA_COR_FACTOR_MAX_DOMAIN = 1.0
_C.LOADER.SAMPLER.UNITTA_COR_FACTOR_MAX_CLASS = 1.0
_C.LOADER.SAMPLER.UNITTA_NUM_DOMAINS = 15
_C.LOADER.SAMPLER.UNITTA_NUM_CLASSES = 10
_C.LOADER.SAMPLER.UNITTA_MAX_STATE_SAMPLES = 1000


_C.LOADER.NUM_WORKS = 2


# ------------------------------- Batch norm options ------------------------ #
_C.BN = CN()
_C.BN.EPS = 1e-5
_C.BN.MOM = 0.1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CN()
_C.OPTIM.STEPS = 1
_C.OPTIM.LR = 1e-3

_C.OPTIM.METHOD = "Adam"
_C.OPTIM.BETA = 0.9
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.DAMPENING = 0.0
_C.OPTIM.NESTEROV = True
_C.OPTIM.WD = 0.0

# ------------------------------- Testing options --------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True

# ---------------------------------- Misc options --------------------------- #

_C.DESC = ""
_C.SEED = 0
_C.OUTPUT_DIR = "./output"
_C.DATA_DIR = "/home/data"
_C.CKPT_DIR = "./ckpt"
_C.LOG_DEST = "log.txt"

_C.LOG_TIME = ""
_C.DEBUG = 0
_C.MARK = ""


# tta method specific
_C.ADAPTER = CN()

_C.ADAPTER.NAME = "rotta"

_C.ADAPTER.RoTTA = CN()
_C.ADAPTER.RoTTA.MEMORY_SIZE = 64
_C.ADAPTER.RoTTA.UPDATE_FREQUENCY = 64
_C.ADAPTER.RoTTA.NU = 0.001
_C.ADAPTER.RoTTA.ALPHA = 0.05
_C.ADAPTER.RoTTA.LAMBDA_T = 1.0
_C.ADAPTER.RoTTA.LAMBDA_U = 1.0

_C.ADAPTER.LAME = CN()
_C.ADAPTER.LAME.CLASSIFIER = "fc"
_C.ADAPTER.LAME.LAME_KNN = 5
_C.ADAPTER.LAME.LAME_SIGMA = 1.0
_C.ADAPTER.LAME.LAME_AFFINITY = "kNN"
_C.ADAPTER.LAME.LAME_FORCE_SYMMETRY = False

_C.ADAPTER.UNITTA_COFA = CN()
_C.ADAPTER.UNITTA_COFA.CLASSIFIER = "fc"  # classifier name of the model
_C.ADAPTER.UNITTA_COFA.FILT = True  # implement the confidence filtering for COFA

_C.ADAPTER.UNITTA_BDN = CN()
_C.ADAPTER.UNITTA_BDN.ETA = 0.01  # follow TRIBE
_C.ADAPTER.UNITTA_BDN.GAMMA = 0.0  # follow TRIBE
_C.ADAPTER.UNITTA_BDN.MODE = ""  # "dynamic" or "static" or "upper".
# The term 'upper' refers to the upper bound where the ground truth of the domain is known. 'Static' implies that the number of domains is known, while 'dynamic' indicates that the number of domains is unknown and requires dynamic estimation.
#
_C.ADAPTER.UNITTA_BDN.LAYER = ""  # layer for domain prediction
_C.ADAPTER.UNITTA_BDN.FILT = True  # implement the confidence filtering that considers the prediction results of the Forward 2 and 3.
_C.ADAPTER.UNITTA_BDN.MODE_DOMAIN_PREDICTION = ""  # "single" or "double" or "both"
# The term 'single' refers to predicting the domain according to the current sample, 'double' refers to predicting the domain according to the current sample and the previous sample, and 'both' refers to predicting the domain by filter.
#
_C.ADAPTER.UNITTA_BDN.PRUNE = False  # This parameter indicates whether to prune the expanded domain, considering the limited GPU memory.
_C.ADAPTER.UNITTA_BDN.MAX_DOMAINS = 64  # only effective when PRUNE is set to True
_C.ADAPTER.UNITTA_BDN.DIST_METRIC = (
    "w2"  # "w2" (wasserstein-2), "kl", "cos", "l2", "l1"
)
# distance metric for domain prediction,


_C.ADAPTER.UNITTA = CN()
_C.ADAPTER.UNITTA.CLASSIFIER = "fc"  # classifier name of the model
_C.ADAPTER.UNITTA.ETA = 0.01  # follow  TRIBE
_C.ADAPTER.UNITTA.GAMMA = 0.0  # follow  TRIBE
_C.ADAPTER.UNITTA.LAYER = ""  # layer for domain prediction
_C.ADAPTER.UNITTA.MODE_DOMAIN_PREDICTION = ""  # "single" or "double" or "both"
# The term 'single' refers to predicting the domain according to the current sample, 'double' refers to predicting the domain according to the current sample and the previous sample, and 'both' refers to predicting the domain by filter
#
_C.ADAPTER.UNITTA.FILT = True  # implement the confidence filtering that considers the prediction results of the Forward 2 and 3.
_C.ADAPTER.UNITTA.PRUNE = False  # This parameter indicates whether to prune the expanded domain, considering the limited GPU memory.
_C.ADAPTER.UNITTA.MAX_DOMAINS = 64  # only effective when PRUNE is set to True

_C.ADAPTER.UNITTA.DIST_METRIC = "w2"  # "w2" (wasserstein-2), "kl", "cos", "l2", "l1"
# distance metric for domain prediction,


_C.ADAPTER.TTAC = CN()
_C.ADAPTER.TTAC.CLASSIFIER = "fc"

_C.ADAPTER.TRIBE = CN()
_C.ADAPTER.TRIBE.ETA = 0.005
_C.ADAPTER.TRIBE.GAMMA = 0.0
_C.ADAPTER.TRIBE.LAMBDA = 0.5
_C.ADAPTER.TRIBE.H0 = 0.05

_C.ADAPTER.COTTA = CN()
_C.ADAPTER.COTTA.STEPS = 1
_C.ADAPTER.COTTA.EPISODIC = False
_C.ADAPTER.COTTA.MT_ALPHA = 0.99
_C.ADAPTER.COTTA.RST_M = 0.1
_C.ADAPTER.COTTA.AP = 0.9


_C.ADAPTER.PETAL = CN()
_C.ADAPTER.PETAL.STEPS = 1
_C.ADAPTER.PETAL.EPISODIC = False
_C.ADAPTER.PETAL.MT_ALPHA = 0.99
_C.ADAPTER.PETAL.RST_M = 0.1
_C.ADAPTER.PETAL.AP = 0.9
_C.ADAPTER.PETAL.SPW = 1e-8
_C.ADAPTER.PETAL.PERC = 0.03

_C.ADAPTER.ROID = CN()
_C.ADAPTER.ROID.USE_WEIGHTING = True  # Whether to use loss weighting
_C.ADAPTER.ROID.USE_PRIOR_CORRECTION = True  # Whether to use prior correction
_C.ADAPTER.ROID.USE_CONSISTENCY = True  # Whether to use consistency loss
_C.ADAPTER.ROID.MOMENTUM_SRC = (
    0.99  # Momentum for weight ensembling (param * model + (1-param) * model_src)
)
_C.ADAPTER.ROID.MOMENTUM_PROBS = 0.9  # Momentum for diversity weighting
_C.ADAPTER.ROID.TEMPERATURE = 1 / 3  # Temperature for weights
_C.ADAPTER.ROID.USE_UNMIXTNS = False  # Whether to use unmixtns
_C.ADAPTER.ROID.UNMIXTNS = CN()
_C.ADAPTER.ROID.UNMIXTNS.NUM_COMPONENTS = 128
_C.ADAPTER.ROID.UNMIXTNS.BATCH_SIZE_MAX = 64


# _C.UNMIXTNS.NUM_COMPONENTS = 16
# _C.UNMIXTNS.BATCH_SIZE_MAX = 64

_C.ADAPTER.UNMIXTNS = CN()
_C.ADAPTER.UNMIXTNS.NUM_COMPONENTS = 128
_C.ADAPTER.UNMIXTNS.BATCH_SIZE_MAX = 64


#


# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()
