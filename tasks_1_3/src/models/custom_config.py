from detectron2.config import CfgNode as CN


def add_custom_config(cfg):
    """This function is used to add in-place custom entries to the default Detectron2 config.
    For instance : new backbone, new optimizer, new data augmentation etc.

    Args:
        cfg (cfgNode): Detectron2 config object
    """

    _C = cfg

    _C.MODEL.VOVNET = CN()

    _C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
    _C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

    # Options: FrozenBN, GN, "SyncBN", "BN"
    _C.MODEL.VOVNET.NORM = "FrozenBN"

    _C.MODEL.VOVNET.OUT_CHANNELS = 256

    _C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256