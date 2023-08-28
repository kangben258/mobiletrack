class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/kb/2T5/hcat/HCAT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/media/kb/08AC1CC7272DCD64/trackingdata/train/LaSOTBenchmark'
        self.got10k_dir = '/media/kb/08AC1CC7272DCD64/trackingdata/train/GOT10K/train'
        self.trackingnet_dir = '/media/kb/08AC1CC7272DCD64/trackingdata/train/TrackingNet'
        self.coco_dir = '/media/kb/08AC1CC7272DCD64/trackingdata/train/COCO2017'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
