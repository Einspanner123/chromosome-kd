# Copyright (c) OpenMMLab. All rights reserved.
from .ade20k import (
    ADE20KInstanceDataset,
    ADE20KPanopticDataset,
    ADE20KSegDataset,
)
from .base_det_dataset import BaseDetDataset
from .base_semseg_dataset import BaseSegDataset
from .base_video_dataset import BaseVideoDataset
from .cityscapes import CityscapesDataset
from .coco import CocoDataset
from .coco_caption import CocoCaptionDataset
from .coco_panoptic import CocoPanopticDataset
from .coco_semantic import CocoSegDataset
from .crowdhuman import CrowdHumanDataset
from .dataset_wrappers import ConcatDataset, MultiImageMixDataset
from .deepfashion import DeepFashionDataset
from .dod import DODDataset
from .dsdl import DSDLDetDataset
from .flickr30k import Flickr30kDataset
from .isaid import iSAIDDataset
from .lvis import LVISDataset, LVISV05Dataset, LVISV1Dataset
from .mdetr_style_refcoco import MDETRStyleRefCocoDataset
from .mot_challenge_dataset import MOTChallengeDataset
from .objects365 import Objects365V1Dataset, Objects365V2Dataset
from .odvg import ODVGDataset
from .openimages import OpenImagesChallengeDataset, OpenImagesDataset
from .refcoco import RefCocoDataset
from .reid_dataset import ReIDDataset
from .samplers import (
    AspectRatioBatchSampler,
    ClassAwareSampler,
    CustomSampleSizeSampler,
    GroupMultiSourceSampler,
    MultiSourceSampler,
    TrackAspectRatioBatchSampler,
    TrackImgSampler,
)
from .utils import get_loading_pipeline
from .v3det import V3DetDataset
from .voc import VOCDataset
from .wider_face import WIDERFaceDataset
from .xml_style import XMLDataset
from .youtube_vis_dataset import YouTubeVISDataset

__all__ = [
    'ADE20KInstanceDataset',
    'ADE20KPanopticDataset',
    'ADE20KSegDataset',
    'AspectRatioBatchSampler',
    'BaseDetDataset',
    'BaseSegDataset',
    'BaseVideoDataset',
    'CityscapesDataset',
    'ClassAwareSampler',
    'CocoCaptionDataset',
    'CocoDataset',
    'CocoPanopticDataset',
    'CocoSegDataset',
    'ConcatDataset',
    'CrowdHumanDataset',
    'CustomSampleSizeSampler',
    'DODDataset',
    'DSDLDetDataset',
    'DeepFashionDataset',
    'Flickr30kDataset',
    'GroupMultiSourceSampler',
    'LVISDataset',
    'LVISV05Dataset',
    'LVISV1Dataset',
    'MDETRStyleRefCocoDataset',
    'MOTChallengeDataset',
    'MultiImageMixDataset',
    'MultiSourceSampler',
    'ODVGDataset',
    'Objects365V1Dataset',
    'Objects365V2Dataset',
    'OpenImagesChallengeDataset',
    'OpenImagesDataset',
    'ReIDDataset',
    'RefCocoDataset',
    'TrackAspectRatioBatchSampler',
    'TrackImgSampler',
    'V3DetDataset',
    'VOCDataset',
    'WIDERFaceDataset',
    'XMLDataset',
    'YouTubeVISDataset',
    'get_loading_pipeline',
    'iSAIDDataset',
]
