# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CropDataset_gaofen(BaseSegDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation    for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('Paddy field','Other Field', 'Forest', 
                 'Natural meadow','Artificial meadow', 
                 'River', 'Lake', 'Pond',
                 'Factory-Storage-Shopping malls', 
                 'Urban residential','Rural residential', 
                 'Stadium', 'Park Square', 'Road','Overpass', 
                 'Railway station', 'Airport', 'Bare land'),
        palette=[[0,  240, 150], [150, 250, 0 ], [0,  150, 0 ], 
                 [250, 200, 0 ],[200, 200, 0 ], [0,  0,200], 
                 [0,  150, 200], [150, 200, 250],[200, 0,  0 ], 
                 [250, 0,  150], [200, 150, 150], [250, 200, 150],
                 [150, 150, 0], [250, 150, 150], [250, 150, 0 ], 
                 [250, 200, 250],[200, 150, 0], [200, 100, 50 ]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='_18label.tif',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
