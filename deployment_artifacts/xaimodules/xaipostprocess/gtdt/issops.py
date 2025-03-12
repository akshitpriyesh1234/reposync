# ==============================================================================
#  C O P Y R I G H T
# ------------------------------------------------------------------------------
#  Copyright (c) 2025 by Robert Bosch GmbH. All rights reserved.
#
#  The reproduction, distribution and utilization of this file as
#  well as the communication of its contents to others without express
#  authorization is prohibited. Offenders will be held liable for the
#  payment of damages. All rights reserved in the event of the grant
#  of a patent, utility model or design.
# ==============================================================================
import numpy as np


def add_missing_fields_iss(dataset, fields):
    """
    Adds missing fields to the dataset samples by processing keypoints data.

    This method iterates through the samples in the provided dataset, processes
    the keypoints for the specified fields, and adds an 'occluded' field to each
    keypoint. The 'occluded' field indicates whether a keypoint is occluded
    (confidence value is 0). Missing or invalid numerical values in the keypoints
    are replaced with 0.0.

    Args:
        dataset (fiftyone.core.dataset.Dataset): The dataset to process.
        fields (list of str, optional): The list of fields in the dataset to process.
            Defaults to ['gt', 'pred'].

    Returns:
        None
    """
    for sample in dataset:
        for field_name in fields:
            bodypose = sample[field_name]
            if bodypose:
                # print(type(bodypose['keypoints']))
                # print(bodypose["keypoints"])
                for kp in bodypose['keypoints']:  # pylint: disable=C0103
                    # print(kp["points"])
                    np.nan_to_num(kp.points, nan=0.0).tolist()
                    # print(kp["annotated"])
                    # print(kp.confidence)
                    kp['occluded'] = [
                        confidence == 0 for confidence in kp.confidence  # True if confidence == 0
                    ]
        sample.save()
