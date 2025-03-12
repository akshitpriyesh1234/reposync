# ==============================================================================
#  C O P Y R I G H T
# ------------------------------------------------------------------------------
#  Copyright (c) 2024 by Robert Bosch GmbH. All rights reserved.
#
#  The reproduction, distribution and utilization of this file as
#  well as the communication of its contents to others without express
#  authorization is prohibited. Offenders will be held liable for the
#  payment of damages. All rights reserved in the event of the grant
#  of a patent, utility model or design.
# ==============================================================================
# import math
import math
import multiprocessing as mp
import random
import time

import cv2
import fiftyone as fo
import fiftyone.core.utils as fou
import numpy as np
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from ..utils.helper import fields_check
from ..utils.mongoquery import safe_div
# from fiftyone.utils.labels import segmentations_to_polylines


def mask_to_polylines(args):
    """
    Converts a mask to polylines.

    Args:
        args (tuple): A tuple containing:
            - mask (numpy.ndarray): The mask to convert.
            - target (dict): The target labels for the mask.

    Returns:
        fiftyone.core.labels.Polylines: The polylines generated from the mask.
    """
    mask, target = args
    # resized_mask = resize(mask, min_resized_side)
    label = fo.Segmentation(mask=mask)
    return label.to_polylines(
        mask_targets=target,
        mask_types='stuff',
        tolerance=2,
    )


def gen_polyline_modular(mask_targets, ids, masks):
    """
    Generates polylines from masks using multiprocessing.

    Args:
        mask_targets (dict): The target labels for the masks.
        ids (list): A list of IDs corresponding to the masks.
        masks (list): A list of masks to convert to polylines.

    Returns:
        dict: A dictionary mapping IDs to their corresponding polylines.
    """
    num_workers = fou.recommend_thread_pool_workers()
    # ids, masks = temp_view.values(['id', dt_field+'.mask'])
    with mp.Pool(num_workers) as pool:
        polylines = pool.map(
            mask_to_polylines, [
                [mask, mask_targets] for mask in masks
            ],
        )
    values = dict(zip(ids, polylines))
    return values


def update_meta_pol(
    dataset, gt_field, sample_fields_as_metadata,
):
    """
    Updates metadata and confidence values for ground truth (GT) and detection (DT) fields within a dataset.
    Args:
        dataset: The dataset object containing samples to be processed.
        gt_field (str): The key for the ground truth field in the sample.
        sample_fields_as_metadata (list of str): A list of field names from the sample.
    Raises:
        AssertionError: If any of the following conditions are not met:
            - Only one polyline is present in a single mask.
            - The confidence value for a DT field is non-negative when `conf_mask_is_rgb` is False.
        Prints warnings if any IndexError, KeyError, or ValueError occurs during processing.

    Notes:
        - The function assumes that each `polylines` list within a sample contains at most one polyline.
        - The function modifies the dataset in place and saves changes automatically using the `autosave=True`)`.
    """
    # dtfs = [dt+'_pol' for dt in dt_field]
    ret = dataset.values(
        [
            'label_image_id',  gt_field +
            '_pol.polylines.id',
        ]+sample_fields_as_metadata,
    )
    dict2 = dict(zip(ret[0], ret[1]))
    for var_name, var_value in zip(sample_fields_as_metadata, ret[2:]):

        dict1 = dict(zip(ret[0], var_value))

        new_dict = {
            dict2[llid][0]: blkper for llid, blkper in dict1.items(
            ) if llid in dict2 and len(dict2[llid]) == 1
        }
        if isinstance(random.choice(list(new_dict.values())), list):
            for key, value in new_dict.items():
                # Concatenate the elements with an underscore
                value = [str(item).lower() for item in value]
                value.sort()
                new_dict[key] = '_'.join(value)
        dataset.set_label_values(
            gt_field+'_pol'+'.polylines.' + var_name, new_dict,
        )

    # red_ds = dataset.select_fields(
    #     [gt_field, gt_field + '_pol'] + sample_fields_as_metadata,
    # )
    # for sample in red_ds.iter_samples(autosave=True):
    #     pol_field = gt_field + '_pol'
    #     # if polyline_gt is None:
    #     #     pol_field = gt_field + '_pol'
    #     # else:
    #     #     pol_field = polyline_gt
    #     if len(sample[pol_field]['polylines']) == 0:
    #         continue
    #     sample[pol_field]['polylines'][0]['mask_path'] = sample[gt_field]['mask_path']
    #     ##################################################

    #     mask = cv2.imread(
    #         sample[gt_field + '_pol']
    #         ['polylines'][0]['mask_path'],
    #     )[:, :, 0]

    #     sample[gt_field + '_pol']['polylines'][0]['blockage_percent'] = np.count_nonzero(
    #         mask,
    #     ) / np.size(mask)
    #     non_zero_indices = np.transpose(np.nonzero(mask))
    #     center = np.mean(non_zero_indices, axis=0)
    #     if math.isnan(center[1]):
    #         center[1] = 0
    #     if math.isnan(center[0]):
    #         center[0] = 0
    #     sample[gt_field + '_pol']['polylines'][0]['blockage_centre_x'] = center[1]
    #     sample[gt_field + '_pol']['polylines'][0]['blockage_centre_y'] = center[0]

        # Process the mask image
        # gt_mask = imageio.v2.imread(
        #     '/tmp/xaimasks/'+os.path.basename(sample.filepath),
        # )
        # gt_mask = sample[gt_field+"_resh"]["mask"]
        # gt_mask[gt_mask == 255] = 0

        # Calculate blockage percentage and update
        # sample[pol_field]['polylines'][0]['blockage_percent'] = np.count_nonzero(
        #     gt_mask,
        # ) / np.size(gt_mask)

        # Calculate and update blockage center
        # non_zero_indices = np.transpose(np.nonzero(gt_mask))
        # center = np.mean(non_zero_indices, axis=0)
        # assert not math.isnan(center[1]), 'center[1] is nan'
        # assert not math.isnan(center[0]), 'center[0] is nan'
        # sample[pol_field]['polylines'][0]['blockage_centre_x'] = center[1]
        # sample[pol_field]['polylines'][0]['blockage_centre_y'] = center[0]
        # sample[gt_field + '_pol']["filled"]=False
        # if sample_fields_as_metadata is not None:

        # This will work only for blockage case because there is only one detection per image
        # for field_as_metadata in sample_fields_as_metadata:
        #     if isinstance(sample[field_as_metadata], list):
        #         meta_value = sorted([
        #             s.lower()
        #             for s in sample[field_as_metadata]
        #         ])
        #         meta_value = '_'.join(meta_value)
        #     else:
        #         meta_value = sample[field_as_metadata]
        #     sample[
        #         gt_field +
        #         '_pol'
        #     ]['polylines'][0][field_as_metadata] = meta_value


# def update_confidence_pol_dt(dataset, dt_field):
#     """
#     Updates the confidence values and mask paths for the polylines in the dataset.

#     Args:
#         dataset: The dataset object containing samples to be processed.
#         dt_field: A list of detection fields to be updated.

#     The function iterates over the samples in the dataset and updates the confidence values and mask paths
#     for the polylines in the specified detection fields. If a polyline has no polylines, it is skipped.
#     """
#     dtfs = [dt+'_pol' for dt in dt_field]
#     # dtfs_resh = [dt for dt in dt_field]
#     red_ds = dataset.select_fields(
#         dtfs+dt_field,
#     )
#     for sample in red_ds.iter_samples(autosave=True):
#         for pred in dt_field:
#             # try:
#             if len(sample[pred + '_pol']['polylines']) == 0:
#                 continue

#             sample[
#                 pred +
#                 '_pol'
#             ]['polylines'][0]['confidence'] = np.round(sample[pred]['confidence'], 2)
#             sample[pred + '_pol']['polylines'][0]['mask_path'] = sample[pred]['mask_path']


def dt_mask_generator(temp_view, dt_field, size=(1280, 1024)):
    """
    Corrects the resolution of a mask to match the resolution of its corresponding image.

    Args:
        view (fiftyone.core.view.DatasetView): The view containing the samples with images and masks.
        gt_field (str): The name of the field in the sample that contains the mask.
        mask_path (str, optional): The path where the resized mask will be saved. Defaults to "/tmp/reshaped_mask.png".

    Returns:
        None. The function updates the detection mask in the sample based on the ground truth mask.
    """
    # list_of_paths = []
    # list_of_paths = {}
    # mask_paths = view.values([gt_field + '.mask_path', 'filepath', 'id'])
    # for mask_path_orig, filepath, sampl_id in zip(mask_paths[0], mask_paths[1], mask_paths[2]):
    # temp_view = view.clone(persistent=False)
    image_width, image_height = size
    # sample_ids_op = []
    # sample_res_mask = []
    # sample_confidence = []
    # temp_view = fo.Dataset.from_dict(view.to_dict())
    # temp_view = view.clone()
    for_loop = time.time()

    def process_sample(sample, dt_field, image_width, image_height):
        sample_id = sample.label_image_id
        mask_path_orig = sample[dt_field]['mask_path']

        # mask = cv2.imread(mask_path_orig)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)[:, :, 0]
        mask = cv2.imread(mask_path_orig)[:, :, 2]
        confidence = np.mean(mask / 255)
        resized_mask = cv2.resize(
            mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST,
        )
        # resized_mask[resized_mask <= 127] = 0
        # resized_mask[resized_mask > 127] = 1
        resized_mask[resized_mask > 0] = 1

        return sample_id, resized_mask, float(np.round(confidence, 2))

    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(process_sample)
        (
            sample, dt_field,
            image_width, image_height,
        )
        for sample in temp_view
    )
    sample_ids_op, sample_res_mask, sample_confidence = zip(*results)
    sample_ids_op = list(sample_ids_op)
    sample_res_mask = list(sample_res_mask)
    sample_confidence = list(sample_confidence)
    # for sample in temp_view:
    #     # sample=temp_view[sample_id]
    #     sample_ids_op.append(sample.label_image_id)
    #     # image_width, image_height = size

    #     # Get the mask resolution from the mask file
    #     mask_path_orig = sample[dt_field]['mask_path']
    #     # imageio.v2.imread(mask_path_orig)
    #     mask = cv2.imread(mask_path_orig)
    #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)[:, :, 0]
    #     confidence = np.mean(mask / 255)
    #     resized_mask = cv2.resize(
    #         mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST,
    #     )
    #     resized_mask[resized_mask <= 127] = 0
    #     resized_mask[resized_mask > 127] = 1
    #     # This is the only way to avoid creating doubled polylines
    #     # Using default segmenttopolyline from voxel will create double polyline

    #     # del sample[dt_field]['mask_path']
    #     #sample[dt_field]['mask'] = resized_mask
    #     sample_res_mask.append(resized_mask)

    #     #sample[dt_field]['confidence'] = float(np.round(confidence, 2))
    #     sample_confidence.append(float(np.round(confidence, 2)))
    #     # sample_pols.append(fo.Detections(detections=sample[dt_field]))
    #     #samples_coll.append(sample)
    print('Time elapsed for for loop of dt ' + str(time.time() - for_loop))

    # set_values = time.time()
    # temp_view.set_values(
    #     dt_field+'.mask', dict(
    #         zip(sample_ids_op, sample_res_mask),
    #     ), key_field='label_image_id', validate=False,
    # )
    # temp_view.set_values(
    #     dt_field+'.confidence', dict(
    #         zip(sample_ids_op, sample_confidence),
    #     ), key_field='label_image_id', validate=False,
    # )
    # print('Time elapsed for set values ' + str(time.time() - set_values))
    dict1 = dict(zip(sample_ids_op, sample_confidence))  # dict 1
    seg_to_pol = time.time()
    values = gen_polyline_modular(
        {1: 'blocked'}, sample_ids_op, sample_res_mask,
    )
    temp_view.set_values(dt_field+'_pol', values, key_field='label_image_id')
    # segmentations_to_polylines(
    #     temp_view, dt_field, dt_field+'_pol', mask_targets={1: 'blocked'},
    #     mask_types='stuff', tolerance=2, progress=False,
    # )
    print('Time take for seg to pol ' + str(time.time() - seg_to_pol))

    # merge_samples = time.time()
    # view._dataset.merge_samples(  # pylint: disable=W0212
    #     temp_view, key_field='label_image_id', fields=[dt_field+'_pol'],
    # )
    # print('Time take for merge samples ' + str(time.time() - merge_samples))
#    fo.delete_non_persistent_datasets()

    set_conf_mask = time.time()
    llid, mask_path, polid = temp_view.values(
        [
            'label_image_id', dt_field +
            '.mask_path',  dt_field+'_pol.polylines.id',
        ],
    )
    # dict1 = dict(zip(llid, conf))
    dict2 = dict(zip(llid, polid))
    new_dict = {
        dict2[llid][0]: conf for llid, conf in dict1.items(
        ) if llid in dict2 and len(dict2[llid]) == 1
    }

    temp_view.set_label_values(
        dt_field+'_pol'+'.polylines.confidence', new_dict, validate=False,
    )
    dict1 = dict(zip(llid, mask_path))
    new_dict = {
        dict2[llid][0]: conf for llid, conf in dict1.items(
        ) if llid in dict2 and len(dict2[llid]) == 1
    }
    temp_view.set_label_values(
        dt_field+'_pol'+'.polylines.mask_path', new_dict, validate=False,
    )
    temp_view.set_label_values(
        dt_field+'_pol'+'.polylines.filled', {key: False for key in new_dict}, validate=False,
    )
    print('Time take for set conf mask ' + str(time.time() - set_conf_mask))


def dt_mask_generator_autogt(temp_view, dt_field, gt_field):
    """
    Generates detection mask such that the 255 in the GT mask is ignored also in the detection mask
    255 is the vehicle itself and if auto gt makes a detection there, its necessary to ignore it

    Args:
        view (fiftyone.core.view.DatasetView): The view containing the samples with images and masks.
        dt_field (str): The name of the field in the sample that contains the detection mask.
        gt_field (str): The name of the field in the sample that contains the ground truth mask.

    Returns:
        None. The function updates the detection mask in the sample based on the ground truth mask.
    """

    # temp_view = view.clone(persistent=False)
    sample_ids_op = []
    sample_res_mask = []
    sample_confidence = []
    # temp_view = view.clone()  # fo.Dataset.from_dict(view.to_dict())

    def process_sample(sample, dt_field, gt_field):
        sample_id = sample.label_image_id
        mask_path_orig_dt = sample[dt_field]['mask_path']
        mask_path_orig_gt = sample[gt_field]['mask_path']

        # mask_dt = cv2.imread(mask_path_orig_dt)
        # mask_dt = cv2.cvtColor(mask_dt, cv2.COLOR_BGR2RGB)[:, :, 0]
        mask_dt = cv2.imread(mask_path_orig_dt)[:, :, 0]

        mask_gt = cv2.imread(mask_path_orig_gt)
        mask_gt = mask_gt[:, :, 0]

        mask_dt[mask_gt == 255] = 0

        return sample_id, mask_dt, 1.0
    for_loop = time.time()
    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(process_sample)
        (sample, dt_field, gt_field)
        for sample in temp_view
    )
    print('Time take for for loop ' + str(time.time() - for_loop))
    sample_ids_op, sample_res_mask, sample_confidence = zip(*results)

    sample_ids_op = list(sample_ids_op)
    sample_res_mask = list(sample_res_mask)
    sample_confidence = list(sample_confidence)

    # set_values = time.time()
    # temp_view.set_values(
    #     dt_field+'.mask', dict(
    #         zip(sample_ids_op, sample_res_mask),
    #     ), key_field='label_image_id', validate=False,
    # )
    # temp_view.set_values(
    #     dt_field+'.confidence', dict(
    #         zip(sample_ids_op, sample_confidence),
    #     ), key_field='label_image_id', validate=False,
    # )
    dict1 = dict(zip(sample_ids_op, sample_confidence))
    # print('Time elapsed for set values ' + str(time.time() - set_values))
    seg_to_pol = time.time()
    values = gen_polyline_modular(
        {255: 'blocked'}, sample_ids_op, sample_res_mask,
    )
    temp_view.set_values(dt_field+'_pol', values, key_field='label_image_id')

    # segmentations_to_polylines(
    #     temp_view, dt_field, dt_field+'_pol', mask_targets={255: 'blocked'},
    #     mask_types='stuff', tolerance=2, progress=False,
    # )
    print('Time take for seg to pol ' + str(time.time() - seg_to_pol))

    # merge_samples = time.time()
    # view._dataset.merge_samples(  # pylint: disable=W0212
    #     temp_view, key_field='label_image_id', fields=[dt_field+'_pol'],
    # )
#    fo.delete_non_persistent_datasets()
    # print('Time take for merge samples ' + str(time.time() - merge_samples))
    set_conf_mask = time.time()
    llid, mask_path, polid = temp_view.values(
        [
            'label_image_id', dt_field +
            '.mask_path',  dt_field+'_pol.polylines.id',
        ],
    )
    # dict1 = dict(zip(llid, conf))
    dict2 = dict(zip(llid, polid))
    new_dict = {
        dict2[llid][0]: conf for llid, conf in dict1.items(
        ) if llid in dict2 and len(dict2[llid]) == 1
    }

    temp_view.set_label_values(
        dt_field+'_pol'+'.polylines.confidence', new_dict, validate=False,
    )
    dict1 = dict(zip(llid, mask_path))
    new_dict = {
        dict2[llid][0]: conf for llid, conf in dict1.items(
        ) if llid in dict2 and len(dict2[llid]) == 1
    }
    temp_view.set_label_values(
        dt_field+'_pol'+'.polylines.mask_path', new_dict, validate=False,
    )
    temp_view.set_label_values(
        dt_field+'_pol'+'.polylines.filled', {key: False for key in new_dict}, validate=False,
    )
    print('Time take for set conf mask ' + str(time.time() - set_conf_mask))


def dont_correct_mask_resolution(temp_view, gt_field):
    """
    Corrects the GT pixel mapping based on the blockage requirements.
    But does not rescale the mask to the image resolution.
    0-16 is not blocked
    255 invalid
    17-37 blocked

    Args:
        sample (fiftyone.core.sample.Sample): The sample containing the image and mask.
        gt_field (str): The name of the field in the sample that contains the mask.
        mask_path (str, optional): The path where the resized mask will be saved. Defaults to "/tmp/reshaped_mask.png".

    Returns:
        None. The function saves the resized mask to `mask_path`.
    """

    # temp_view = view.clone()  # fo.Dataset.from_dict(view.to_dict())
    # sample_res_mask = []
    # sample_ids_op = []
    # blockage_centre_x_coll =[]
    # blockage_centre_y_coll =[]
    # blockage_percent_coll =[]

    def process_sample(sample, gt_field):
        sample_id = sample.label_image_id
        mask_path_orig = sample[gt_field]['mask_path']

        # mask = cv2.imread(mask_path_orig)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)[:, :, 0]
        mask = cv2.imread(mask_path_orig)[:, :, 0]
        height, width = mask.shape
        mask[mask == 255] = 0
        mask[mask < 17] = 0
        mask[mask >= 17] = 1

        blockage_percent = np.count_nonzero(mask) / np.size(mask)
        non_zero_indices = np.transpose(np.nonzero(mask))
        center = np.mean(non_zero_indices, axis=0)
        if math.isnan(center[1]):
            center[1] = 0
        if math.isnan(center[0]):
            center[0] = 0
        normalized_x = center[1] / width
        normalized_y = center[0] / height

        return sample_id, mask, blockage_percent, normalized_x, normalized_y

    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(process_sample)
        (sample, gt_field)
        for sample in temp_view
    )

    sample_ids_op, sample_res_mask, blockage_percent_coll, \
        blockage_centre_x_coll, blockage_centre_y_coll = zip(
            *results,
        )
    sample_ids_op = list(sample_ids_op)
    sample_res_mask = list(sample_res_mask)
    blockage_percent_coll = list(blockage_percent_coll)
    blockage_centre_x_coll = list(blockage_centre_x_coll)
    blockage_centre_y_coll = list(blockage_centre_y_coll)
    # for sample in temp_view:
    #     sample_ids_op.append(sample.label_image_id)
    #     # image_width, image_height = size

    #     # Get the mask resolution from the mask file
    #     mask_path_orig = sample[gt_field]['mask_path']
    #     # imageio.v2.imread(mask_path_orig)
    #     mask = cv2.imread(mask_path_orig)[:, :, 0]
    #     mask[mask == 255] = 0
    #     mask[mask < 17] = 0
    #     mask[mask >= 17] = 1
    #     sample_res_mask.append(mask)

    #     blockage_percent = np.count_nonzero(mask,
    #                                         ) / np.size(mask)
    #     blockage_percent_coll.append(blockage_percent)
    #     non_zero_indices = np.transpose(np.nonzero(mask))
    #     center = np.mean(non_zero_indices, axis=0)
    #     if math.isnan(center[1]):
    #         center[1] = 0
    #     if math.isnan(center[0]):
    #         center[0] = 0
    #     blockage_centre_x_coll.append(center[1])
    #     blockage_centre_y_coll.append(center[0])

    # del sample[gt_field]['mask_path']
    # sample[gt_field]['mask'] = mask
    # temp_view.set_values(
    #     gt_field+'.mask', dict(
    #         zip(sample_ids_op, sample_res_mask),
    #     ), key_field='label_image_id', validate=False,
    # )

    # values = gen_polyline_modular(gt_field, temp_view, mask_targets = {1: 'blocked'})
    values = gen_polyline_modular(
        {1: 'blocked'}, sample_ids_op, sample_res_mask,
    )
    temp_view.set_values(gt_field+'_pol', values, key_field='label_image_id')

    # segmentations_to_polylines(
    #     temp_view, gt_field, gt_field+'_pol', mask_targets={1: 'blocked'},
    #     mask_types='stuff', tolerance=2, progress=False,
    # )
    # view._dataset.merge_samples(  # pylint: disable=W0212
    #     temp_view, key_field='label_image_id', fields=[gt_field+'_pol'],
    # )

    # create a dict that maps lid:polid
    # map pol id to blockage_percent
    # Now you have lid:blockage_percent
    llid,  polid, mask_path = temp_view.values(
        [
            'label_image_id', gt_field+'_pol.polylines.id', gt_field+'.mask_path',
        ],
    )
    dict2 = dict(zip(llid, polid))
    for var_name, var_value in zip(
        [
            'blockage_percent', 'blockage_centre_x',
            'blockage_centre_y', 'mask_path',
        ],
        [
            blockage_percent_coll, blockage_centre_x_coll,
            blockage_centre_y_coll, mask_path,
        ],
    ):

        dict1 = dict(zip(sample_ids_op, var_value))

        new_dict = {
            dict2[llid][0]: blkper for llid, blkper in dict1.items(
            ) if llid in dict2 and len(dict2[llid]) == 1
        }
        temp_view.set_label_values(
            gt_field+'_pol'+'.polylines.' + var_name, new_dict, validate=False,
        )
    temp_view.set_label_values(
        gt_field+'_pol'+'.polylines.filled', {key: False for key in new_dict}, validate=False,
    )


def correct_mask_resolution(temp_view, gt_field, size=(1280, 1024)):
    """
    Corrects the resolution of a mask to match the resolution of its corresponding image.
    Applies only for 8x10 low_resolution use case GT masks.

    Args:
        sample (fiftyone.core.sample.Sample): The sample containing the image and mask.
        gt_field (str): The name of the field in the sample that contains the mask.
        mask_path (str, optional): The path where the resized mask will be saved.
        Defaults to "/tmp/reshaped_mask.png".

    Returns:
        None. The function saves the resized mask to `mask_path`.
    """

    # temp_view = view.clone()  # fo.Dataset.from_dict(view.to_dict())
    # sample_res_mask = []
    # sample_ids_op = []
    # blockage_centre_x_coll =[]
    # blockage_centre_y_coll =[]
    # blockage_percent_coll =[]

    image_width, image_height = size

    def process_sample(sample, image_width, image_height, gt_field):
        sample_id = sample.label_image_id
        mask_path_orig = sample[gt_field]['mask_path']
        # mask = cv2.imread(mask_path_orig)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)[:, :, 0]
        mask = cv2.imread(mask_path_orig)[:, :, 0]
        resized_mask = cv2.resize(
            mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST,
        )
        blockage_percent = np.count_nonzero(mask) / np.size(mask)
        non_zero_indices = np.transpose(np.nonzero(resized_mask))
        center = np.mean(non_zero_indices, axis=0)
        if math.isnan(center[1]):
            center[1] = 0
        if math.isnan(center[0]):
            center[0] = 0
        normalized_x = center[1] / image_width
        normalized_y = center[0] / image_height
        return sample_id, resized_mask, blockage_percent, normalized_x, normalized_y

    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(process_sample)
        (
            sample, image_width,
            image_height, gt_field,
        )
        for sample in temp_view
    )

    sample_ids_op, sample_res_mask, blockage_percent_coll, blockage_centre_x_coll, blockage_centre_y_coll = zip(
        *results,
    )

    sample_ids_op = list(sample_ids_op)
    sample_res_mask = list(sample_res_mask)
    blockage_percent_coll = list(blockage_percent_coll)
    blockage_centre_x_coll = list(blockage_centre_x_coll)
    blockage_centre_y_coll = list(blockage_centre_y_coll)
    # for sample in temp_view:
    #     sample_ids_op.append(sample.label_image_id)

    #     # Get the mask resolution from the mask file
    #     mask_path_orig = sample[gt_field]['mask_path']
    #     # imageio.v2.imread(mask_path_orig)
    #     mask = cv2.imread(mask_path_orig)[:, :, 0]
    #     # mask_height, mask_width = mask.shape[:2]

    #     # Check if the mask resolution matches the image resolution
    #     # if mask_width != image_width or mask_height != image_height:
    #     # Resize the mask to match the image resolution
    #     resized_mask = cv2.resize(
    #         mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST,
    #     )
    #     sample_res_mask.append(resized_mask)
    #     blockage_percent = np.count_nonzero(mask,
    #                                         ) / np.size(mask)
    #     blockage_percent_coll.append(blockage_percent)
    #     non_zero_indices = np.transpose(np.nonzero(resized_mask))
    #     center = np.mean(non_zero_indices, axis=0)
    #     if math.isnan(center[1]):
    #         center[1] = 0
    #     if math.isnan(center[0]):
    #         center[0] = 0
    #     blockage_centre_x_coll.append(center[1])
    #     blockage_centre_y_coll.append(center[0])

    # temp_view.set_values(
    #     gt_field+'.mask', dict(
    #         zip(sample_ids_op, sample_res_mask),
    #     ), key_field='label_image_id', validate=False,
    # )

    # values = gen_polyline_modular(gt_field, temp_view, mask_targets = {1: 'blocked'})
    values = gen_polyline_modular(
        {1: 'blocked'}, sample_ids_op, sample_res_mask,
    )
    temp_view.set_values(gt_field+'_pol', values, key_field='label_image_id')

    # segmentations_to_polylines(
    #     temp_view, gt_field, gt_field+'_pol', mask_targets={1: 'blocked'},
    #     mask_types='stuff', tolerance=2, progress=False,
    # )
    # view._dataset.merge_samples(  # pylint: disable=W0212
    #     temp_view, key_field='label_image_id', fields=[gt_field+'_pol'],
    # )

    # create a dict that maps lid:polid
    # map pol id to blockage_percent
    # Now you have lid:blockage_percent
    llid,  polid, mask_path = temp_view.values(
        [
            'label_image_id', gt_field+'_pol.polylines.id', gt_field+'.mask_path',
        ],
    )
    dict2 = dict(zip(llid, polid))
    for var_name, var_value in zip(
        [
            'blockage_percent', 'blockage_centre_x',
            'blockage_centre_y', 'mask_path',
        ],
        [
            blockage_percent_coll, blockage_centre_x_coll,
            blockage_centre_y_coll, mask_path,
        ],
    ):

        dict1 = dict(zip(sample_ids_op, var_value))

        new_dict = {
            dict2[llid][0]: blkper for llid, blkper in dict1.items(
            ) if llid in dict2 and len(dict2[llid]) == 1
        }
        temp_view.set_label_values(
            gt_field+'_pol'+'.polylines.' + var_name, new_dict, validate=False,
        )

    temp_view.set_label_values(
        gt_field+'_pol'+'.polylines.filled', {key: False for key in new_dict}, validate=False,
    )


def batch_ops(gt_field, temp_ds, batch, gt_res):
    """
    Performs operations on a batch view of a dataset.

    Args:
        gt_field (str): The ground truth field name.
        temp_ds (fiftyone.core.dataset.Dataset): The temporary dataset.
        batch (list): A list of sample IDs for the batch.

    This function performs the following operations on the batch view of the dataset:
    1. Corrects the mask resolution.
    2. Converts segmentations to polylines.

    Note: This function modifies the input dataset in-place.
    """

    batch_view = temp_ds.match(fo.ViewField('label_image_id').is_in(batch))
    # in blockage the GT is 8*10, this is not handled well in the fiftyone, so we resize it and extract the polylines
    # mapping_gt = {1: 'blocked'}
    if gt_res == 'low_res':
        correct_mask_resolution(
            batch_view, gt_field,
        )
    elif gt_res == 'high_res':
        dont_correct_mask_resolution(
            batch_view, gt_field,
        )
    else:
        raise ValueError(
            'Invalid value for gt_res. Must be either "low_res" or "high_res"',
        )
    # segmentations_to_polylines(
    #     batch_view, gt_field, gt_field+'_pol', mask_targets=mapping_gt,
    #     mask_types='stuff', tolerance=2, progress=False,
    # )


def custom_vars_add(gt_field, sample_fields_as_metadata, temp_ds, batch):
    """
    Adds custom variables to a batch view of a dataset.

    Args:
        gt_field (str): The ground truth field name.
        dt_field (str): The detection field name.
        sample_fields_as_metadata (list): A list of sample fields to be added as metadata.
        temp_ds (fiftyone.core.dataset.Dataset): The temporary dataset.
        batch (list): A list of sample IDs for the batch.

    This function updates the metadata and confidence of the samples in the batch
    view based on the ground truth and detection fields. The updated metadata includes
    the sample fields specified in `sample_fields_as_metadata` and the ground truth polyline data.

    Note: This function modifies the input dataset in-place.
    """
    batch_view = temp_ds.match(fo.ViewField('label_image_id').is_in(batch))
    update_meta_pol(
        batch_view, gt_field,
        sample_fields_as_metadata,
    )


def segment_to_polylines(
    dataset, gt_field, dt_field, batch_size=50,
    sample_fields_as_metadata=None, gt_res='low_res', update_samples=False,
):
    """Convert the ground truth and prediction fields from segmentaion to
     polylines for the given voxel51 dataset.

    Args:
        dataset(51 dataset): dataset with gt and dt segmantation fields.
        gt_field (string): Ground truth field
        models (list): List of models
        gt_attrs (dict):  Ground truth attributes
        conf_mask_is_rgb (bool):  Is this a 3-channel RGB prediction mask
        sample_fields_as_metadata (set): Set of fields to be added as metadata in gt_dt
        gt_res (str): Resolution of the ground truth mask (high_res:auto_gt/low_res:Viper)
    """
    if update_samples:
        view_wo_gtdt = dataset.match({'gt_dt': None})
        all_pol_fields = [model + '_pol' for model in dt_field]
        # update_samples is False because you create a view manually already above
        _, pol_gen_fields = fields_check(view_wo_gtdt, all_pol_fields, False)
        dataset = view_wo_gtdt
    else:
        all_pol_fields = [model + '_pol' for model in dt_field]
        # update_samples is False because you create a view manually already above
        _, pol_gen_fields = fields_check(dataset, all_pol_fields, False)
    if len(pol_gen_fields) > 0:
        pol_gen_fields = [field[:-4] for field in pol_gen_fields]
    # pol_gen_fields = []
    # for model in dt_field:
    #     if not dataset.has_field(model + '_pol'):
    #         pol_gen_fields.append(model)

    # if pol_gen_fields:  # and gt_res == 'low_res':
        # we will create polylines on the cloned dataset
        # temp_ds = dataset#dataset.clone(persistent=False)
    # elif pol_gen_fields and gt_res == 'high_res':
    #    raise ValueError('In high resolution case, you are expected to use the off-the-shelf \
    #                     segment to polyline conversion from fiftyone for the detection field.')

    ids = dataset.values('label_image_id')
    batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    for model in pol_gen_fields:
        print(f'{model} conversion to polylines')
        # dataset.clone_sample_field(
        #     model, model+'_resh',
        # )
        for batch in tqdm(batches):
            batch_view = dataset.match(
                fo.ViewField('label_image_id').is_in(batch),
            )
            if gt_res == 'low_res':
                dt_mask_generator(batch_view, model)

            elif gt_res == 'high_res':
                dt_mask_generator_autogt(batch_view, model, gt_field)

    # if pol_gen_fields:
        # for model in pol_gen_fields:
        #    dataset.delete_sample_field(model+'_resh')
        # dataset.merge_samples(temp_ds, key_field='label_image_id')
        # fo.delete_dataset(temp_ds.name)

    gt_presence, _ = fields_check(dataset, [gt_field + '_pol'], False)
    if not gt_presence:
        # ids = dataset.values('label_image_id')
        # batches = np.array_split(ids, len(ids) // batch_size + 1)
        # temp_ds = dataset.clone(persistent=False)
        # dataset.clone_sample_field(
        #    gt_field, gt_field+'_resh',
        # )
        for batch in tqdm(batches):
            batch_ops(gt_field, dataset, batch, gt_res)
        # Parallel(n_jobs=3, prefer="threads")(
        #     delayed(batch_ops)(gt_field, dataset, batch, gt_res)
        #     for batch in tqdm(batches)
        # )

        for batch in tqdm(batches):
            custom_vars_add(
                gt_field, sample_fields_as_metadata, dataset, batch,
            )

        # dataset.delete_sample_field(gt_field+'_resh')
        # dataset.merge_samples(temp_ds, key_field='label_image_id')
        # fo.delete_dataset(temp_ds.name)


def aggregate_iou_per_slice(
    samples,
    modelName,
):  # pylint: disable=C0103
    """
    Computes the custom mean iou for a given model  for a set of samples.

    Args:
        samples (FiftyOne View or Dataset):
            The FiftyOne dataset or view containing samples to evaluate.

        modelName (str):
            The name of the model whose mean ious are calculated.

    Returns:
        float:
            The mean IOU score, representing the model's performance .
            The score is averaged across all samples in the dataset or view .
    """
    iou_values = samples.values('gt_dt.polylines.'+modelName+'_iou')
    iou_values = [lst for lst in iou_values if lst]
    flattened_iou = [
        item for sublist in iou_values for item in sublist if item != -1.0
    ]
    print(f'Calculating iou score for {modelName}')
    if len(flattened_iou) == 0:
        return 0.0
    return np.round(safe_div(sum(flattened_iou), len(samples)), 2)
