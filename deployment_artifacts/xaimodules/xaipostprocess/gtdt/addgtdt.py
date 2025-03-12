"""
This module is responsible for linking ground truth and prediction in voxel51 dataset.
Every ground truth (gt) is mapped to every detection (dt) and likewise every detection is
also mapped to each other. This is a mandatory field needed for the XAI plugins to work.
Mapping of gt to dt enables in identify true positive, false positive, true negative and false negative.
But in addition to this the detections of different detectors are also mapped to each other.
Example: When model 1 is predicting a FP with respect to gt and model 2 is not making
the same false positive, then the detection of model 1 is treated as a true negative for model 2.
That is, there is no wolf (ground truth) and the shepherd (model 2) also predicts there is no wolf, is a true negative.
This is how a detection of one model is mapped to another.
Likewise, if a box is FP or FN or TN, the IoUs are assigned to -1, this is also taken care by this module.
"""
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
import fiftyone as fo
from fiftyone import ViewField as F
from xaipostprocess.gtdt.gtdt2d import gtdt2d
from xaipostprocess.gtdt.gtdt3d import gtdt3d
from xaipostprocess.gtdt.gtdtkeypoints import gtdtkeypoints
from xaipostprocess.gtdt.gtdtpolylines import gtdtpolylines
from xaipostprocess.utils.logging import setup_logging
from xaipostprocess.utils.merge import common_prefix
from xaipostprocess.utils.merge import has_nested_list
#from xaipostprocess.utils.merge import merge_detections

def calculate_gtdt(
    dataset_type, dataset_name, models, gt_field,
    gt_attrs, gt_dt_field='gt_dt', iou=0.4, prop_methods=None, fill=None,
    batch_size=5, update_samples=False, update_models=None, verbose=False,
):
    """Calculate the gt_dt (ground truth detection) field for the given voxel51 dataset.

    Args:
        dataset_type (str): Type of the dataset (e.g., classification, detection).
        dataset_name (str): Name of the dataset.
        models (list): List of models to be evaluated.
        gt_field (str): Name of the ground truth field.
        gt_attrs (dict): Ground truth attributes as key-value pairs.
        gt_dt_field (str, optional): Name of the field to store ground truth detection results. Defaults to 'gt_dt'.
        iou (float, optional): Intersection over Union (IoU) threshold for matching ground truth and detections.
        prop_methods (set, optional): Set of methods for calculating derived fields. Defaults to None.
        fill (any, optional): Placeholder or fill value for missing data. Defaults to None.
        batch_size (int, optional): Batch size for processing data. Defaults to 5.
        update_samples (bool, optional): If True, processes new samples in the dataset that
            have not yet been evaluated. Defaults to False.
        update_models (list, optional): List of specific models that should be updated. If None,
            all provided models are updated. Defaults to None.
        verbose (bool, optional): If True, enables verbose logging.

    Raises:
        ValueError: If any required parameter is invalid or computation fails.

    Returns:
        None: This function updates the dataset in place and does not return a value.
    """
    # Configure logging
    logger, log_filename = setup_logging(verbose, 'gtdt')

    logger.info(f"Starting GTDT calculation for dataset '{dataset_name}'")
    logger.debug(f'Using models: {models}')
    logger.debug(f'Ground truth field: {gt_field}')
    logger.debug(f'Ground truth attributes: {gt_attrs}')

    default_2d = {
        'height': lambda dt, img_h, img_w: dt['bounding_box'][3] * img_h,
        'xmin': lambda dt, img_h, img_w: dt['bounding_box'][0],
        'ymin': lambda dt, img_h, img_w: dt['bounding_box'][1],
        'width': lambda dt, img_h, img_w: dt['bounding_box'][2] * img_w,
        'area': lambda dt, img_h, img_w: (dt['bounding_box'][3] * img_h)
        * (dt['bounding_box'][2] * img_w),
        'aspect_ratio': lambda dt, img_h, img_w: dt['bounding_box'][2]
        / dt['bounding_box'][3],
    }
    default_3d = {
        'xmin': lambda dt, img_h, img_w: dt[gt_attrs['mandatoryprops']['xmin']],
        'ymin': lambda dt, img_h, img_w: dt[gt_attrs['mandatoryprops']['ymin']],
        'area': lambda dt, img_h, img_w: dt[gt_attrs['mandatoryprops']['area']],
    }
    default_keypoints = {
        'x_coordinate': lambda dt, : dt['points'][0][0],
        'y_coordinate': lambda dt, : dt['points'][0][1],
    }

    default_poylines = {
        'min_x': lambda dt, : min(point[0] for point in dt['points'][0]),
        'max_x': lambda dt, : max(point[0] for point in dt['points'][0]),
        'min_y': lambda dt, : min(point[1] for point in dt['points'][0]),
        'max_y': lambda dt, : max(point[1] for point in dt['points'][0]),
    }

    # Use the provided prop_methods or fall back to defaults based on dataset_type
    if prop_methods is None:
        if dataset_type == '2d':
            prop_methods = default_2d
        elif dataset_type == '3d':
            prop_methods = default_3d
        elif dataset_type == 'keypoints':
            prop_methods = default_keypoints
        elif dataset_type == 'polylines':
            prop_methods = default_poylines
        else:
            raise ValueError('Invalid dataset_type')
    dataset = fo.load_dataset(dataset_name)
    if update_samples:
        dataset = dataset.match({'gt_dt': None})
        if len(dataset) == 0:
            print('No new samples. Will not run gt_dt in update_samples True ')
            return
    match = F('label').is_in(dataset.default_classes)
    ds_view_no_unsure = dataset.filter_labels(gt_field, match)
    # Combine muliple models
    if has_nested_list(models) is True:
        combined_model_fields = []
        to_be_removed = []
        combine_models = [item for item in models if isinstance(item, list)]

        for dts in combine_models:
            combined_model = common_prefix(dts)
            combined_model_fields.append(combined_model)
            merge_detections(dataset_name, dts, combined_model)
            logger.debug(f'Combined models: {dts} into {combined_model}')

            # Collect models to be removed
            to_be_removed.extend(dts)

        # Remove models from the 'models' list
        models = [
            item for sublist in models for item in (
                sublist if isinstance(sublist, list) else [sublist]
            )
        ]
        models = [elem for elem in models if elem not in to_be_removed]
        models += combined_model_fields
        logger.info(f'Models used for gtdt calculation: {models}')
        # Print the updated list of models
        # print(f'Models used for gtdt calculation: {models}')
        # dataset = fo.load_dataset(dataset_name)
        # match = F('label').is_in(dataset.default_classes)
        # ds_view_no_unsure = dataset.filter_labels(gt_field, match)
        for model in models:
            ds_view_no_unsure.evaluate_detections(
                pred_field=model,
                gt_field=gt_field,
                eval_key=model,
                # compute_mAP=True,
                classwise=False,
                iou=iou,
            )
            ds_view_no_unsure = ds_view_no_unsure.set_field(
                model+'.detections.'+model,
                (F(model+'_id') == '').if_else(
                    'fp',
                    F(model),
                ),
            )

            ds_view_no_unsure = ds_view_no_unsure.set_field(
                model+'_fp',
                F(model+'.detections').filter(F(model) == 'fp').length(),
            )
            ds_view_no_unsure.save()
        logger.info('Evaluating detections complete')

    # dataset = fo.load_dataset(dataset_name)
    # match = F('label').is_in(dataset.default_classes)
    # ds_view_no_unsure = dataset.filter_labels(gt_field, match)
    if dataset_type == '2d':
        if len(ds_view_no_unsure) != len(dataset):
            # Handle no Gt & Dt
            clean_samp = ds_view_no_unsure.filter_labels(
                gt_field, F(
                    'label',
                ).is_in(ds_view_no_unsure.default_classes),
            ).first()[gt_field]
            clean_samp = clean_samp['detections'][0]
            ds_view_no_unsure = dataset[0:len(dataset)]
        else:
            clean_samp = None
        gtdt2d(
            gt_attrs, gt_dt_field, ds_view_no_unsure, gt_field,
            models, prop_methods, clean_samp=clean_samp, dataset=dataset, batch_size=batch_size,
            update_models=update_models, verbose=verbose,
        )
    if dataset_type == '3d':
        if len(ds_view_no_unsure) != len(dataset):
            # Handle no Gt & Dt
            clean_samp = ds_view_no_unsure.filter_labels(
                gt_field, F(
                    'label',
                ).is_in(ds_view_no_unsure.default_classes),
            ).first()[gt_field]
            clean_samp = clean_samp['polylines'][0]
            ds_view_no_unsure = dataset[0:len(dataset)]
        else:
            clean_samp = None
        gtdt3d(
            gt_attrs, gt_dt_field, ds_view_no_unsure,
            gt_field, models, prop_methods, clean_samp, dataset=dataset, batch_size=batch_size,
            update_models=update_models, verbose=verbose,
        )

    if dataset_type == 'keypoints':
        if len(ds_view_no_unsure) != len(dataset):
            # Handle no Gt & Dt
            clean_samp = ds_view_no_unsure.filter_labels(
                gt_field, F(
                    'label',
                ).is_in(ds_view_no_unsure.default_classes),
            ).first()[gt_field]
            clean_samp = clean_samp['keypoints'][0]
            ds_view_no_unsure = dataset[0:len(dataset)]
        else:
            clean_samp = None
        gtdtkeypoints(
            gt_attrs, gt_dt_field, ds_view_no_unsure,
            gt_field, models, prop_methods, clean_samp, dataset, batch_size=batch_size,
            update_models=update_models, verbose=verbose,
        )

    if dataset_type == 'polylines':
        if len(ds_view_no_unsure) != len(dataset):
            clean_samp = ds_view_no_unsure.filter_labels(
                gt_field, F(
                    'label',
                ).is_in(ds_view_no_unsure.default_classes),
            ).first()[gt_field]
            clean_samp = clean_samp['polylines'][0]
            ds_view_no_unsure = dataset[0:len(dataset)]
        else:
            clean_samp = None
        gtdtpolylines(
            gt_attrs, gt_dt_field, ds_view_no_unsure,
            gt_field, models, prop_methods, fill, clean_samp, dataset, batch_size=batch_size,
            update_models=update_models, verbose=verbose,
        )
    # if batch_size is not None:
    #     for i in range(0, len(sample_ids), batch_size):
    #         batch_sample_ids = sample_ids[i: i + batch_size]
    #         batch_gt_dt = sample_gt_dts[i: i + batch_size]
    #         dataset.set_values(
    #             gt_dt_field, dict(
    #                 zip(batch_sample_ids, batch_gt_dt),
    #             ), key_field='id',
    #         )
    #         dataset.save()
    # else:
    #     dataset.set_values(
    #         gt_dt_field, dict(
    #             zip(sample_ids, sample_gt_dts),
    #         ), key_field='id',
    #     )
    #     dataset.save()
    # dataset.add_dynamic_sample_fields()
    # dataset.persistent = True
    logger.info('GTDT calculation completed!')
    logger.info(f'Logs saved to {log_filename}')
    # logger.info('Finished calculate_gtdt')
    for handler in logger.handlers:
        handler.flush()
        handler.close()
