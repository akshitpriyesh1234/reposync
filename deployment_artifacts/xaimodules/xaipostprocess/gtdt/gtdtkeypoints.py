"""
This module is responsible for linking Keypoints-ground truth and Keypoints-prediction in voxel51 dataset.
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
from xaipostprocess.gtdt.gtdtsupport import custom_tostd
from xaipostprocess.gtdt.gtdtsupport import get_default_value
from xaipostprocess.utils.logging import setup_logging


def gtdtkeypoints(
    gt_attrs, gt_dt_field, ds_view_no_unsure, gt_field, models,
    prop_methods, clean_samp=None, dataset=None, batch_size=5, update_models=None, verbose=False,
):  # pylint: disable=R0912, R0915, R0914, R0913
    """Calculate the gtdt field for the given voxel51 Keypoints dataset.

    Args:
        gt_attrs (dict): Ground truth attributes to be utilized in calculations.
        gt_dt_field (str): Name of the field in the dataset representing ground-truth-to-detection relationships.
        ds_view_no_unsure (fiftyone.core.view.DatasetView): Filtered view of the dataset excluding "unsure" samples.
        gt_field (str): Name of the field in the dataset containing ground truth keypoints.
        models (list): List of models to be used for processing.
        prop_methods (set, optional): Methods or metrics for calculating derived fields.
        clean_samp (optional): Strategy or function to handle cases with missing ground truth and detections.
        dataset (fiftyone.core.dataset.Dataset, optional): The dataset object to operate on.
        batch_size (int, optional): Number of samples to process in each batch. Defaults to 5.
        update_models (list, optional): List of specific models that should be updated. If None,
            all provided models are updated. Defaults to None.
        verbose (bool, optional): If True, enables detailed logging for debugging.

    Raises:
        ValueError: gtdt
    """
    # Setup logging
    logger, log_filename = setup_logging(verbose, 'gtdt')

    logger.info('Starting gtdtkeypoints')
    logger.debug(f'User inputs: {locals()}')
    gtattrlist = {'label'}
    all_dict = {}
    for eachkey in gt_attrs.keys():
        if isinstance(gt_attrs[eachkey], dict):
            all_dict.update(gt_attrs[eachkey])
        if isinstance(gt_attrs[eachkey], set):
            gt_attrs[eachkey] = {item: item for item in gt_attrs[eachkey]}
            all_dict.update(gt_attrs[eachkey])
    for eachkey_sub in all_dict.values():
        gtattrlist.add(eachkey_sub)
    logger.debug(f'Ground truth attributes list: {gtattrlist}')
    # print(gtattrlist)
    # gt_dt_dict = {}
    sample_ids = ds_view_no_unsure.values('id')
    for i in range(0, len(sample_ids), batch_size):  # pylint: disable=R1702
        batch_ids = sample_ids[i:i + batch_size]
        batch_view = ds_view_no_unsure.select(batch_ids)
        sample_ids_op = []
        sample_gt_dts = []
        for sample in batch_view.iter_samples():  # pylint: disable=R1702
            try:
                logger.debug(f'Processing sample ID: {sample.id}')
                gt_dt = []
                for ground_truth in sample[gt_field].keypoints:
                    attrs = {}

                    # get gt attrs
                    for gt_attr in gtattrlist:
                        attrs[gt_attr] = ground_truth[gt_attr]
                    attrs['class'] = attrs.pop('label')
                    logger.debug(
                        f'Ground truth attributes for detection: {attrs}',
                    )
                    for model in models:
                        logger.debug(f'Processing model: {model}')
                        logger.debug(
                            f'Processing model: {model} for sample ID: {sample.id}',
                        )
                        if model + '_id' not in ground_truth:
                            continue
                        if ground_truth[model + '_id'] != '':
                            if sample[model] is None:
                                continue
                            dts = sample[model].keypoints
                            for det in dts:
                                if det.id == ground_truth[model + '_id']:
                                    attrs['gt_id'] = ground_truth.id
                                    attrs[model] = ground_truth[model]
                                    attrs[
                                        model +
                                        '_id'
                                    ] = ground_truth[model + '_id']
                                    attrs[
                                        model +
                                        '_iou'
                                    ] = ground_truth[model + '_iou']
                                    attrs[model + '_label'] = det['label']
                                    attrs[
                                        model +
                                        '_confidence'
                                    ] = det['confidence'][0]
                                    logger.debug(
                                        f'Attributes for True Positive: {attrs}',
                                    )
                        else:
                            attrs['gt_id'] = ground_truth.id
                            attrs[model] = ground_truth[model]
                            attrs[model + '_id'] = ground_truth[model + '_id']
                            # attrs[model + '_iou'] = -1.0
                            if update_models is not None and model not in update_models:
                                attrs[model + '_iou'] = \
                                    [
                                        p for p in sample['gt_dt'].polylines
                                    if p['gt_id'] == ground_truth.id
                                ][0][model + '_iou']
                            else:
                                attrs[model + '_iou'] = -1.0
                            attrs[model + '_label'] = 'Background'
                            attrs[model + '_confidence'] = -1.0
                            logger.debug(
                                f'Attributes for False Negative: {attrs}',
                            )
                    # confidence = attrs[model + '_confidence']

                    gt_dt.append(
                        fo.Keypoint(
                            label=ground_truth['label'],
                            points=ground_truth['points'],
                            # confidence=[confidence],
                            confidence=ground_truth['confidence'],
                            **attrs,
                        ),
                    )

                for model in models:
                    logger.debug(
                        f'Checking false positives for model: {model}',
                    )
                    logger.debug(
                        f'Checking false positives for model: {model} for sample ID: {sample.id}',
                    )
                    if sample[model] is None:
                        continue
                    dts = sample[model].keypoints
                    for det in dts:
                        if model + '_id' not in det:
                            continue
                        # if det[model + '_id'] == '':
                        if det[model] == 'fp' or det[model + '_id'] == '':
                            attrs = {}
                            attrs['gt_id'] = ''
                            attrs[model] = det[model]
                            attrs[model + '_id'] = det.id
                            attrs[model + '_iou'] = -1.0
                            attrs[model + '_label'] = det['label']
                            attrs[model + '_confidence'] = det['confidence'][0]
                            for prop_key in gtattrlist:
                                if prop_key == 'label':
                                    continue  # Skip processing the "label" attribute
                                if (
                                    custom_tostd(gt_attrs, prop_key)
                                    in prop_methods
                                ):
                                    translate_tostandard = custom_tostd(
                                        gt_attrs, prop_key,
                                    )
                                    try:
                                        attrs[prop_key] = prop_methods[
                                            translate_tostandard
                                        ](ground_truth)  # pylint: disable=W0631
                                    except NameError:
                                        attrs[prop_key] = prop_methods[
                                            translate_tostandard
                                        ](clean_samp)
                                else:
                                    try:
                                        attrs[prop_key] = get_default_value(
                                            prop_key,
                                            ground_truth,    # pylint: disable=W0631
                                        )
                                    except NameError:
                                        attrs[prop_key] = get_default_value(
                                            prop_key,
                                            clean_samp,    # pylint: disable=W0631

                                        )
                                # print(attrs[prop_key])

                            for other_model in models:
                                if other_model != model:
                                    attrs['gt_id'] = ''
                                    attrs[other_model] = 'tn'
                                    attrs[other_model + '_id'] = ''
                                    attrs[other_model + '_iou'] = -1.0
                                    attrs[
                                        other_model +
                                        '_label'
                                    ] = 'Background'
                                    attrs[
                                        other_model +
                                        '_confidence'
                                    ] = -1.0

                            gt_dt.append(
                                fo.Keypoint(
                                    label='Background',
                                    points=det['points'],
                                    confidence=[-1.0],
                                    **attrs,
                                ),
                            )
                            logger.debug(
                                f'Attributes for False Positive: {attrs}',
                            )

                # sample[gt_dt_field] = fo.Keypoints(keypoints=gt_dt)
                # gt_dt_dict[sample.id] = fo.Keypoints(keypoints=gt_dt)
                sample_ids_op.append(sample.id)
                sample_gt_dts.append(fo.Keypoints(keypoints=gt_dt))
                # sample.save()
                logger.debug(f'Successfully processed sample ID: {sample.id}')
            except KeyError as e:
                logger.error(
                    f'Error processing sample ID: {sample.id}',
                )
                logger.debug(f'Current sample: {sample}')
                raise e  # Raise the exception after logging

            except UnboundLocalError as e:
                logger.error(f'UnboundLocalError: {e}')
                logger.debug(f'Current sample: {sample}')
                raise e  # Raise the exception after logging

            except Exception as e:
                logger.error(
                    f'Error processing sample ID: {sample.id}', exc_info=True,
                )
                logger.debug(f'Current sample: {sample}')
                raise e  # Raise the exception after logging

            # Log statements after processing the sample, before moving to the next iteration
            logger.debug('Sample processing completed successfully')

        dataset.set_values(
            gt_dt_field, dict(
                zip(sample_ids_op, sample_gt_dts),
            ), key_field='id',
        )
    # dataset.save()
    # dataset.add_dynamic_sample_fields()
    # dataset.persistent = True
    try:
        dataset.add_dynamic_sample_fields()
    except AttributeError:
        dataset._dataset.add_dynamic_sample_fields()  # pylint: disable=W0212

    logger.info('Finished gtdtkeypoints')
    logger.info(f'Logs saved to {log_filename}')

    # Flush and close log handlers to ensure logs are written to the file
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    return sample_ids, sample_gt_dts
