"""
This module is responsible for linking Polylines-ground truth and Polylines-prediction in voxel51 dataset.
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

# Setup default logging
default_logger, default_log_filename = setup_logging(
    False, 'gtdt',
)


def gtdtpolylines_sw(
    gtattrlist, gt_attrs, sample, gt_field,
    models, prop_methods, fill, clean_samp=None,
    update_models=None, logger=default_logger,
):  # pylint: disable=R0915, R0912, R0914, R0913
    """Calculate the gtdt field for the given voxel51 Polylines sample.

    Args:
        gtattrlist (list):  Ground truth attributes as a list (auto generated from the dict and passed)
        gt_attrs (dict):  Ground truth attributes as dict
        ds_view_no_unsure (Fiftyone Dataset View): Filtered Dataset View
        gt_field (string): Ground truth field
        models (list): List of models
        prop_methods (set, optional): Calculation of derived fields.
        update_models (list, optional): List of specific models that should be updated. If None,
            all provided models are updated. Defaults to None.
        logger (logging.Logger, optional): Logger for logging information.
            Defaults to default_logger.

    Raises:
        ValueError: gtdt
    """
    # with fo.ProgressBar() as progress_bar:
    #     for sample in progress_bar(ds_view_no_unsure):  # pylint: disable=R1702
    try:
        logger.debug(f'Processing sample ID: {sample.id}')
        if 'width' in dir(sample):
            img_w = sample.width  # pylint: disable=W0612
        else:
            img_w = (    # noqa
                sample.metadata.width * 1.0 if
                hasattr(sample, 'metadata') and
                hasattr(sample.metadata, 'width')
                else None
            )

        if 'height' in dir(sample):
            img_h = sample.height  # pylint: disable=W0612
        else:
            img_h = sample.metadata.height * \
                1.0 if hasattr(sample, 'metadata') and hasattr(  # noqa
                    sample.metadata, 'height',
                ) else None
        gt_dt = []
        for ground_truth in sample[gt_field].polylines:
            # filled, closed = False, False
            if fill is not None:
                filled = fill
                closed = ground_truth['closed']
            elif hasattr(ground_truth, 'filled') and hasattr(ground_truth, 'closed'):
                filled = ground_truth['filled']
                closed = ground_truth['closed']
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
                    dts = sample[model].polylines
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
                            ] = det['confidence']
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
                    # if restrict_to_model is not none and attrs[model + '_iou']
                    # has a non negative value dont run attrs[model + '_iou']
                    # else run attrs[model + '_iou'] = -1.0
                    attrs[model + '_label'] = 'Background'
                    attrs[model + '_confidence'] = -1.0
                    logger.debug(
                        f'Attributes for False Negative: {attrs}',
                    )
            # confidence = attrs[model + '_confidence']#*******
            gt_dt.append(
                fo.Polyline(
                    label=ground_truth['label'],
                    points=ground_truth['points'],
                    confidence=ground_truth['confidence'],
                    closed=closed,
                    filled=filled,
                    **attrs,
                ),
            )
        for model in models:  # **************
            logger.debug(
                f'Checking false positives for model: {model}',
            )
            logger.debug(
                f'Checking false positives for model: {model} for sample ID: {sample.id}',
            )
            if sample[model] is None:
                continue
            dts = sample[model].polylines
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
                    attrs[model + '_confidence'] = det['confidence']
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
                                closed = clean_samp['closed']
                                filled = clean_samp['filled']

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
                        fo.Polyline(
                            label='Background',
                            points=det['points'],
                            confidence=-1.0,
                            closed=closed,
                            filled=filled,
                            **attrs,
                        ),
                    )
                    logger.debug(
                        f'Attributes for False Positive: {attrs}',
                    )
        # sample[gt_dt_field] = fo.Polylines(polylines=gt_dt)
        # sample.save()
        sample_gtdt = fo.Polylines(polylines=gt_dt)
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
    return sample_gtdt


def gtdtpolylines(
    gt_attrs, gt_dt_field, ds_view_no_unsure, gt_field,
    models, prop_methods, fill=False, clean_samp=None, dataset=None, batch_size=5,
    update_models=None, verbose=False,
):  # pylint: disable=R0915, R0912, R0914, R0913
    """Calculate the gtdt field for the given FiftyOne Polylines dataset.

    Args:
        gt_attrs (dict): Ground truth attributes.
        gt_dt_field (str): Ground truth-detection field name.
        ds_view_no_unsure (fiftyone.core.view.DatasetView): Filtered dataset view without "unsure" labels.
        gt_field (str): Ground truth field name.
        models (list): List of models to be applied.
        prop_methods (set, optional): Methods for calculating derived fields. Defaults to None.
        fill (bool, optional): Whether to fill missing data. Defaults to False.
        clean_samp (any, optional): Custom cleaning samples parameter. Defaults to None.
        dataset (any, optional): Dataset to process. Defaults to None.
        batch_size (int, optional): Batch size for processing. Defaults to 5.
        update_models (list, optional): List of specific models that should be updated. If None,
            all provided models are updated. Defaults to None.
        verbose (bool, optional): If True, enables verbose logging. Defaults to False.

    Raises:
        ValueError: Raised when invalid values are encountered in the gtdt process.

    Returns:
        None
    """
    # Setup logging
    logger, log_filename = setup_logging(verbose, 'gtdt')

    logger.info('Starting gtdtpolylines')
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
    # dataset_inds = np.arange(0,len(ds_view_no_unsure))
    # batches = [dataset_inds[i:i + batch_size] for i_ind,i in enumerate(range(0, len(dataset_inds), batch_size))]
    # batches=[[batch[0], batch[-1]] for batch in batches if len(batch) > 1]

    # print(gtattrlist)
    # gt_dt_dict = {}
    sample_ids = ds_view_no_unsure.values('id')
    for i in range(0, len(sample_ids), batch_size):
        batch_ids = sample_ids[i:i + batch_size]
        batch_view = ds_view_no_unsure.select(batch_ids)
        sample_ids_op = []
        sample_gt_dts = []
        # with fo.ProgressBar() as progress_bar:
        for sample in batch_view.iter_samples():  # pylint: disable=R1702
            sample_gtdt = gtdtpolylines_sw(
                gtattrlist, gt_attrs, sample, gt_field,
                models, prop_methods, fill, clean_samp, update_models, logger,
            )
            sample_gt_dts.append(sample_gtdt)
            sample_ids_op.append(sample.id)

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
    logger.info('Finished gtdtpolylines')
    logger.info(f'Logs saved to {log_filename}')

    # Flush and close log handlers to ensure logs are written to the file
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    return sample_ids, sample_gt_dts
