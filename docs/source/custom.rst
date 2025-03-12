Custom functions for users
===========================
In some cases the customer requested to perform some operations on our side (XAI) as this needed only for XAI.
Hence we provide these custom utils also as a part of our package.

Enabling xai on ISS dataset
---------------------------

To set up and enable `xaipostprocess` on the ISS dataset, you can use the following configuration and example code. This demonstrates how to configure and process the ISS dataset with a custom setup. Here is an example:

::

    import fiftyone as fo

    # Load the ISS dataset from the specified directory
    dataset = fo.Dataset.from_dir(
        dataset_dir="/mnt/ai_testing/07_temp/export_xai/",
        dataset_type=fo.types.FiftyOneDataset,
        name="iss_live_new",
    )
    dataset.save()
Next, initialize the `Xai` class to configure the evaluation using custom metrics.

::

    from xaipostprocess.main import Xai
    from xaipostprocess.utils.custom_ims.getmapscore import get_custom_map

    # Define dataset properties and configuration
    dataset_type = "keypoints"
    dataset_name = "iss_live_new"
    models = ['pred_pgt_keypoints_bodypose2d_keypoints_d42d8efc9e254bf1af7429e0db7de75b']
    gt_field = "gt_pgt_keypoints_bodypose2d_keypoints_455ea56cf98b49d39114b6919b573c63"
    gt_attrs = {
        "mandatorypropvars": {"x_coordinate", "y_coordinate"},
    }

    # Initialize the Xai class with the custom metrics
    init_cls = Xai(
        dataset_type=dataset_type,
        dataset_name=dataset_name,
        models=models,
        gt_field=gt_field,
        gt_attrs=gt_attrs,
        scenario_custom_metrics={'map_score': get_custom_map}
    )
Additionally, configure the parameters for custom body pose evaluations.

::

    params_addl = {
        'bodypose_gt': "gt",
        "bodypose_dt": ["pred"],
        "add_missing_fields": True,
        "label_field": "pgt_keypoints_bodypose2d",
        "batch_size": 5,
        'verbose': None,
    }

Finally, instantiate the evaluation process for the `iss_bodypose` use case:

::

    init_cls.instantiate_from_scratch(
        use_case='iss_bodypose', params_addl=params_addl,
    )



DAT3 blockage use case
----------------------
In case of blockage the ground truth and predictions are in semantic segments.
Additionally, the segments are in the form of a three chanel png image, its from this image the red channels are to be taken as the inferences.
The ground truth and inferences will be converted as polylines, these are used for
gt_dt generation and finally these fields are removed by our clear operations.
However, please note that the evaluations are still made based on the masks and not on
the basis of the pipelines.
Here is an example on how to perform this on the legacy DT__cool_tree_ctd6v6sk6z__POLY2D__blockage model:

::

    from xaipostprocess.main import Xai
    import fiftyone as fo
    import os
    import time
    dataset_name = 'non_leg_ss'
    try: fo.delete_dataset(dataset_name)
    except: print("Dataset does not exist")
    fo.Dataset.from_dir(
        dataset_dir='/mnt/ai_testing/02_datasets/ford/blockage/non_leg_ss',
        dataset_type=fo.types.FiftyOneDataset,
        name=dataset_name, persistent=True, overwrite=True,
    )

    ds= fo.load_dataset(dataset_name)
    # In the local testing we did not download all masks, in this case if a mask is missing, voxel51 does not render the image on frontend.
    # Hence, we deleted locally the fields which are not necessary and mapped the paths to our local subset.
    # Not relevant for blockage team/DYPER51 team. Relevant only for XAI team developers to do local test.
    ds.delete_sample_fields(["DT__plum_plum_sxtzyxnvq3__POLY2D__blockage", "DT__honest_pea_ptk5kqr0m4__POLY2D__blockage", "DT__clever_napa_9vldtnq6kz__POLY2D__blockage",
                            "M_GT_blockage_segmentation", "M_GT_blockage_segmentation_16x20", "M_GT_blockage_segmentation_32x40", "M_GT_blockage_segmentation_64x80"])
    for s in ds:
        s["filepath"] = os.path.join("/mnt/ai_testing/02_datasets/ford/blockage/rectified", os.path.basename(s["filepath"]))
        s["M_GT_blockage_segmentation_8x10"]["mask_path"]= os.path.join("/mnt/ai_testing/02_datasets/ford/blockage/non_leg_ss_gt", s["M_GT_blockage_segmentation_8x10"]["mask_path"].split("/")[-2],
                                                                        os.path.basename(s["M_GT_blockage_segmentation_8x10"]["mask_path"]))

        if s["DT__tidy_house_b85wrnbmvf__POLY2D__blockage"]["mask_path"] is not None:
            s["DT__tidy_house_b85wrnbmvf__POLY2D__blockage"]["mask_path"] = os.path.join("/mnt/ai_testing/02_datasets/ford/blockage/rectified_tidyhouse",
                                                                                            os.path.basename(s["DT__tidy_house_b85wrnbmvf__POLY2D__blockage"]["mask_path"]))
        del s["thumbnail_path"]
        s.save()
    ds.persistent = True

    # Rename the fields to the model names without double underscore
    # Relvant only for DYPER51 team and XAI team developers
    models_orig = ['DT__tidy_house_b85wrnbmvf__POLY2D__blockage']
    models = [m.replace("__", "_") for m in models_orig]
    for old_field, new_field in zip(models_orig, models):
        ds.rename_sample_field(old_field, new_field)

    tic = time.time()
    sample_fields_as_metadata = ["roadType", "timeOfDay", "forward_sel", "random_all", "gen_algo_mid", "weatherSky",
                                "blockageCondition", "weatherPrecipitation", "lightCondition"]
    del_field_for_st = ["forward_sel", "gen_algo_mid", "random_all"]
    gt_attrs = {
        'fvprops': {
            'blockage_percent', 'blockage_centre_y',
            'blockage_centre_x',
        },
    }
    params_addl ={"iou_prop": 0.5,
                "default_classes": ["blocked"],
                "sample_fields_as_metadata": sample_fields_as_metadata,
                "del_field_for_st": del_field_for_st,
                "batch_size": 50,
                "manipulate_mask_pth": True, # For databricks
                "manipulate_from": "/mnt/", # For databricks
                "manipulate_to": "/dbfs/mnt/", # For databricks
                "gt_res": "high_res", # high_res: 1280, 1024, low_res = 8x10
                "parse_ious": True,
                }

    init_cls = Xai(dataset_type = 'polylines', dataset_name = dataset_name,
                models = models,
                gt_field = 'M_GT_blockage_segmentation_8x10', gt_attrs = gt_attrs)
    init_cls.instantiate_from_scratch(use_case="blockage", params_addl=params_addl)
    print("Time taken to process the dataset: ", time.time()-tic)

Blockage Viper use case employs detection in the form of confidence mask. The confidence mask is converted to polylines based on the red channel threshold.
That is, the red channel of the confidence mask is taken and all pixels greater than 127 is treated as a blockage.
To be used to draw a polyline for visualization and for IoU computation.
The polylines for a model, example, model_pol, is created only if that field does not exist.

::

    from xaipostprocess.gtdt.blockageops import segment_to_polylines
    models = ["DT_purple_line_4fpd7ww47t_segmentation", "DT_ivory_napkin_z2fyj6m8w9_segmentation",
             "DT_shy_house_2lxq0py20h_segmentation", "DT_nifty_worm_rx0mg1hysk_segmentation",
             "DT_keen_soca_mvh48ksj7y_segmentation", "DT_epic_brake_h1cqnnjhwd_segmentation"]
    gt_field = "M_GT_blockage_segmentation_8x10"
    segment_to_polylines(dataset, gt_field, dt_field=models, batch_size=5, sample_fields_as_metadata=None, gt_res='low_res',)

There are cases where there is a new model and used want to update the xai fields, without rerunning the evalutaion for all the other models.
If the user wants to update only particular model/models in the dataset without rerunning the evaluation of the rest of the other model, in ``instantiate_from_scratch()`` method, the
`update_models` parameter can be used by setting its value to the list of models we need update in the dataset.

::

    init_cls = Xai(dataset_type = 'polylines', dataset_name = dataset_name,
                models = models,
                gt_field = 'M_GT_blockage_segmentation_8x10', gt_attrs = gt_attrs)

    models_pol = [m+'_pol' for m in models]
    dataset_blockage = fo.load_dataset(dataset_name)
    # incase you want to automatically detect the new models for which evaluation is absent
    models_to_update = list(set(models_pol) - set(dataset_blockage.list_evaluations()))
    init_cls.instantiate_from_scratch(use_case="blockage", params_addl=params_addl, update_models=models_to_update)

For enabling mean iou in slice teller, use the ``postprocess_custommetrics()`` function from `xaipostprocess`. This method takes in the dictionary `scenario_custom_metrics`
as argument where key is the name of the custom metric that you need to add to sliceteller and it takes in a function that calculates the custom metric as value.

::

    # importing the custom metric function defined for getting the mean iou
    from xaipostprocess.gtdt.blockageops import aggregate_iou_per_slice
    from xaipostprocess.slicefinder.custommetricsslices import postprocess_custommetrics

    scenario_custom_metrics = {'mean_iou': aggregate_iou_per_slice}
    #scenario_custom_metrics (dict): A dictionary of custom metric  and the values are the functions that compute the metrics.
    #dataset_name (str): The name of the dataset to be processed.
    #gt_dt_field (str): The field in the dataset where gt_dt data is stored most likely "gt_dt".
    postprocess_custommetrics(
        scenario_custom_metrics,
        dataset_name, gt_dt_field='gt_dt',
    )



Integrating custom  metrics into XAI back/front-end
----------------------------------------------------

By following these steps, you can integrate any custom metrics into XAI, allowing for enhanced evaluation of model performance.
For this, you need to ensure that both frontend and backend updated appropriately.
Custom metrics are to be integrated at two levels:
a) When a user adds a query/scenario to the scenario tab: In this case the custom metric is calculated on the fly.
b) When a user runs the slice teller: For each slice the custom metric is precalculated and stored into the database for rendering in the slice teller visualization.

Changes in frontend (for realizing (a))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use an example for our ISS use-case where we need to integrate the custom mAP score.

Step 1: Define Custom Metrics for example ``get_custom_map()``.
In the v3d-fo-plugin folder, add your custom metric definition, e.g., ``def get_custom_map()`` into a python file.
In this example we have ``get_custom_map()`` as a part of xaipostprocess package.
Now, when a user selects a query from the distribution on the frontend and adds to the scenario bucket,
this query is treated as a fiftyone view. This view is passed to the backend.


Step 2: Import the defined custom metric into __init__.py.
Kindly ignore this if you are a user, because this is handled already by XAI team for ISS and other projects.
That is if the dataset is keypoint we trigger map score computation for scenario else we return the accuracy and support.

.. code-block:: python

    from xaipostprocess.utils.custom_ims.getmapscore import get_custom_map
    class ScenarioMetrics(foo.Operator):
        @property
        def config(self):
            return foo.OperatorConfig(name="scenario", label="Scenario")

        def execute(self, ctx):
            accuracy = {}
            map_score = {}

            for modelName in ctx.params['modelNames']:
                accuracy[modelName] = getAccuracy(ctx.params['filteredRows'], modelName)
                map_score[modelName] = get_custom_map(samples=ctx, modelName=modelName, is_ctx=True, map_exists=True)
            support = getSupport(ctx.params['filteredRows'], ctx.params['ndxSize'])
            return {'accuracy': accuracy, 'support': support, "map_score": map_score}


Changes in Backend  (for realizing (b))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step 1: Integrate custom metrics into your dataset by a similar function (example shown for IMS) that appends metric fields to each image. This allows enhanced runtime (for getting scores for filters).
.. code-block:: python

    from xaipostprocess.utils.custom_ims.custom_map import add_custom_map

    add_custom_map(
        dataset_name,
        groupby_field="chunk,
        labeling_job='baseline',
        inference_job='ims-bodypose-v156-20240418-233121',
    )

Step 2: Use the Custom Metric in Post-Processing After running the slice teller, integrate the custom metrics in post-processing using the postprocess_custommetrics function which takes following arguments:
::

    from xaipostprocess.slicefinder.custommetricsslices import postprocess_custommetrics
    from xaipostprocess.utils.custom_ims.getmapscore import get_custom_map
    scenario_custom_metrics = {'map_score': get_custom_map}
    #scenario_custom_metrics (dict): A dictionary of custom metric functions and the values are the functions that compute the metrics.
    #dataset_name (str): The name of the dataset to be processed.
    #gt_dt_field (str): The field in the dataset where gt_dt data is stored most likely "gt_dt".
    postprocess_custommetrics(
        scenario_custom_metrics,
        dataset_name, gt_dt_field='gt_dt',
    )

Integrating Sample-Level Metrics into XAI Back/Front-End
---------------------------------------------------------

By including sample-level metrics in `gt_attrs` and processing `DSConfig`, the XAI plugin is updated to incorporate these metrics at the sample level.
To add a plugins with sample-level metrics, follow the steps outlined below.

Changes in Frontend
~~~~~~~~~~~~~~~~~~~~

No direct changes are required in the frontend for sample-level metrics.
A new plugin, `v3d-samples`, includes the distribution of sample-level metrics.

Changes in Backend
~~~~~~~~~~~~~~~~~~~

We use an example for our ISS use-case where `sample-level metrics` need to be integrated into the XAI system.

Step 1: Add Sample-Level Metrics to `gt_attrs`
Sample-level metrics must be included in the `gt_attrs` dictionary under the key `"sample_level_metrics"`.

For example:
::

    gt_attrs["sample_level_metrics"] = {f"map_image_{field}" for field in generated_inference_jobs_fields}

In this case, `generated_inference_jobs_fields` is a list of inference job fields, e.g., `['imsbodyposev15620240418233121']`.


Step 2: Process `DSConfig` with Updated `gt_attrs`
Once the sample-level metrics are added to `gt_attrs`, call the `process_dsconfig` function to update the dataset configuration with these metrics.

For example:
::

    process_dsconfig(dataset_type, dataset_name, generated_inference_jobs_fields, gt_attrs)

PACE TL use case
---------------

Here is an example of how to perform the xaipostprocess for PACE traffic light usecase in order to populate the required fields in the dataset for enabling the XAI plugin.
Before running the postprocessing part we need to ensure the evaluation is performed on the dataset with the model name as the eval key.

::

    from xaipostprocess.main import Xai
    dataset = fo.load_dataset("tl_pace_test")
    dataset.default_classes= ["traffic_light_housing", "traffic_light_bulb"]
    dstype = '2d'
    gtfield = 'ground_truth'
    det_type = 'detections'
    models = ['calm_boot_w6ssd7g4f4_predictions']
    dsname = "tl_pace_test"
    gt_attrs = {
            'mandatoryprops': {'xmin': 'box_left', 'ymin': 'box_top'},
            'custompropvars': {
                'bulb_count',
                'direction',
                'occlusion',
                'stacking',
                'status',
                'truncation',
                'inlay',
                'relevance_driven_path',
                'target_road_user',
                'is_bounding_box_estimated',
                'is_warning_only',
                'is_decipherable',
            },
    }
    config = Xai(
                dataset_type=dstype,
                dataset_name=dsname,
                models=models,
                gt_field=gtfield,
                gt_attrs=gt_attrs,
                min_sup=0.05,
                max_combo=3,
            )
    config.instantiate_from_scratch()
