Usage
=====
Learn how to leverage the xaipostprocess package to enhance the functionality of standard v3d-fo-plugin and integrate additional addons seamlessly.


Standard v3d-fo-plugin
-----------------------

Create gt_dt
~~~~~~~~~~~~~~

To create the `gt_dt` field, you can use the `calculate_gtdt` function from the `xaipostprocess.gtdt.addgtdt` module. Here's an example:

::

   from xaipostprocess.gtdt.addgtdt import calculate_gtdt
   # Define the arguments
   dataset_type = "2d"  # Replace with your actual dataset type
   dataset_name = "my_dataset"  # Replace with your actual dataset name
   models = ["model1", "model2"]  # Replace with the list of your model names
   gt_field = "ground_truth"  # Replace with your ground truth field name
   gt_attrs = {
      "mandatorypropvars": {
         "xmin": "left",
         "ymin": "top"
      },
      "derivedpropsvars": {
         "size",
         "width",
         "height",
         "aspect_ratio"
      },
      "otherderivedpropsvars": {
         "area": "size"
      }
   }  # Replace with your ground truth attributes
   iou = 0.5  # Replace with your desired IoU threshold (needed when combining output of models predicting different objects)
   prop_methods = {
      'height': lambda dt, img_h, img_w: dt['bounding_box'][3] * img_h,
      'xmin': lambda dt, img_h, img_w: dt['bounding_box'][0],
      'ymin': lambda dt, img_h, img_w: dt['bounding_box'][1],
      'width': lambda dt, img_h, img_w: dt['bounding_box'][2] * img_w,
      'area': lambda dt, img_h, img_w: (dt['bounding_box'][3] * img_h) * (dt['bounding_box'][2] * img_w),
      'aspect_ratio': lambda dt, img_h, img_w: dt['bounding_box'][2] / dt['bounding_box'][3],
   }  # Add your own function if deriving fields from model output
   verbose (bool, optional): If True, enables verbose logging. Defaults to False.

   # Call the function to calculate gt_dt
   # Use for custom derived values or combining multiple models
   calculate_gtdt(dataset_type, dataset_name, models, gt_field, gt_attrs, iou, prop_methods, verbose=False)

   # Call the function for standard cases
   calculate_gtdt(dataset_type, dataset_name, models, gt_field, gt_attrs, verbose=False)

More about creating `prop_method` field:
-----------------------------------------

The `prop_method` field is used to create the derived fields for detections, which are present in gt_attrs.
Let us take a false positive detection.
In this case, there are meta properties like xmin and ymin that can anyway be derived from the predicted
bounding box.

::

   'xmin': lambda dt, img_h, img_w: dt['bounding_box'][0]

On the other hand, we cannot derive properties like occlusion or truncated from the predicted bounding box. The detector
may not provide these information.
Hence for cases where you can derive, we can use the `prop_method` field with the definition of how to derive the field.
In other cases, where the derived properties are not present in `prop_method` it means that those properties cant be derived
from a false positive predicted bounding box.
Hence, if it's a `string`, the value will be "none".
For `numerical` values, the value will be "-1".
The primary condition is that the derived field (e.g., xmin) should be present in the ground_truth.
If its in an alternate name, you can map it in the `gt_attrs` field. Example: {"xmin": "left"}.
To have this field for your `detections`, you need to derive it from your detections bounding box with the `function`:


Create dsconfig
~~~~~~~~~~~~~~~~
To create the `dsconfig` on MongoDB, you can use the `process_dsconfig` function from the `xaipostprocess.dsconfig.autodsconfig` module. Here's an example:


::

   from xaipostprocess.dsconfig.autodsconfig import process_dsconfig
   # Define the arguments
   dataset_type = "2d"  # Replace with your actual dataset type
   dataset_name = "my_dataset"  # Replace with your actual dataset name
   models = ["model1", "model2"]  # Replace with the list of your model names
   gt_attrs = {
      "mandatorypropvars": {
         "xmin": "left",
         "ymin": "top"
      },
      "derivedpropsvars": {
         "size",
         "width",
         "height",
         "aspect_ratio"
      },
      "otherderivedpropsvars": {
         "area": "size"
   }  # Replace with your ground truth attributes
   # Call the function to add dsconfig
   process_dsconfig(dataset_type, dataset_name, models, gt_attrs)

SliceTeller
~~~~~~~~~~~
To add `slices`, you can use the `compute_slicefinding` function from the `xaipostprocess.slicefinder.slicefinding` module. Here's an example:


::

   from xaipostprocess.slicefinder.slicefinding import compute_slicefinding
   # Define the arguments
   dataset_name = "my_dataset"  # Replace with your actual dataset name
   models = ["model1", "model2"] # Replace with the list of your model names
   gt_attrs = {
      "mandatorypropvars": {
         "xmin": "left",
         "ymin": "top"
      },
      "derivedpropsvars": {
         "size",
         "width",
         "height",
         "aspect_ratio"
      },
      "otherderivedpropsvars": {
         "area": "size"
   }  # Replace with your ground truth attributes
   verbose (bool, optional): If True, enables verbose logging. Defaults to False.
   # Call the function to compute slices
   compute_slicefinding(dataset_name, models, min_sup=0.05, max_combo=3, features=gt_attrs)

Xai Main Class
~~~~~~~~~~~~~~~~~
The `Xai` class serves as the `main` entry point for post-processing methods in the `xaipostprocess` package. It offers flexibility for various use cases in handling and updating datasets for XAI. Below are different scenarios and their respective methods:

Basic Initialization
---------------------
To initialize the `Xai` class, you need to define the dataset, ground truth fields, models, and attributes:

   ::

      from xaipostprocess.main import Xai
      # Define the arguments
      dataset_name = "my_dataset"  # Replace with your actual dataset name
      gt_field = "ground_truth"  # Replace with your ground truth field name
      models = ["model1", "model2"] # Replace with the list of your model names
      gt_attrs = {
         "mandatorypropvars": {
            "xmin": "left",
            "ymin": "top"
         },
         "derivedpropsvars": {
            "size",
            "width",
            "height",
            "aspect_ratio"
         },
         "otherderivedpropsvars": {
            "area": "size"
      }  # Replace with your ground truth attributes
      # Call the function to compute slices
      initial_class = Xai(dataset_type, dataset_name, models, gt_field, gt_attrs)

Use Cases and Examples
-----------------------

**Case 1**: Initialize XAI for the first time
If a user wants to run XAI on a dataset that has no XAI-relevant fields (e.g., a new dataset):

.. code-block:: python

    initial_class.instantiate_from_scratch(
        use_case='others',
        update_samples=False,
        params_addl=None,
        update_models=None
    )

**Case 2**: Update new samples in the dataset
If a user has already set up XAI but added new samples (e.g., new images) to the dataset:

.. code-block:: python

    initial_class.instantiate_from_scratch(
        use_case='others',
        update_samples=True,
        params_addl=None,
        update_models=None
    )

**Case 3**: Update new models
If a user wants to add and process new models:

.. code-block:: python

    initial_class.instantiate_from_scratch(
        use_case='others',
        update_samples=False,
        params_addl=None,
        update_models=["new_model"]
    )

**Case 4**: Update both new Samples and new models simultaneously
If a user needs to handle both new samples and new models, the `update_new_samples_new_models` wrapper method simplifies the process:

.. code-block:: python

    initial_class.update_new_samples_new_models(
        use_case='others',
        update_samples=True,
        params_addl=None,
        update_models=["new_model"]
    )
Key Notes for methods of `Xai` class :
----------
1. **`params_addl`**: Use this argument to define additional processing configurations, such as IoU thresholds, metadata fields, and paths for manipulating masks.
2. **Order of Operations**: Ensure new samples are processed before models when using `instantiate_from_scratch`.


Xai Main Class For Blockage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `Xai` class serves as the `main` entry point for post-processing methods in the `xaipostprocess` package. It offers flexibility for various use cases in handling and updating datasets for XAI. Below are different scenarios and their respective methods:

   **preporcessing step 1**: In order to ensure that model names are compatible with FiftyOne datasets, it is important to standardize the usage of underscores in the field names. FiftyOne datasets can only handle single underscores in model names, and any double underscores `(__)` should be converted to single underscores `(_)` before working with the dataset.
   Suppose a user has a dataset and wants to perform XAI analysis, but the model names contain double underscores. Here is the procedure to follow:
   ::
      ds = fo.load_dataset(dataset_name)
      models_orig = ['DT__tidy_house_b85wrnbmvf__POLY2D__blockage']
      models = [m.replace("__", "_") for m in models_orig]
      for old_field, new_field in zip(models_orig, models):
         ds.rename_sample_field(old_field, new_field)
      ds.save()

   **preporcessing step 2**: To efficiently manage blockage operations on a FiftyOne dataset, a structured post-processing workflow is implemented. Key metadata fields and attributes relevant to the dataset, such as `roadType`, `timeOfDay`, and `blockageCondition`, are defined to guide the processing. Additionally, parameters like the `IoU` , `threshold` and `batch size` are set to control the operation's specifics.
   The dataset is then prepared by loading it and splitting its IDs into batches, allowing for parallel processing. A temporary dataset is created by cloning the original dataset to make intermediate modifications without affecting the original data. One key operation involves cloning a sample field, such as `M_GT_blockage_segmentation_8x10`, and appending a reshaped version to the temporary dataset.
   Next, batch operations are performed in parallel using joblib, applying predefined blockage operations to each batch. Following this, custom variables are added to each batch, enhancing the dataset with additional metadata fields. Once the batch operations are complete, the temporary reshaped field is deleted, and the processed data from the temporary dataset is merged back into the original dataset, keyed on the `label_image_id`. Finally, any necessary adjustments are made to specific fields, such as `blockage_percent` and `blockage_centre_x/y` fields, ensuring the data is ready for further analysis. The dataset is then saved, with persistence enabled to store the changes permanently. This workflow ensures that the dataset is efficiently processed while maintaining data integrity.
   ::
      from joblib import Parallel, delayed
      from xaipostprocess.gtdt.blockageops import batch_ops, custom_vars_add
      from tqdm import tqdm
      import numpy as np
      sample_fields_as_metadata = ["roadType", "timeOfDay", "forward_sel", "random_all", "gen_algo_mid", "weatherSky",
                                 "blockageCondition", "weatherPrecipitation", "lightCondition"]
      del_field_for_st = ["forward_sel"]
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
                  "polyline_gt": "M_GT_blockage_segmentation_8x10_pol",
                  "manipulate_mask_pth": False,
                  "manipulate_from": "/mnt/",
                  "manipulate_to": "/dbfs/mnt/",
                  "restore_manipulations": True,}
      batch_size=50
      ids = ds.values('label_image_id')
      batches = np.array_split(ids, len(ids) // batch_size + 1)
      dt_field = models
      gt_field = 'M_GT_blockage_segmentation_8x10'
      temp_ds = ds.clone(persistent=False)
      temp_ds.clone_sample_field(
         gt_field, gt_field+'_resh'
         )
      Parallel(n_jobs=-1)(delayed(batch_ops)(gt_field, temp_ds, batch) for batch in tqdm(batches))
      for batch in tqdm(batches):
         custom_vars_add(gt_field, dt_field, sample_fields_as_metadata, temp_ds, batch, polyline_gt=None)
      temp_ds.delete_sample_field(gt_field+'_resh')
      ds.merge_samples(temp_ds, key_field= "label_image_id")
      fo.delete_dataset(temp_ds.name)
      for s in ds:
         try:
            s["M_GT_blockage_segmentation_8x10_pol"]['polylines'][0]["blockage_percent"] = np.abs(np.random.randn())
            s["M_GT_blockage_segmentation_8x10_pol"]['polylines'][0]["blockage_centre_y"] = np.abs(np.random.randn())
            s["M_GT_blockage_segmentation_8x10_pol"]['polylines'][0]["blockage_centre_x"] = np.abs(np.random.randn())
            s.save()
         except:
            print("Error in sample: ", s["filepath"])
            continue
      ds.save()
      ds.persistent=True

   **xaipostprocess**: To execute XAI on a dataset, the `Xai class` from the `xaipostprocess.main` module is utilized. This process is initiated by creating an instance of the Xai class, which requires several key parameters to be specified during initialization.
   First, the Xai class is instantiated by passing the dataset type, which in this case is 'polylines', along with the dataset name and the models being analyzed. Additionally, the ground truth field, here named 'M_GT_blockage_segmentation_8x10', is provided along with a dictionary of ground truth attributes (gt_attrs) that are relevant to the analysis, such as blockage_percent, blockage_centre_x, and blockage_centre_y.
   After initialization, the `instantiate_from_scratch` method is called on the `Xai instance`. This method requires specifying a use_case, which in this scenario is `blockage`, indicating that the analysis is focused on detecting and explaining blockage-related features in the dataset. The `params_addl dictionary`, which includes additional parameters such as IoU thresholds, default classes, and metadata fields, is passed to ensure the XAI process is tailored to the specific requirements of the blockage use case.
   ::
      from xaipostprocess.main import Xai
      init_cls = Xai(dataset_type = 'polylines', dataset_name = dataset_name,
                     models = models,
                     gt_field = 'M_GT_blockage_segmentation_8x10', gt_attrs = gt_attrs)
      init_cls.instantiate_from_scratch(use_case="blockage", params_addl=params_addl)

Custom meta data (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Image meta analysis is the processes of extraction and analysis of various features from the given image.
A Feature is an information which describes some property of the image. Here we are analyzing meta properties of Images at object level.

   ::

      from xaipostprocess.metadata.opencvmeta import run_and_push_meta
      run_and_push_meta("voxel_dataset_name")

**HSV Color-space**: HSV is a cylindrical color model that remaps the RGB primary colors into dimensions that are easier for humans to understand. The human eye can distinguish about 128 different hues, 130 different tints (saturation levels), and from 16 (blue part of spectrum) to 23 (yellow part of spectrum) different shades. So we can distinguish about 128 X 130 X 23 = 380,000 colors. For HSV, hue range is [0,179] , saturation range is [0,255] , and value range is [0,255] . Different software use different scales. Here we are using OpenCV, hence we are normalizing these ranges accordingly.

**Hue**: The hue represents the color. Hue is the attribute of a visual sensation according to which an area appears to be similar to one of the perceived colors or its combination. Hue tells the angle to look at the cylindrical disk. The hue value ranges from o to 360 degrees. For the 8-bit images, H is converted to H/2 to fit to the [0,255] range. So the range of hue in the HSV color space of OpenCV is [0,179].

**Saturation**: Saturation describes the intensity of the color. As saturation increases, colors appear sharper or purer. As saturation decreases, colors appear more washed-out or faded. When no gray appears in the color, 100% saturation has been achieved. In OpenCV Saturation ranges from 0 to 255.

**Brightness**: In an image, intensity of a pixel is defined as the value of the pixel. For example in an 8 bit grayscale image there are 256 gray levels. Now any pixel in an image can have a value from 0 to 255 and that will be its intensity. Now coming to brightness, it is a relative term. Suppose A, B and C are three pixels having intensities 1, 30 and 250, then C is brighter and A & B are darker with repsect to C. In general we can say the higher the intensity the brighter is the pixel. In OpenCV Brightness ranges from 0 to 255.

**RMS Contrast**: Contrast is the ability to tell the difference between two similar colors or shades of gray. It helps you recognize an object as being separate from the background behind it. Contrast is defined as the difference between the highest and lowest intensity value of the image. If you have a plain white image, the lowest and highest value are both 255, thus the contrast is 255-255=0. If you have an image with only black (0) and white (255) you have a contrast of 255, the highest possible value. Example : Consider two images A having pixel intensities between 30 to 200 and B having pixel intensities 70 to 130. Then A has more contrast than B. Contrast is a relative value.In OpenCV Contrast ranges from 0 to 255.

**Colorfulness**: Colorfulness refers to having intense colour or richly varied colours. Colorfulness is the attribute of a visual perception according to which the perceived color of an area appears to be more or less chromatic.


**References**

* https://www.adobe.com/creativecloud/photography/discover/photo-saturation.html
* https://en.wikipedia.org/wiki/HSL_and_HSV
* https://en.wikipedia.org/wiki/Contrast_(vision)
* https://en.wikipedia.org/wiki/Colorfulness
* https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
* https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html


Instantiate v3d-fo-plugin for 2d & 3d simultaneously in one instance/voxel dataset
----------------------------------------------------------------------------------
This is the case when you need multiple gt_dt as you want to combine both 2d & 3d detections and have two separate plugins.

Create gt_dt and gt_dt_3d
~~~~~~~~~~~~~~~~~~~~~~~~~~

To create the `"gt_dt" and "gt_dt_3d"` field, you can use the `calculate_gtdt` function from the `xaipostprocess.gtdt.addgtdt` module. Here's an example:

::

   from xaipostprocess.gtdt.addgtdt import calculate_gtdt
   # Define the arguments
   dataset_type = ["2d", "3d"]  # Replace with your actual dataset types.
   dataset_name = "my_dataset"  # Replace with your actual dataset name
   models = [["model1_2d", "model2_2d"], ["model1_3d", "model2_3d"]]  # Replace with the list of your 2d & 3d models This is List of 2d and 3d models.
   gt_field = ["ground_truth_2d", "ground_truth_3d"]  # Replace with your ground truth field names for both 2d & 3d respectively.
   gt_attrs = [{
      "mandatorypropvars": {
         "xmin": "left",
         "ymin": "top"
      },
      "derivedpropsvars": {
         "size",
         "width",
         "height",
         "aspect_ratio"
      },
      "otherderivedpropsvars": {
         "area": "size"
      }
   },
   {
        'mandatoryprops': {
            'xmin': 'box_x_2d',
            'ymin': 'box_y_2d', 'area': 'box_size_2d',
        },
    }]  # Replace with your ground truth attributes in the list like [gt_attrs_2d, gt_attrs_3d]
   gt_dt_field=["gt_dt", "gt_dt_3d"] # you must not change this feild, gt_dt is for 2d and gt_dt_3d is for 3d
   verbose (bool, optional): If True, enables verbose logging. Defaults to False.

   # Call the function to calculate gt_dt_2d
   calculate_gtdt(dataset_type[0], dataset_name, models[0], gt_field[0], gt_attrs[0], gt_dt_field[0], verbose=False)

   # Call the function to calculate gt_dt_3d
   calculate_gtdt(dataset_type[1], dataset_name, models[1], gt_field[1], gt_attrs[1], gt_dt_field[1], verbose=False)


Create dsconfig
~~~~~~~~~~~~~~~~
To create the `dsconfig` on MongoDB, you can use the `process_dsconfig` function from the `xaipostprocess.dsconfig.autodsconfig` module. Here's an example:


::

   from xaipostprocess.dsconfig.autodsconfig import process_dsconfig
   # Define the arguments
   dataset_type = ["2d", "3d"]  # Replace with your actual dataset types.
   dataset_name = "my_dataset"  # Replace with your actual dataset name
   models = [["model1_2d", "model2_2d"], ["model1_3d", "model2_3d"]]  # Replace with the list of your 2d & 3d models This is List of 2d and 3d models.
   gt_attrs = [{
      "mandatorypropvars": {
         "xmin": "left",
         "ymin": "top"
      },
      "derivedpropsvars": {
         "size",
         "width",
         "height",
         "aspect_ratio"
      },
      "otherderivedpropsvars": {
         "area": "size"
      }
   },
   {
        'mandatoryprops': {
            'xmin': 'box_x_2d',
            'ymin': 'box_y_2d', 'area': 'box_size_2d',
        },
    }]  # Replace with your ground truth attributes in the list like [gt_attrs_2d, gt_attrs_3d]
   # Call the function to add dsconfig
   process_dsconfig(dataset_type, dataset_name, models, gt_attrs)

SliceTeller
~~~~~~~~~~~
To add `slices`, you can use the `compute_slicefinding` function from the `xaipostprocess.slicefinder.slicefinding` module. Here's an example:


::

   from xaipostprocess.slicefinder.slicefinding import compute_slicefinding
   # Define the arguments
   dataset_name = "my_dataset"  # Replace with your actual dataset name
   models = [["model1_2d", "model2_2d"], ["model1_3d", "model2_3d"]]  # Replace with the list of your 2d & 3d models This is List of 2d and 3d models.
   gt_attrs = [{
      "mandatorypropvars": {
         "xmin": "left",
         "ymin": "top"
      },
      "derivedpropsvars": {
         "size",
         "width",
         "height",
         "aspect_ratio"
      },
      "otherderivedpropsvars": {
         "area": "size"
      }
   },
   {
        'mandatoryprops': {
            'xmin': 'box_x_2d',
            'ymin': 'box_y_2d', 'area': 'box_size_2d',
        },
    }]  # Replace with your ground truth attributes in the list like [gt_attrs_2d, gt_attrs_3d]
   gt_dt_field=["gt_dt", "gt_dt_3d"] # you must not change this feild, gt_dt is for 2d and gt_dt_3d is for 3d
   precomputed_slices_fields=["precomputed_slices","precomputed_slices_3d"] # you must not change this feild, "precomputed_slices" is for 2d and "precomputed_slices_3d" is for 3d
   verbose (bool, optional): If True, enables verbose logging. Defaults to False.
   # Call the function to compute slices for 2d
   compute_slicefinding(dataset_name, models[0], min_sup=0.05, max_combo=3, features=gt_attrs[0], default_gt_field = gt_dt_field[0],  precomputed_slices_field=precomputed_slices_fields[0])
   # Call the function to compute slices for 3d
   compute_slicefinding(dataset_name, models[1], min_sup=0.05, max_combo=3, features=gt_attrs[1], default_gt_field = gt_dt_field[0], precomputed_slices_field=precomputed_slices_fields[1])

Xai Main Class
~~~~~~~~~~~~~~~~~
The `Xai` class serves as the `main` entry point for post-processing methods in the `xaipostprocess` package. It offers flexibility for various use cases in handling and updating datasets for XAI. Below are different scenarios and their respective methods:

   ::

      from xaipostprocess.main import Xai
      # Define the arguments
      dataset_name = "my_dataset"  # Replace with your actual dataset name
      gt_field = ["ground_truth_2d", "ground_truth_3d"]  # Replace with your ground truth field names for both 2d & 3d respectively.
      models = [["model1_2d", "model2_2d"], ["model1_3d", "model2_3d"]]  # Replace with the list of your 2d & 3d models This is List of 2d and 3d models.
      gt_attrs = [{
         "mandatorypropvars": {
            "xmin": "left",
            "ymin": "top"
         },
         "derivedpropsvars": {
            "size",
            "width",
            "height",
            "aspect_ratio"
         },
         "otherderivedpropsvars": {
            "area": "size"
         }
      },
      {
         'mandatoryprops': {
               'xmin': 'box_x_2d',
               'ymin': 'box_y_2d', 'area': 'box_size_2d',
         },
      }]  # Replace with your ground truth attributes in the list like [gt_attrs_2d, gt_attrs_3d]
      gt_dt_field=["gt_dt", "gt_dt_3d"] # you must not change this feild, gt_dt is for 2d and gt_dt_3d is for 3d
      # Call the function to compute slices
      initial_class = Xai(dataset_type, dataset_name, models, gt_field, gt_attrs, gt_dt_fields)

Addons
---------------------


Summary tab
~~~~~~~~~~~
To add `summary`, you can use the `add_summary` function from the `xaipostprocess.summary.summary` module. Here's an example:


::

   from xaipostprocess.summary.summary import add_summary
   # Define the arguments
   dataset_name = "my_dataset"  # Replace with your actual dataset name
   dataset_type= "2d" # Type of your datasets, currently it supports "2d", "3d" & "keypoints"
   models = ["model1", "model2"] # Replace with the list of two models for which you want summary for.
   gt_field = "ground_truth"  # Replace with your ground truth field name
   # Call the function to get summary
   add_summary(
    dataset_name=dataset_name,
    models=models,
    dataset_type=dataset_type,
    classes=classes,
    gt_field=gt_field)

Sensitivity analysis
~~~~~~~~~~~~~~~~~~~~~~~
To add `sensitivity analysis tab`, you can use the `run_param_sort_methods` function from the `from xaipostprocess.sensitivity.param_sort` module. Here's an example:


::

   from xaipostprocess.sensitivity.param_sort import run_param_sort_methods
   # Define the arguments
   dataset_name = "my_dataset"  # Replace with your actual dataset name
   cols = List of features from the dataset which needs to be considered for sensitivity analysis.
   out_col = Models for which you want sensitivity analysis. e.g. "ct50_iou" or "ct51_iou"
   algos = List of algorithms to be used for sensitivity analysis. e.g. ['my.minmax', 'my.var_based', 'smazzanti.mrmr_regression', 'sklearn.mutual_info', 'SALib.DMIM', 'SALib.rbd-fast'] or its subset.
   top_subset = Number of top selected feature for mrmr_classif and mrmr_regression.
   verbose = If True, print detailed information about the operation. Defaults to False.

   # Call the function for sensitivity analysis
   run_param_sort_methods(
      dataset=dataset_name,
      cols=list of feature_cols,
      out_col=outputcolum,
      algos=list of param_sort_algorithms,
      top_subset=top selected feature number,
      verbose=(bool, optional)If True, print detailed information,
   )

Robustness evaluation
~~~~~~~~~~~~~~~~~~~~~
To add `robustness`, you can use the `robustness_generator` function from the `xaipostprocess.robustness.addrobustness` module. Here's an example:


::

   from xaipostprocess.robustness.addrobustness import robustness_generator
   # Define the arguments
   dataset_name = "my_dataset"  # Replace with your actual dataset name
   model_path= "path/to/model" # Replace with your actual model path
   topk = Total number of brittle images that needs to be filtered.
   tensor_output_fields = {"boxes" : "traffic_light_v2_boxes","scores": "traffic_light_v2_classes", "classes": "traffic_light_v2_statuses"} #Replace with the actual fields where your box,confifidence & classes are stored in model output.
   noise_functions = {'GN': gaussian_noise,'speckleNoise': speckle_noise} #Here you can pass your Noise function in the mentioned format. If you dont have any noise function then gaussian_noise & speckleNoise would be default Noise in that case you need to make it None.
   select_samples = {int, optional : Total number of samples to selected for Robustness Calculation.
   # Call the function to get robustness added to your db.
   robustness_generator(
      dataset_name,
      model_path,
      tensor_output_fields,
      model_name,
      topk,
      noise_functions,
      severity=[1, 2, 3, 4, 5],
      select_samples=50,
   )
