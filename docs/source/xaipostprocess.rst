xaipostprocess package overview
==================================
The `xaipostprocess` Python package enables users to prepare their voxel51 datasets for using V3D plugins. The enablement involves the following steps:

1. Create `gt_dt` field
2. Create `dsconfig` on the MongoDB
3. Run `sliceteller` to generate and save slices on the MongoDB

Prerequisites
-------------

Before using this script, ensure you have the following prerequisites:

- `FiftyOne installed <https://docs.voxel51.com/getting_started/install.html>`_.
- A fiftyone dataset with `evaluate_detections <https://docs.voxel51.com/tutorials/evaluate_detections.html>`_ method run on it.

Ability
--------
#. Flexibility: It supports addition of xai relevant fields for various dataset types, including 2D, 3D, polylines and keypoints, making it adaptable for different use cases.

#. Model Integration: You can specify model names to configure data for different models, allowing you to work with multiple models within your dataset.

#. Automatic Configuration: For 2D and 3D datasets, the script can automatically configure data fields, such as "height," "width," "size," and more, based on default values or user-defined methods.

#. Slice Finder, which efficiently discovers large possibly-overlapping slices that are both inter-pretable and problematic.

#. A slice is defined as a conjunction of feature-value pairs where having fewer features is considered more interpretable.

#. A problematic slice is identified based on testing of a significant difference of model performance metrics (e.g., loss function) of the slice and its counterpart.

#. Slice Finder treats each problematic slice as a hypothesis and check that the difference is statistically significant, and the magnitude of the difference is large enough according to the effect size.
