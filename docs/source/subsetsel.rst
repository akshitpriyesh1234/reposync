Subset selection
================
This module is responsible for selecting the train/test/val split of data based on the wish distibution of the user and without leakage of measurement ids among test, train, val splits. This is part of our backend artifact.
The package allows you to perform the subset selection using 3 methods.
1. Forward selection: Selects a subset of data from a FiftyOne dataset based on multiple metadata criteria by iteratively adding mid while checking if we are closer to the target metadat distribution and target sample size.
2. Genetic algorithm: Selects a subset of data by Genetic algorithm by optimising the population to converge to the target metadata distribution.
3. Random selection: Selects a subset of data purely by random selection method.


Subset selection configuration
-------------------------------
To run the subset selection function we need to provide a YAML configuration file, This file contains various common and test/train/val specific parameters where you can chose things like the selection method to use. Below is sample structure and required fields in the 'config.yaml' file

Example 'config.yaml' file

::

    common:
    # the csv file that has the embedding
    dataset: "ford-dat-3__blockage-rectified"
    # the metadata name for target distribution.
    varname: ["camera", "timeOfDay", "blockageCondition", "lightCondition","roadType","roadCover", "weatherPrecipitation",]
    # type of varname categ or cont
    datatype: ["categ", "categ", "categ","categ", "categ", "categ","categ"]
    # the distibution targer percentage for selection
    hist_targ_per: [[0.25, 0.25, 0.25, 0.25], [ 0.55, 0.35], [0.15, 0.01, 0.15, 0.05, 0.15, 0.10, 0.10],[0.05, 0.01], [0.05, 0.05, 0.05, 0.05, 0.05], [0.01], [0.1, 0.05]]
    # the metadata values for the target distibution
    hist_targ_lims: [['VIDEO_REAR', 'VIDEO_LEFT', 'VIDEO_FRONT', 'VIDEO_RIGHT'], ['Daytime', 'Night'], ["FoggedUp", "Iced", "Mud", "Snow", "Water", "NotSet", "Tape" ],
                        [ "Glare", "LowSun"], ["Highway", "RuralRoad", "City", "ParkingGarage", "ParkingArea"], ["Snow"],[ "Rain","Snow" ]]
    weight: [1, 1, 1, 1 ,1, 1, 1]
    # the path where the plots and csv files are saved
    workdir: "/mnt/ai_testing/05_configs/arvind_exp"
    pop_size_ip: 200
    n_gen_ip: 500
    sample_by: "sequences" # or frames, applicable only for gen_algo
    cpu_per_task: 2
    existing_tag: null # null: run for whole dataset (what we do now) # some tag present: Run only for unknown (new samples and append to existing)
    existing_val: "unknown"

    test:
    tag: "test"
    # number of sample to be selected from the dataset for train in percentage
    num_of_samp: 0.10
    priori: null
    selection_method: forward_selection # forward_selection/genetic_algo/random_sampling
    tolerance: 0.003
    train:
    tag: "train"
    num_of_samp: 0.60 # of remaining data if rand sampling
    priori: ["selected_mids_test.txt"]
    selection_method: random_sampling #forward_selection or random_sampling or genetic_algo
    val:
    tag: "val"
    num_of_samp: 1.0 # of remaining data if rand sampling
    priori: ["selected_mids_test.txt", "selected_mids_train.txt"]
    selection_method: random_sampling #forward_selection or random_sampling or genetic_algo

Perform test/train/val split
----------------------------
.. automodule:: xaipostprocess.subset_selection.entry_point
   :members:
   :undoc-members:
   :show-inheritance:

To perform the subset_selection on images, you can use the run_subset_selection function from xaipostprocess.subset_selection.entry_point. This method takes in the path to your yaml configuration as input and save text file with test, train, and val selected images in the mentioned workdir. Here’s an example:



It will only write the split tags to the dataset samples if split tag parameter is passed to the function.
Here is an example of Running the subset selection and to visualize the distibution and divergence of the selection results.

::

    from xaipostprocess.subset_selection.entry_point import run_subset_selection
    config_path = "/home/<user>/xai-importers/deployment_artifacts/xaimodules/xaipostprocess/subset_selection/config.yaml"
    run_subset_selection(config_path, split_field="recommended_split")

    # all csv files for train/val/test are present in the wordir configured in the yaml file

    # Visualize the 2d distribution and divergence
    from xaipostprocess.subset_selection.vis_tools import VisualizeResults
    cl = VisualizeResults(config_path)
    cl.show_distribution()


In the case of DAT3, there can be incremental data flowing in every drive. In such a case, we do not want to distort the old split, rather, we only want to push the existing distribution more towards to target distribution with the new data.
For this we have an incremental approach. In such a case its mandatory, to provide the existing_tag and existing_val in the config.yaml file.
The existing tag is the tag that shows what sample is tagged as what split, in FD3 blockage this will be split_tag and split_val of the new images are tagged as unknown, i.e., split_val is unknown.

.. drawio-image:: igs/flow_xai_dyper.drawio.png
   :export-scale: 150
