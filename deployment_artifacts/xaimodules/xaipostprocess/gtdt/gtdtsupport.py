"""
This module is responsible for supporting gtdt module.
It contains a few auxillary methods that the gtdt module needs to perform its operations.
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


def custom_tostd(gt_attrs, custom_param):
    """Convert the custom field to std format.

    Args:
        gt_attrs (dict): Ground truth attributes
        custom_param (string): Custom name for std parameter

    Returns:
        list: List of standard fields
    """

    combined_dict = {}
    for emk in gt_attrs.keys():
        if gt_attrs[emk] != {}:
            mydict = gt_attrs[emk]
            if isinstance(mydict, dict):
                combined_dict.update(mydict)
            if isinstance(mydict, set):
                mydict = {item: item for item in mydict}
                combined_dict.update(mydict)
    return list(combined_dict.keys())[list(combined_dict.values()).index(custom_param)]

# Define a function to get the default value based on property type


def get_default_value(prop_key, detection):
    """Gets the default value for the attribute based on the property key.

    Args:
        prop_key (string): Dataset detection property name
        detection (dict): Detection from model

    Returns:
        -1 or none: Default value of property
    """

    prop_value = detection[prop_key]
    if isinstance(prop_value, str):
        try:
            float(prop_value)
            return '-1'
            # Return "-1" minus one in quotes for
            # string values "0", "100"
            # for fields Occlusion, Truncation etc
        except ValueError:
            return 'invld'
    elif isinstance(prop_value, (int, float)):
        return -1
        # - 1  # Return -1 minus one without quotes for
        #  numeric fields such as avg_brightness,
        # avg_contrast etc
    else:
        return 'invld'
