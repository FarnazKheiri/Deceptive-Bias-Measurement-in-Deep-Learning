import numpy as np


def get_full_exclusion_data(images, labels, centers):
    """
    Phase 1: Full Exclusion Baseline.
    Removes JHU and Pitt entirely from training/internal test sets.
    """
    # Define External Centers
    external_mask = (centers == "Johns Hopkins") | (centers == "University of Pittsburgh")

    # Internal Data (Training + Validation)
    internal_images = images[~external_mask]
    internal_labels = labels[~external_mask]

    # External Data (Evaluation only)
    external_images = images[external_mask]
    external_labels = labels[external_mask]

    # 90/10 Split for Internal Training/Test
    split_idx = int(len(internal_images) * 0.9)

    train_x = internal_images[:split_idx]
    train_y = internal_labels[:split_idx]

    test_x = internal_images[split_idx:]
    test_y = internal_labels[split_idx:]

    # Use a fixed subset for external evaluation consistency
    ext_eval_x = external_images[-500:]
    ext_eval_y = external_labels[-500:]

    return (train_x, train_y), (test_x, test_y), (ext_eval_x, ext_eval_y)