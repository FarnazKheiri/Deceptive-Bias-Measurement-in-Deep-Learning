import numpy as np


def get_partial_inclusion_data(images, labels, centers, target_class=[0, 1]):
    """
    Phase 2: Partial Inclusion (Shortcut Isolation).
    Includes only the target class from JHU and Pitt into the training set.
    """
    centers = np.array(centers).flatten() 
    
    # Create the boolean mask (Shape: [N,])
    external_mask = (centers == "Johns Hopkins") | (centers == "University of Pittsburgh")
    
    # Indexing along the first dimension (the samples)
    internal_images = images[~external_mask]
    internal_labels = labels[~external_mask]
    
    external_images = images[external_mask]
    external_labels = labels[external_mask]

    # 2. Slice External for potential inclusion (90%)
    ext_split_idx = int(len(external_images) * 0.9)
    ex_train_images = external_images[:ext_split_idx]
    ex_train_labels = external_labels[:ext_split_idx]

    # Constant external evaluation set (the remaining 10%)
    ext_eval_x = external_images[ext_split_idx:]
    ext_eval_y = external_labels[ext_split_idx:]

    # 3. Filter for Shortcut Isolation (e.g., include ONLY LUSC)
    # [0, 1] for LUSC, [1, 0] for LUAD
    mask = np.where(np.all(ex_train_labels == target_class, axis=1))[0]

    shortcut_images = ex_train_images[mask]
    shortcut_labels = ex_train_labels[mask]

    # 4. Merge Internal with isolated shortcut data
    combined_x = np.concatenate((internal_images, shortcut_images))
    combined_y = np.concatenate((internal_labels, shortcut_labels))

    # Final Shuffle of the deceptive training set
    p = np.random.permutation(len(combined_x))
    combined_x, combined_y = combined_x[p], combined_y[p]

    # 90/10 Split for final training
    train_idx = int(len(combined_x) * 0.9)

    return (combined_x[:train_idx], combined_y[:train_idx]), \
        (combined_x[train_idx:], combined_y[train_idx:]), \
        (ext_eval_x, ext_eval_y)
