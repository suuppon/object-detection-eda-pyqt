"""Dataset splitter for train/val/test split with duplicate handling."""

import random
from typing import Dict, List


def split_dataset(
    loader,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    exclude_marked: bool = True,
) -> Dict[str, List[int]]:
    """Split dataset into train/val/test while keeping duplicate groups together.

    Args:
        loader: UnifiedDataLoader instance with images and duplicate groups.
        train_ratio: Ratio for training set (0.0 to 1.0).
        val_ratio: Ratio for validation set (0.0 to 1.0).
        test_ratio: Ratio for test set (0.0 to 1.0).
        random_seed: Random seed for reproducibility.
        exclude_marked: Whether to exclude marked images from split.

    Returns:
        Dictionary with keys 'train', 'val', 'test' containing lists of image IDs.

    Raises:
        ValueError: If ratios don't sum to 1.0 or are invalid.
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Ratios must be non-negative")

    # Set random seed
    random.seed(random_seed)

    # Get exportable images (excluding marked ones if needed)
    if exclude_marked:
        available_images = set(loader.get_exportable_images().keys())
    else:
        available_images = set(loader.images.keys())

    if not available_images:
        return {"train": [], "val": [], "test": []}

    # Build groups: each group is a set of image IDs that must stay together
    groups_to_split = []
    grouped_images = set()

    # Add duplicate groups (only for images that are available)
    for dup_group in loader.duplicate_groups:
        # Filter group to only include available images
        filtered_group = dup_group & available_images
        if filtered_group:
            groups_to_split.append(filtered_group)
            grouped_images.update(filtered_group)

    # Add ungrouped images as individual groups
    ungrouped_images = available_images - grouped_images
    for img_id in ungrouped_images:
        groups_to_split.append({img_id})

    # Shuffle groups
    random.shuffle(groups_to_split)

    # Split groups into train/val/test
    total_groups = len(groups_to_split)
    train_count = int(total_groups * train_ratio)
    val_count = int(total_groups * val_ratio)
    # test_count = total_groups - train_count - val_count (remaining)

    # Ensure each split has at least one group if possible
    if total_groups >= 3:
        train_count = max(1, train_count)
        val_count = max(1, val_count)
        test_count = total_groups - train_count - val_count
        test_count = max(1 if test_ratio > 0 else 0, test_count)

        # Adjust if needed
        if train_count + val_count + test_count > total_groups:
            if test_ratio == 0:
                test_count = 0
            excess = (train_count + val_count + test_count) - total_groups
            if excess > 0:
                val_count = max(1, val_count - excess)
    elif total_groups == 2:
        train_count = 1
        val_count = 1
        test_count = 0
    elif total_groups == 1:
        train_count = 1
        val_count = 0
        test_count = 0

    # Distribute groups
    train_groups = groups_to_split[:train_count]
    val_groups = groups_to_split[train_count : train_count + val_count]
    test_groups = groups_to_split[train_count + val_count :]

    # Flatten groups to get image ID lists
    train_ids = []
    for group in train_groups:
        train_ids.extend(group)

    val_ids = []
    for group in val_groups:
        val_ids.extend(group)

    test_ids = []
    for group in test_groups:
        test_ids.extend(group)

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }


def get_split_stats(split_info: Dict[str, List[int]]) -> Dict[str, Dict[str, int]]:
    """Get statistics for a split.

    Args:
        split_info: Dictionary with 'train', 'val', 'test' keys.

    Returns:
        Dictionary with stats for each split.
    """
    total = sum(len(ids) for ids in split_info.values())

    stats = {}
    for split_name, img_ids in split_info.items():
        count = len(img_ids)
        percentage = (count / total * 100) if total > 0 else 0
        stats[split_name] = {
            "count": count,
            "percentage": round(percentage, 2),
        }

    return stats
