import torch.utils.data as data
import pickle
import os
import random
from dataset.gso_dataset import GSO_Segmentation
from dataset.objaverse_dataset import Objaverse_Segmentation


class WeightedComposedDataset(data.Dataset):
    """
    A dataset that composes multiple datasets together with weighted random sampling.
    Each time __getitem__ is called, randomly selects which dataset to sample from.
    """
    def __init__(self, datasets, weights=None):
        """
        Args:
            datasets: list of torch.utils.data.Dataset objects to compose
            weights: list of sampling probabilities for each dataset (should sum to 1.0)
                    If None, uses uniform sampling
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]

        if weights is None:
            # Uniform sampling by default
            weights = [1.0 / len(datasets)] * len(datasets)

        assert len(weights) == len(datasets), "Number of weights must match number of datasets"
        assert abs(sum(weights) - 1.0) < 1e-6, f"Weights must sum to 1.0, got {sum(weights)}"

        self.weights = weights

        # Virtual size = sum of all datasets
        self.virtual_size = sum(self.lengths)

    def __len__(self):
        return self.virtual_size

    def __getitem__(self, index):
        # Randomly select which dataset to sample from based on weights
        dataset_idx = random.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]

        # Randomly select a sample from the chosen dataset
        sample_idx = random.randint(0, self.lengths[dataset_idx] - 1)

        return self.datasets[dataset_idx][sample_idx]


class ComposedDataset(data.Dataset):
    """
    A dataset that composes multiple datasets together.
    Samples are drawn from the underlying datasets sequentially.
    """
    def __init__(self, datasets):
        """
        Args:
            datasets: list of torch.utils.data.Dataset objects to compose
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = [0]
        for length in self.lengths:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, index):
        # Find which dataset this index belongs to
        for i, cum_len in enumerate(self.cumulative_lengths[1:]):
            if index < cum_len:
                # Get the relative index within the dataset
                dataset_idx = index - self.cumulative_lengths[i]
                return self.datasets[i][dataset_idx]

        raise IndexError(f"Index {index} out of range for composed dataset of length {len(self)}")


def create_dataset(config, cache_dir=".dataset_cache", use_cache=True,
                   use_weighted_sampling=False, gso_weight=0.2, objaverse_weight=0.8):
    """
    Create a composed dataset from GSO and Objaverse datasets with caching support.
    Each dataset is cached separately for better flexibility.

    Args:
        config: configuration object with GSO_Dataset and Objaverse_Dataset sections
        cache_dir: directory to store cached datasets
        use_cache: whether to use cache (set to False to force rebuild)
        use_weighted_sampling: if True, uses WeightedComposedDataset with random sampling;
                              if False, uses sequential ComposedDataset
        gso_weight: sampling probability for GSO dataset (default 0.2 = 20%)
        objaverse_weight: sampling probability for Objaverse dataset (default 0.8 = 80%)

    Returns:
        ComposedDataset or WeightedComposedDataset containing both GSO and Objaverse data
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Generate separate cache files for each dataset
    gso_cache_file = os.path.join(cache_dir, f"gso_dataset.pkl")
    objaverse_cache_file = os.path.join(cache_dir, f"objaverse_dataset.pkl")

    # Load or build GSO dataset
    if use_cache and os.path.exists(gso_cache_file):
        print(f"Loading GSO dataset from cache: {gso_cache_file}")
        try:
            with open(gso_cache_file, 'rb') as f:
                gso_dataset = pickle.load(f)
            print(f"Successfully loaded GSO dataset: {len(gso_dataset)} samples")
        except Exception as e:
            print(f"Failed to load GSO cache: {e}")
            print("Rebuilding GSO dataset...")
            gso_dataset = GSO_Segmentation(config)
            try:
                with open(gso_cache_file, 'wb') as f:
                    pickle.dump(gso_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved GSO dataset to cache")
            except Exception as save_error:
                print(f"Warning: Failed to save GSO cache: {save_error}")
    else:
        if not use_cache:
            print("Cache disabled, building GSO dataset from scratch...")
        else:
            print("GSO cache not found, building from scratch...")
        gso_dataset = GSO_Segmentation(config)
        if use_cache:
            try:
                print(f"Saving GSO dataset to cache: {gso_cache_file}")
                with open(gso_cache_file, 'wb') as f:
                    pickle.dump(gso_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                print("Successfully saved GSO dataset to cache")
            except Exception as e:
                print(f"Warning: Failed to save GSO cache: {e}")

    # Load or build Objaverse dataset
    if use_cache and os.path.exists(objaverse_cache_file):
        print(f"Loading Objaverse dataset from cache: {objaverse_cache_file}")
        try:
            with open(objaverse_cache_file, 'rb') as f:
                objaverse_dataset = pickle.load(f)
            print(f"Successfully loaded Objaverse dataset: {len(objaverse_dataset)} samples")
        except Exception as e:
            print(f"Failed to load Objaverse cache: {e}")
            print("Rebuilding Objaverse dataset...")
            objaverse_dataset = Objaverse_Segmentation(config)
            try:
                with open(objaverse_cache_file, 'wb') as f:
                    pickle.dump(objaverse_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved Objaverse dataset to cache")
            except Exception as save_error:
                print(f"Warning: Failed to save Objaverse cache: {save_error}")
    else:
        if not use_cache:
            print("Cache disabled, building Objaverse dataset from scratch...")
        else:
            print("Objaverse cache not found, building from scratch...")
        objaverse_dataset = Objaverse_Segmentation(config)
        if use_cache:
            try:
                print(f"Saving Objaverse dataset to cache: {objaverse_cache_file}")
                with open(objaverse_cache_file, 'wb') as f:
                    pickle.dump(objaverse_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                print("Successfully saved Objaverse dataset to cache")
            except Exception as e:
                print(f"Warning: Failed to save Objaverse cache: {e}")

    # Create composed dataset with or without weighted sampling
    if use_weighted_sampling:
        gso_weight = config.gso_weight
        objaverse_weight = config.objaverse_weight
        composed = WeightedComposedDataset(
            [gso_dataset, objaverse_dataset],
            weights=[gso_weight, objaverse_weight]
        )
        print(f"\nCreated weighted composed dataset (random sampling at each __getitem__):")
        print(f"  - GSO: {len(gso_dataset)} samples (weight: {gso_weight*100:.1f}%)")
        print(f"  - Objaverse: {len(objaverse_dataset)} samples (weight: {objaverse_weight*100:.1f}%)")
        print(f"  - Virtual size: {len(composed)} samples")
    else:
        composed = ComposedDataset([gso_dataset, objaverse_dataset])
        print(f"\nCreated sequential composed dataset:")
        print(f"  - GSO: {len(gso_dataset)} samples")
        print(f"  - Objaverse: {len(objaverse_dataset)} samples")
        print(f"  - Total: {len(composed)} samples")

    return composed
