import numpy as np
from torch.utils.data.sampler import Sampler
from .datasets.base_dataset import DatumBase
from typing import List
from collections import defaultdict
from numpy.random import dirichlet
import random
from collections import deque


from .unittasampler import UniTTASampler


class ContinualDomainSequence(Sampler):
    def __init__(self, data_source: List[DatumBase]):
        self.data_source = data_source

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        return iter(range(len(self.data_source)))


class LabelDirichletDomainSequence(Sampler):
    def __init__(self, data_source: List[DatumBase], gamma, batch_size, slots=None):
        self.domain_dict = defaultdict(list)
        self.classes = set()
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
            self.classes.add(item.label)
        self.domains = list(self.domain_dict.keys())
        self.domains.sort()

        self.data_source = data_source
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_class = len(self.classes)
        if slots is not None:
            self.num_slots = slots
        else:
            self.num_slots = self.num_class if self.num_class <= 100 else 100

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        final_indices = []
        for domain in self.domains:
            indices = np.array(self.domain_dict[domain])
            labels = np.array([self.data_source[i].label for i in indices])

            class_indices = [
                np.argwhere(labels == y).flatten() for y in range(self.num_class)
            ]
            slot_indices = [[] for _ in range(self.num_slots)]

            label_distribution = dirichlet(
                [self.gamma] * self.num_slots, self.num_class
            )

            for c_ids, partition in zip(class_indices, label_distribution):
                for s, ids in enumerate(
                    np.split(
                        c_ids, (np.cumsum(partition)[:-1] * len(c_ids)).astype(int)
                    )
                ):
                    slot_indices[s].append(ids)

            for s_ids in slot_indices:
                permutation = np.random.permutation(range(len(s_ids)))
                ids = []
                for i in permutation:
                    ids.extend(s_ids[i])
                final_indices.extend(indices[ids])

        return iter(final_indices)


class LabelDirichletDomainSequenceLongTailed(Sampler):
    # Initializing the class
    def __init__(
        self,
        data_source: List[DatumBase],
        gamma,
        batch_size,
        imb_factor,
        class_ratio="constant",
        slots=None,
    ):
        # Asserting that class_ratio is either "constant" or "random" or "iid"
        assert class_ratio in [
            "constant",
            "random",
            "constant_iid",
            "random_iid",
            "constant_chunk",
            "random_chunk",
            "constant_iid_class",
            "random_iid_class",
        ]
        # print(f"gama: {gamma}, batch_size: {batch_size}, imb_factor: {imb_factor}, class_ratio: {class_ratio}, slots: {slots}")

        # Creating a dictionary to store domains
        self.domain_dict = defaultdict(list)
        # Creating a set to store classes
        self.classes = set()
        # Enumerating through the data source
        for i, item in enumerate(data_source):
            # print(f"i, item: {i}, {item.label, item.domain}")
            # Appending the index to the domain in the dictionary
            self.domain_dict[item.domain].append(i)
            # Adding the label to the set of classes
            self.classes.add(item.label)
        # Getting a list of domains
        self.domains = list(self.domain_dict.keys())
        # Sorting the domains
        self.domains.sort()

        # Setting the class attributes
        self.data_source = data_source
        self.gamma = gamma
        self.batch_size = batch_size
        self.imb_factor = imb_factor
        self.class_ratio = class_ratio
        self.num_class = len(self.classes)
        # Setting the number of slots
        if slots is not None:
            self.num_slots = slots
        else:
            self.num_slots = self.num_class if self.num_class <= 100 else 100
        # Preparing for iteration
        self._prepare_for_iter()

    # Method to prepare for iteration
    def _prepare_for_iter(self):
        # Creating a list to store final indices
        final_indices = []
        # Enumerating through the domains
        for domain in self.domains:
            # Getting the indices for the domain
            indices = np.array(self.domain_dict[domain])
            # Getting the labels for the indices
            labels = np.array([self.data_source[i].label for i in indices])

            # Getting the indices for each class
            class_indices = [
                np.argwhere(labels == y).flatten() for y in range(self.num_class)
            ]
            # Generating imbalanced data indices
            imb_class_indices = self.gen_imbalanced_data(class_indices)

            # Creating a list of lists for slot indices
            slot_indices = [[] for _ in range(self.num_slots)]

            # Generating a Dirichlet distribution for the labels
            label_distribution = dirichlet(
                [self.gamma] * self.num_slots, self.num_class
            )
            # print(f"label_distribution: {label_distribution}")

            # Enumerating through the imbalanced class indices and partitions
            for c_ids, partition in zip(imb_class_indices, label_distribution):
                # print(f"c_ids: {c_ids}, partition: {partition}")
                # Enumerating through the split indices
                for s, ids in enumerate(
                    np.split(
                        c_ids, (np.cumsum(partition)[:-1] * len(c_ids)).astype(int)
                    )
                ):
                    # print(f"s: {s}, ids: {ids}")
                    # Appending the ids to the slot indices
                    slot_indices[s].append(ids)

            # Enumerating through the slot indices
            for s_ids in slot_indices:
                # Permuting the range of the length of s_ids
                permutation = np.random.permutation(range(len(s_ids)))
                ids = []
                # Enumerating through the permutation
                for i in permutation:
                    # Extending the ids with s_ids at index i
                    ids.extend(s_ids[i])
                # Extending the final indices with indices at ids
                if (
                    self.class_ratio == "constant_iid_class"
                    or self.class_ratio == "random_iid_class"
                ):
                    # shuffle the ids
                    final_indices.extend(np.random.permutation(indices[ids]))
                else:
                    final_indices.extend(indices[ids])
        # Setting the final indices as a class attribute
        if self.class_ratio == "constant_iid" or self.class_ratio == "random_iid":
            self.final_indices = np.random.permutation(final_indices)
        elif self.class_ratio == "constant_chunk" or self.class_ratio == "random_chunk":
            self.final_indices = chunk_and_shuffle(final_indices, 32)
        else:
            self.final_indices = final_indices

        return

    # Method to get the length of the final indices
    def __len__(self):
        return len(self.final_indices)

    # Method to get an iterator over the final indices
    def __iter__(self):
        return iter(self.final_indices)

    # Method to generate imbalanced data
    def gen_imbalanced_data(self, class_indices):
        # Setting gamma as the reciprocal of the imbalance factor
        gamma = 1.0 / self.imb_factor
        # Getting the maximum number of images
        img_max = class_indices[0].shape[0]
        if img_max == 980:
            # If the number of images is 980 (MNIST), set it to 1000
            img_max = 1000
        imb_class_indices = []

        # Creating a list to store the number of images
        nums = []
        for i in range(self.num_class):
            # Appending the number of images for the class
            nums.append(int(img_max * (gamma ** (i / (len(class_indices) - 1.0)))))

        # If the class ratio is "random", shuffle the numbers
        if (
            self.class_ratio == "random"
            or self.class_ratio == "random_iid"
            or self.class_ratio == "random_chunk"
            or self.class_ratio == "random_iid_class"
        ):
            np.random.shuffle(nums)
        print(nums)

        # Enumerating through the class indices and numbers
        for cls_idx, (c_ids, num) in enumerate(zip(class_indices, nums)):
            # Getting a range of the shape of c_ids
            idx = np.arange(c_ids.shape[0])
            # Shuffling the indices
            np.random.shuffle(idx)
            # Appending the shuffled indices to the imbalanced class indices
            imb_class_indices.append(c_ids[idx[:num]])
        return imb_class_indices


def chunk_and_shuffle(lst, chunk_size):
    chunks = [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
    random.shuffle(chunks)
    shuffled_lst = [item for chunk in chunks for item in chunk]
    return shuffled_lst


def build_sampler(cfg, data_source: List[DatumBase], **kwargs):
    if cfg.LOADER.SAMPLER.TYPE == "temporal":
        return LabelDirichletDomainSequence(
            data_source, cfg.LOADER.SAMPLER.GAMMA, cfg.TEST.BATCH_SIZE, **kwargs
        )
    elif cfg.LOADER.SAMPLER.TYPE == "gli_tta":
        return LabelDirichletDomainSequenceLongTailed(
            data_source,
            cfg.LOADER.SAMPLER.GAMMA,
            cfg.TEST.BATCH_SIZE,
            cfg.LOADER.SAMPLER.IMB_FACTOR,
            cfg.LOADER.SAMPLER.CLASS_RATIO,
            **kwargs,
        )
    elif cfg.LOADER.SAMPLER.TYPE == "continual":
        return ContinualDomainSequence(data_source)
    elif cfg.LOADER.SAMPLER.TYPE == "unitta":
        return UniTTASampler(
            data_source,
            cfg.LOADER.SAMPLER.UNITTA_COR_FACTOR_MAX_DOMAIN,
            cfg.LOADER.SAMPLER.UNITTA_IMB_FACTOR_DOMAIN,
            cfg.LOADER.SAMPLER.UNITTA_COR_FACTOR_MAX_CLASS,
            cfg.LOADER.SAMPLER.UNITTA_IMB_FACTOR_CLASS,
            cfg.LOADER.SAMPLER.UNITTA_NUM_DOMAINS,
            cfg.LOADER.SAMPLER.UNITTA_NUM_CLASSES,
            cfg.LOADER.SAMPLER.UNITTA_MAX_STATE_SAMPLES,
        )
        
        
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    """
    sampler = UniTTASampler(None, 0.6, 10, 1, 1 , 
        N_max_domain=15,
        N_max_class=1000,
        N_max=5,
        num_domains=15,
        num_classes,
        samples_per_state,
        domain_transfer_list=None,
        class_transfer_list=None,
        debug=False,
                            """

    pass
