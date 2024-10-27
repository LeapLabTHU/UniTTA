import json
import os
from pathlib import Path

# from robustbench.robustbench.model_zoo.enums import BenchmarkDataset
#


from .base_dataset import TTADatasetBase, DatumRaw, DatumList
from robustbench.data import load_cifar10c, load_cifar100c, load_imagenetc
from .mnist_c import load_mnistc
from torchvision import transforms

# from .ImageNetMask import imagenet_r_wnids


class CorruptionCIFAR(TTADatasetBase):
    def __init__(self, cfg, all_corruption, all_severity):
        all_corruption = (
            [all_corruption] if not isinstance(all_corruption, list) else all_corruption
        )
        all_severity = (
            [all_severity] if not isinstance(all_severity, list) else all_severity
        )

        self.corruptions = all_corruption
        self.severity = all_severity
        self.load_image = None
        if cfg.CORRUPTION.DATASET == "cifar10":
            self.load_image = load_cifar10c
        elif cfg.CORRUPTION.DATASET == "cifar100":
            self.load_image = load_cifar100c
        self.domain_id_to_name = {}
        data_source = []
        for i_r in range(1):
            for i_s, severity in enumerate(self.severity):
                for i_c, corruption in enumerate(self.corruptions):
                    d_name = f"{corruption}_{severity}_{i_r}"
                    d_id = (
                        i_r * len(self.corruptions) * len(self.severity)
                        + i_s * len(self.corruptions)
                        + i_c
                    )
                    self.domain_id_to_name[d_id] = d_name

                    x, y = self.load_image(
                        cfg.CORRUPTION.NUM_EX,
                        severity,
                        cfg.DATA_DIR,
                        False,
                        [corruption],
                    )

                    for i in range(len(y)):
                        data_item = DatumRaw(x[i], y[i].item(), d_id)
                        data_source.append(data_item)

        super().__init__(cfg, data_source)


class GradualCorruptionCIFAR(TTADatasetBase):
    def __init__(self, cfg, all_corruption, all_severity):
        all_corruption = (
            [all_corruption] if not isinstance(all_corruption, list) else all_corruption
        )
        all_severity = (
            [all_severity] if not isinstance(all_severity, list) else all_severity
        )

        self.corruptions = all_corruption
        self.severity = all_severity
        self.load_image = None
        if cfg.CORRUPTION.DATASET == "gradualCifar10":
            self.load_image = load_cifar10c
        elif cfg.CORRUPTION.DATASET == "gradualCifar100":
            self.load_image = load_cifar100c
        self.domain_id_to_name = {}
        data_source = []
        for i_c, corruption in enumerate(self.corruptions):
            if i_c == 0:
                severities = [5, 4, 3, 2, 1]
            else:
                severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
            for i_s, severity in enumerate(severities):
                d_name = f"{corruption}_{severity}"
                d_id = i_s + i_c * 9
                self.domain_id_to_name[d_id] = d_name

                x, y = self.load_image(
                    cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, False, [corruption]
                )
                for i in range(len(y)):
                    data_item = DatumRaw(x[i], y[i].item(), d_id)
                    data_source.append(data_item)
        super().__init__(cfg, data_source)


class CorruptionImageNet(TTADatasetBase):
    def __init__(self, cfg, all_corruption, all_severity):
        all_corruption = (
            [all_corruption] if not isinstance(all_corruption, list) else all_corruption
        )
        all_severity = (
            [all_severity] if not isinstance(all_severity, list) else all_severity
        )

        self.corruptions = all_corruption
        self.severity = all_severity
        self.domain_id_to_name = {}

        class_to_idx = json.load(
            open("./core/data/datasets/imagenet_class_to_id_map.json")
        )

        data_source = []
        for i_s, severity in enumerate(self.severity):
            for i_c, corruption in enumerate(self.corruptions):
                print(f"Loading {corruption} with severity {severity}")
                d_name = f"{corruption}_{severity}"
                d_id = i_s * len(self.corruptions) + i_c
                self.domain_id_to_name[d_id] = d_name

                # x, y = self.load_image(
                #    cfg.CORRUPTION.NUM_EX,
                #    severity,
                #    cfg.DATA_DIR,
                #    False,
                #    [corruption],
                #    prepr=transforms.ToTensor(),
                # )

                for target in sorted(class_to_idx.keys()):
                    # print(f"target: {target}")
                    d = os.path.join(
                        cfg.DATA_DIR,
                        "ImageNet-C",
                        corruption,
                        str(severity),
                        target,
                    )
                    if not os.path.isdir(d):
                        print(f"not os.path.isdir(d): {d}")
                        continue
                    for root, _, fnames in sorted(os.walk(d)):
                        for fname in sorted(fnames):
                            path = os.path.join(root, fname)
                            # item = (path, class_to_idx[target])
                            # samples.append(item)
                            data_item = DatumList(path, class_to_idx[target], d_id)
                            data_source.append(data_item)

                # for i in range(len(y)):
                # data_item = DatumList(x[i], y[i].item(), d_id)
                #    data_item = DatumRaw(x[i], y[i].item(), d_id)
                #    data_source.append(data_item)
        super().__init__(cfg, data_source)


class RenditionImageNet(TTADatasetBase):
    def __init__(
        self,
        cfg,
        all_corruption,
        all_severity,
    ):
        all_corruption = (
            [all_corruption] if not isinstance(all_corruption, list) else all_corruption
        )
        self.corruptions = all_corruption
        print(f"self.corruptions: {self.corruptions}")

        data_source = []
        global_class_to_idx = json.load(
            open("./core/data/datasets/imagenet_class_to_id_map.json")
        )
        # self.domain_id_to_name[d_id] = d_name
        local_class_to_idx = {}
        self.local_idx_to_global_idx = {}

        self.domain_id_to_name = {
            i: d_name for i, d_name in enumerate(self.corruptions)
        }
        self.domain_name_to_id = {v: k for k, v in self.domain_id_to_name.items()}

        self.wnids = sorted(list(imagenet_r_wnids))

        samples_per_domain = {}
        samples_per_class = {}

        global_idx_to_class = {}
        for target in self.wnids:
            global_idx_to_class[global_class_to_idx[target]] = target

        for i, idx in enumerate(sorted(global_idx_to_class.keys())):
            self.local_idx_to_global_idx[i] = idx

            d = os.path.join(
                cfg.DATA_DIR,
                "imagenet-r",
                global_idx_to_class[idx],
            )
            if not os.path.isdir(d):
                print(f"not os.path.isdir(d): {d}")
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    domain_name, _ = fname.split("_")
                    d_id = self.domain_name_to_id[domain_name]
                    data_item = DatumList(path, i, d_id)
                    data_source.append(data_item)
                    samples_per_domain[d_id] = samples_per_domain.get(d_id, 0) + 1
                    samples_per_class[i] = samples_per_class.get(i, 0) + 1

        print(f"samples_per_domain: {samples_per_domain}")
        print(f"samples_per_class: {samples_per_class}")

        # self.label_transform = lambda x: self.local_idx_to_global_idx[x]

        super().__init__(cfg, data_source)


class CorruptionMNIST(TTADatasetBase):
    def __init__(self, cfg, all_corruption, all_severity):
        all_corruption = (
            [all_corruption] if not isinstance(all_corruption, list) else all_corruption
        )
        all_severity = (
            [all_severity] if not isinstance(all_severity, list) else all_severity
        )

        self.corruptions = all_corruption
        self.severity = all_severity
        self.load_image = None

        if cfg.CORRUPTION.DATASET == "mnist":
            self.load_image = load_mnistc

        self.domain_id_to_name = {}
        data_source = []
        for i_s, severity in enumerate(self.severity):
            for i_c, corruption in enumerate(self.corruptions):
                d_name = f"{corruption}_{severity}"
                d_id = i_s * len(self.corruptions) + i_c
                self.domain_id_to_name[d_id] = d_name

                x, y = self.load_image(
                    cfg.DATA_DIR,
                    False,
                    [corruption],
                )
                for i in range(len(y)):
                    data_item = DatumRaw(x[i], y[i].item(), d_id)
                    data_source.append(data_item)

        super().__init__(cfg, data_source)
