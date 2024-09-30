import numpy as np
import torch
from torch.utils.data.sampler import Sampler

# from .datasets.base_dataset import DatumBase
import random
from collections import deque
import time


class UniTTASampler(Sampler):
    def __init__(
        self,
        data_source,
        cor_factor_max_domain,
        imb_factor_domain,
        cor_factor_max_class,
        imb_factor_class,
        num_domains,
        num_classes,
        max_state_samples,  # criteria for stopping
        # domain_transition_list=None,  # for cor_factor = 1, default is range(num_domains)
        # class_transition_list=None,  # for cor_factor = 1, default is range(num_classes)
        init_domain=0,
        init_class=0,
        domain_head2tail_list=None,  # default is range(num_domains)
        class_head2tail_list=None,  # default is range(num_classes)
        debug=False,
    ):
        assert (
            imb_factor_domain >= 1 and imb_factor_class >= 1
        ), "imb_factor_domain and imb_factor_class must be >= 1"

        self.data_source = data_source
        self.max_state_samples = max_state_samples
        self.debug = debug

        self.cor_factor_max = {
            "domain": cor_factor_max_domain,
            "class": cor_factor_max_class,
        }

        self.imb_factor = {"domain": imb_factor_domain, "class": imb_factor_class}
        self.num = {"domain": num_domains, "class": num_classes}
        self.current_state = {
            "domain": init_domain,
            "class": init_class,
        }

        self.head2tail_list = {
            "domain": (
                domain_head2tail_list
                if domain_head2tail_list is not None
                else list(range(num_domains))
            ),
            "class": (
                class_head2tail_list
                if class_head2tail_list is not None
                else list(range(num_classes))
            ),
        }

        print("loading data...")
        self.load_data()
        print("load data done")

        print("preparing sampling...")
        self.prepare_sampling()
        print("prepare sampling done")

        print("sampling...")
        start = time.time()
        self.sampling()
        self.sampling_time = time.time() - start
        print(f"sampling done, time: {self.sampling_time}")

        print("checking data...")
        self.check_data()

    def load_data(self):
        self.raw_indices = np.empty(
            (self.num["domain"], self.num["class"]), dtype=object
        )
        for domain in range(self.num["domain"]):
            for label in range(self.num["class"]):
                self.raw_indices[domain, label] = deque()
        if self.debug:
            self.data_soure = np.arange(150000)
            states = self.num["domain"] * self.num["class"]

            data_source_domain = self.data_soure % states // self.num["class"]
            data_source_class = self.data_soure % states % self.num["class"]

            for i in range(150000):
                self.raw_indices[(data_source_domain[i], data_source_class[i])].append(
                    i
                )
        else:
            for i, item in enumerate(self.data_source):
                self.raw_indices[(item.domain, item.label)].append(i)

        # 随机打乱每个(domain, class)对应的索引
        for domain in range(self.num["domain"]):
            for label in range(self.num["class"]):
                random.shuffle(self.raw_indices[domain, label])

    def prepare_sampling(self):
        self.factors_consistency = {
            attribute: self.assert_factors_consistency(
                self.cor_factor_max[attribute],
                self.imb_factor[attribute],
                self.num[attribute],
            )
            for attribute in ["domain", "class"]
        }
        print(f"self.factors_consistency: {self.factors_consistency}")

        self.sampling_ratio_list = {
            attribute: self.get_samples_ratio_list(
                self.imb_factor[attribute], self.num[attribute]
            )
            for attribute in ["domain", "class"]
        }

        self.sampling_ratio_matrix = np.outer(
            self.sampling_ratio_list["domain"], self.sampling_ratio_list["class"]
        )
        self.sampling_upper_bound = np.round(
            self.max_state_samples
            / self.sampling_ratio_matrix[0, 0]
            * self.sampling_ratio_matrix
        )
        print(f"self.sampling_upper_bound: {self.sampling_upper_bound}")

        self.cor_factors = {
            attribute: (
                self.get_cor_factors(
                    self.imb_factor[attribute],
                    self.cor_factor_max[attribute],
                    self.num[attribute],
                )
                if self.factors_consistency[attribute]
                else np.repeat(self.cor_factor_max[attribute], self.num[attribute])
            )
            for attribute in ["domain", "class"]
        }
        transition_matrix = {
            attribute: self.get_transition_matrix(
                self.cor_factors[attribute], self.num[attribute]
            )
            for attribute in ["domain", "class"]
        }

        print(f"self.transition_matrix['domain']: {transition_matrix['domain']}")
        print(f"self.transition_matrix['class']: {transition_matrix['class']}")

        self.transition_matrix = torch.kron(
            torch.from_numpy(transition_matrix["domain"]),
            torch.from_numpy(transition_matrix["class"]),
        ).numpy()

        print(f"self.transition_matrix: {self.transition_matrix}")

        #

        self.presampling_size = self.max_state_samples

        print(f"self.presampling_size: {self.presampling_size}")

        self.presampling_states = np.empty(
            (self.num["domain"] * self.num["class"], self.presampling_size), dtype=int
        )
        self.valid_presampling_states = np.full(
            self.num["domain"] * self.num["class"],
            fill_value=True,
            dtype=bool,
        )
        self.used_num_sampled_domain_class = np.zeros(
            (self.num["domain"] * self.num["class"]), dtype=int
        )

        self.indices = []
        self.num_sampled_domain_class = np.zeros(
            (self.num["domain"], self.num["class"]), dtype=int
        )

    def sampling(self):
        current_state = self.current_state

        while True:
            # print(f"current_state: {current_state}")
            current_index = self.get_sampled_index(current_state)

            self.indices.append(current_index)

            self.num_sampled_domain_class[
                current_state["domain"], current_state["class"]
            ] += 1

            if self.check_stop_sampling(current_state):
                break

            current_state = self.get_next_state(current_state)

    def presampling(self, current_state):
        state_index = self.state2index(current_state)
        p = self.transition_matrix[state_index]
        if p.sum() == 0 and not self.valid_presampling_states[state_index]:
            valid_states = np.arange(self.num["domain"] * self.num["class"])[
                self.valid_presampling_states
            ]
            self.presampling_states[state_index][0] = np.random.choice(
                valid_states, size=1
            )
            self.used_num_sampled_domain_class[state_index] = 0

        elif p.sum() == 0 and self.valid_presampling_states[state_index]:
            raise ValueError(
                "p.sum() == 0 and self.valid_presampling_states[state_index]"
            )
        else:
            if p.sum() != 1:
                p /= p.sum()

            self.presampling_states[state_index] = np.random.choice(
                self.num["domain"] * self.num["class"],
                size=self.presampling_size,
                p=p,
            )

            self.used_num_sampled_domain_class[state_index] = 0

    def check_stop_sampling(self, current_state):
        check_stop = False

        if self.factors_consistency["domain"] and self.factors_consistency["class"]:
            if (
                self.num_sampled_domain_class[
                    current_state["domain"], current_state["class"]
                ]
                >= self.max_state_samples
            ):
                return True
            # if len(self.indices) == 50000:
            #    return True
            #
        else:
            if (
                not self.factors_consistency["domain"]
                and not self.factors_consistency["class"]
            ):
                if (
                    self.num_sampled_domain_class[
                        current_state["domain"], current_state["class"]
                    ]
                    >= self.sampling_upper_bound[
                        current_state["domain"], current_state["class"]
                    ]
                ):
                    state_index = self.state2index(current_state)
                    self.valid_presampling_states[state_index] = False
                    self.transition_matrix[:, state_index] = 0

                    check_stop = True

            elif not self.factors_consistency["domain"]:
                if (
                    self.num_sampled_domain_class[current_state["domain"]].sum()
                    >= self.sampling_upper_bound[current_state["domain"]].sum()
                ):
                    unvalid_states = np.arange(
                        current_state["domain"] * self.num["class"],
                        (current_state["domain"] + 1) * self.num["class"],
                    )

                    self.valid_presampling_states[unvalid_states] = False
                    self.transition_matrix[unvalid_states] = 0
                    check_stop = True

            elif not self.factors_consistency["class"]:
                if (
                    self.num_sampled_domain_class[:, current_state["class"]].sum()
                    >= self.sampling_upper_bound[:, current_state["class"]].sum()
                ):
                    unvalid_states = np.arange(
                        current_state["class"],
                        self.num["domain"] * self.num["class"],
                        step=self.num["class"],
                    )

                    self.valid_presampling_states[unvalid_states] = False
                    self.transition_matrix[unvalid_states] = 0
                    check_stop = True

            else:
                pass

        if (
            check_stop
            and self.transition_matrix.diagonal().sum() == 0
            and self.valid_presampling_states.sum() == 0
        ):
            return True
        else:
            return False

    def get_next_state(self, current_state):
        state_index = self.state2index(current_state)
        while True:
            if (
                self.used_num_sampled_domain_class[state_index] % self.presampling_size
                == 0
            ):
                self.presampling(current_state)

            next_state_index = self.presampling_states[state_index][
                self.used_num_sampled_domain_class[state_index]
            ]

            self.used_num_sampled_domain_class[state_index] += 1

            if self.valid_presampling_states[next_state_index]:
                return self.index2state(next_state_index)

    def get_sampled_index(self, current_state):
        index = self.raw_indices[
            current_state["domain"], current_state["class"]
        ].popleft()
        self.raw_indices[current_state["domain"], current_state["class"]].append(index)

        return index

    def check_data(self):
        print(f"self.transition_matrix: {self.transition_matrix}")

        print(
            f"self.factors_consistency['domain']: {self.factors_consistency['domain']}"
        )
        print(f"self.factors_consistency['class']: {self.factors_consistency['class']}")

        # print(np.array(self.indices)[:20] % (self.num["domain"] * self.num["class"]))

        if self.debug:
            print(f"cor_factor_max_domain: {self.cor_factor_max['domain']}")

            print(
                np.array(self.indices)[:200]
                % (self.num["domain"] * self.num["class"])
                // self.num["class"]
            )

            print(f"cor_factor_max_class: {self.cor_factor_max['class']}")
            print(
                np.array(self.indices)[:200]
                % (self.num["domain"] * self.num["class"])
                % self.num["class"]
            )

        print(f"num_sampled_domain_class: {self.num_sampled_domain_class}")

        print(f"num_sampled_domain: {self.num_sampled_domain_class.sum(axis=1)}")
        print(f"imb_factor_domain: {self.imb_factor['domain']}")
        print(
            f"data_imb_domain: {self.num_sampled_domain_class[0].sum() / self.num_sampled_domain_class[-1].sum()}"
        )

        print(f"num_sampled_class: {self.num_sampled_domain_class.sum(axis=0)}")
        print(f"imb_factor_class: {self.imb_factor['class']}")
        print(
            f"data_imb_class: {self.num_sampled_domain_class[:, 0].sum() / self.num_sampled_domain_class[:, -1].sum()}"
        )

        print(f"num samples: {len(self.indices)}")
        print(f"sampling time: {self.sampling_time}")
        print(f"sampling time per sample: {self.sampling_time / len(self.indices)}")

    def state2index(self, state):
        return state["domain"] * self.num["class"] + state["class"]

    def index2state(self, index):
        return {
            "domain": index // self.num["class"],
            "class": index % self.num["class"],
        }

    @staticmethod
    def assert_factors_consistency(cor_factor_max, imb_factor, num_states):
        return (1 - cor_factor_max) * imb_factor <= (
            1 - 1 / num_states
        ) and cor_factor_max < 1

    @staticmethod
    def get_cor_factors(imb_factor, cor_factor_max, num_states):
        imb_exp_factor = imb_factor ** (1 / (num_states - 1))
        min_cor_value = 1 - imb_factor * (1 - cor_factor_max)

        cor_factors = 1 - np.power(
            1 / imb_exp_factor, np.arange(num_states - 1, -1, -1)
        ) * (1 - min_cor_value)
        return cor_factors

    @staticmethod
    def get_samples_ratio_list(imb_factor, num_states):
        imb_exp_factor = (imb_factor) ** (1 / (num_states - 1))
        # return np.power(1 / imb_exp_factor, np.arange(num_states))
        ratio = np.power(1 / imb_exp_factor, np.arange(num_states))
        return ratio / ratio.sum()

    @staticmethod
    def get_transition_matrix(cor_factors, num_states):
        transition_matrix = np.tile(
            (1 - cor_factors) / (num_states - 1), (num_states, 1)
        ).T
        transition_matrix = transition_matrix.astype(np.float64)

        np.fill_diagonal(transition_matrix, cor_factors)
        return transition_matrix

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, precision=3)
    random_seed = 425
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # for cor_factor_max_class in [0.001, 0.95]:
    for cor_factor_max_class in [
        1,
    ]:
        for imb_factor_class in [1, 10]:
            for cor_factor_max_domain in [0.067, 0.85, 1]:
                for imb_factor_domain in [1, 5]:
                    # for imb_factor_domain in [
                    #    5,
                    # ]:
                    print("-" * 100)

                    # print(
                    #    f"cor_factor_max_domain: {cor_factor_max_domain}, imb_factor_domain: {imb_factor_domain}, cor_factor_max_class: {cor_factor_max_class}, imb_factor_class: {imb_factor_class}"
                    # )
                    #

                    # cor_factor_max_domain = 0.85
                    # imb_factor_domain = 1
                    # cor_factor_max_class = 0.95
                    # imb_factor_class = 1
                    # max_state_samples = 100
                    #
                    # cor_factor_max_domain = 0.85
                    # imb_factor_domain = 1
                    # cor_factor_max_class = 0.95
                    # imb_factor_class = 10
                    # max_state_samples = 100

                    # cor_factor_max_domain = 0.85
                    # imb_factor_domain = 5
                    # cor_factor_max_class = 0.95
                    # imb_factor_class = 1
                    # max_state_samples = 100

                    # cor_factor_max_domain = 0.85
                    # imb_factor_domain = 5
                    # cor_factor_max_class = 0.95
                    # imb_factor_class = 10
                    # max_state_samples = 100

                    # cor_factor_max_domain = 0.067
                    # imb_factor_domain = 1
                    # cor_factor_max_class = 0.95
                    # imb_factor_class = 1
                    # max_state_samples = 50

                    # cor_factor_max_domain = 0.067
                    # imb_factor_domain = 1
                    # cor_factor_max_class = 0.95
                    # imb_factor_class = 10
                    # max_state_samples = 50
                    if cor_factor_max_class == 0.95:
                        if cor_factor_max_domain == 0.85:
                            max_state_samples = 100
                        elif cor_factor_max_domain == 0.067 and imb_factor_domain == 1:
                            max_state_samples = 50
                        else:
                            max_state_samples = 10

                    else:
                        max_state_samples = 10

                    sampler = UniTTASampler(
                        data_source=None,
                        cor_factor_max_domain=cor_factor_max_domain,
                        imb_factor_domain=imb_factor_domain,
                        cor_factor_max_class=cor_factor_max_class,
                        imb_factor_class=imb_factor_class,
                        num_domains=15,
                        num_classes=1000,
                        max_state_samples=max_state_samples,
                        init_domain=0,
                        init_class=0,
                        domain_head2tail_list=None,
                        class_head2tail_list=None,
                        debug=True,
                    )

                    # raise ValueError("test")

    # set print width
