import logging
import torch
import argparse

from core.configs import cfg
from core.utils import mkdir, setup_logger, set_random_seed, clear_loggers

from core.model import build_model
from core.data import build_loader
from core.optim import build_optimizer
from core.adapter import build_adapter
from tqdm import tqdm
from setproctitle import setproctitle
from sklearn.metrics import confusion_matrix
import numpy as np

# import wandb

import time

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


def testTimeAdaptation(cfg, loader, processor, logger):
    # model, optimizer
    model = build_model(cfg)
    model.eval()
    optimizer = build_optimizer(cfg)
    tta_adapter = build_adapter(cfg)
    tta_model = tta_adapter(cfg, model, optimizer)
    tta_model.cuda()

    # domain_preds = torch.empty(0, dtype=torch.long).cuda()
    domain_gt = torch.empty(0, dtype=torch.long).cuda()

    preds = []
    gts = []

    times = []

    domain_num = loader.dataset.domain_id_to_name.keys().__len__()
    class_num = cfg.CORRUPTION.NUM_CLASS

    tbar = tqdm(loader)

    for batch_id, data_package in enumerate(tbar):
        data, label, domain = (
            data_package["image"],
            data_package["label"],
            data_package["domain"],
        )
        if batch_id == 0:
            logger.info("first batch")
            logger.info(f"label: {label}")
            logger.info(f"domain: {domain}")

        if len(label) == 1:
            torch.cuda.synchronize()
            start = time.time()
            continue  # ignore the final single point
        data, label, domain = data.cuda(), label.cuda(), domain.cuda()

        torch.cuda.synchronize()
        start = time.time()

        if cfg.ADAPTER.NAME == "unitta_bdn":
            output = tta_model([data, domain])

        elif cfg.ADAPTER.NAME == "unitta":
            output, num_domain = tta_model(data)

            # wandb.log({"num_domain": num_domain}, commit=False, step=batch_id)

        else:
            output = tta_model(data)

        domain_gt = torch.cat((domain_gt, domain))

        torch.cuda.synchronize()
        times.extend([(time.time() - start) / len(label)] * len(label))

        predict = torch.argmax(output, dim=1)
        accurate = predict == label

        preds.extend((predict.cpu() + domain.cpu() * class_num).numpy().tolist())
        gts.extend((label.cpu() + domain.cpu() * class_num).numpy().tolist())

        processor.process(accurate, domain)

        if batch_id % 10 == 0:
            if "tta_model" in vars() and hasattr(tta_model, "mem"):
                tbar.set_postfix(
                    acc=processor.cumulative_acc(), bank=tta_model.mem.get_occupancy()
                )
            else:
                tbar.set_postfix(acc=processor.cumulative_acc())
            # wandb.log({"acc": processor.cumulative_acc()}, commit=True, step=batch_id)

        else:
            pass
            # wandb.log({"acc": processor.cumulative_acc()}, commit=False, step=batch_id)

    processor.calculate()

    logger.info(f"All Results\n{processor.info()}")

    cm = confusion_matrix(gts, preds)
    acc_per_class = (np.diag(cm) + 1e-5) / (cm.sum(axis=1) + 1e-5)

    str_ = ""
    catAvg = np.zeros(domain_num)
    for i in range(domain_num):
        catAvg[i] = acc_per_class[i * class_num : (i + 1) * class_num].mean()
        str_ += "%d %.2f\n" % (i, catAvg[i] * 100.0)

        key = list(processor.label2name.keys())[i]

        # wandb.run.summary[f"err_{processor.label2name[key]}"] = (
        #    1 - processor.result_per_class[key]
        # ) * 100
        # wandb.run.summary[f"catAvgErr_{processor.label2name[key]}"] = (
        #    1.0 - catAvg[i]
        # ) * 100.0

    # wandb.run.summary["err_total"] = (1 - processor.cumulative_acc()) * 100
    # wandb.run.summary["catAvgErr_total"] = 100.0 - catAvg.mean() * 100.0

    # str_ += "Avg: %.2f\n" % (catAvg.mean() * 100.)
    logger.info("per domain catAvg:\n" + str_)
    logger.info(f"per domain catAvgAcc: {catAvg.mean() * 100.:.2f}")
    logger.info(f"per domain catAvgErr: {100. - catAvg.mean() * 100.:.2f}")

    print("average adaptation time:", np.mean(times))
    # wandb.run.summary["average_adaptation_time"] = np.mean(times)


def main():
    parser = argparse.ArgumentParser("Pytorch Implementation for Test Time Adaptation!")
    parser.add_argument(
        "-acfg",
        "--adapter-config-file",
        metavar="FILE",
        default="",
        help="path list of adapter config files",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "-dcfg",
        "--dataset-config-file",
        metavar="FILE",
        default="",
        help="path to dataset config file",
        type=str,
    )
    parser.add_argument(
        "-ocfg",
        "--order-config-file",
        metavar="FILE",
        default="",
        help="path to order config file",
        type=str,
    )
    parser.add_argument(
        "-pcfg",
        "--protocol-config-file",
        metavar="FILE",
        default="",
        help="path to protocol config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="modify the configuration by command line",
        nargs=argparse.REMAINDER,
        default=None,
    )

    args = parser.parse_args()

    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip("\r\n")

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.dataset_config_file)
    if not args.order_config_file == "":
        cfg.merge_from_file(args.order_config_file)
    cfg.merge_from_file(args.protocol_config_file)
    cfg.merge_from_list(args.opts)

    set_random_seed(cfg.SEED)

    loader, processor = build_loader(
        cfg, cfg.CORRUPTION.DATASET, cfg.CORRUPTION.TYPE, cfg.CORRUPTION.SEVERITY
    )
    mark = cfg.MARK
    # wandb.login(key="")

    adapter_config_file = args.adapter_config_file

    processor.reset()
    cfg.defrost()
    cfg.merge_from_file(adapter_config_file)
    cfg.MARK = f"{mark}_{cfg.ADAPTER.NAME}"
    cfg.OUTPUT_DIR = f"output/{cfg.MARK}"
    cfg.freeze()

    ds = cfg.CORRUPTION.DATASET
    adapter = cfg.ADAPTER.NAME
    setproctitle(f"TTA:{ds:>8s}:{adapter:<10s}")

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

        # wandb.init(
        #    # set the wandb project where this run will be logged
        #    project="UniTTA",
        #    name=cfg.WANDB.NAME,
        #    mode=cfg.WANDB.MODE,
        #    # track hyperparameters and run metadata
        #    config=cfg,
        #    reinit=True,
        # )

    clear_loggers()
    logger = setup_logger("TTA", cfg.OUTPUT_DIR, 0, filename=cfg.LOG_DEST)
    logger.info(args)

    logger.info(
        f"Loaded configuration file: \n"
        f"\tadapter: {args.adapter_config_file}\n"
        f"\tdataset: {args.dataset_config_file}\n"
        f"\torder: {args.order_config_file}"
    )
    logger.info("Running with config:\n{}".format(cfg))

    testTimeAdaptation(cfg, loader, processor, logger)


if __name__ == "__main__":
    main()
