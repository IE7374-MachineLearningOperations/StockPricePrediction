from typing import Union, Dict

import torch
import time
import logging
from datetime import datetime
import os
import pathlib

from omegaconf import OmegaConf
import random
import wandb
import yaml


def get_py_logger(dataset_name: str, job_id: str = None):
    def _get_logger(logger_name, log_path, level=logging.INFO):
        logger = logging.getLogger(logger_name)  # global variance?
        formatter = logging.Formatter("%(asctime)s : %(message)s")

        fileHandler = logging.FileHandler(log_path, mode="w")
        fileHandler.setFormatter(formatter)  # `formatter` must be a logging.Formatter

        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(fileHandler)
        logger.addHandler(streamHandler)
        return logger

    logger_path = "logs/{dataset_name}"
    logger_name_format = "%Y-%m-%d--%H-%M-%S.%f"

    logger_name = f"{job_id} - {datetime.now().strftime(logger_name_format)}.log"

    logger_folder = logger_path.format(dataset_name=dataset_name)
    pathlib.Path(logger_folder).mkdir(parents=True, exist_ok=True)

    logger_path = os.path.join(logger_folder, logger_name)
    logger = _get_logger(logger_name, logger_path)
    return logger  # , logger_name, logger_path


def init_wandb(project_name):
    # wandb.run.dir
    # https://docs.wandb.ai/guides/track/advanced/save-restore
    with open("dags/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    try:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb.login(key=config["WANDB_API_KEY"])
    except Exception as e:
        print(f"--- was trying to log in Weights and Biases... e={e}")

    watermark = time.strftime("%I-%M%p-%B-%d-%Y")
    wandb.init(
        project=project_name,
        entity="vynguyen-analyst-neu",
        name=watermark,
        settings=wandb.Settings(start_method="fork"),
    )

    # if exists(model):
    ## TODO: fix later
    #     wandb.watch(model, log_freq=log_freq, log_graph=True, log="all")  # log="all" to log gradients and parameters
    return watermark


class DummyLogger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.use_ddp = cfg.hardware.multi_gpus == "ddp"

    def info(self, msg: Union[Dict, str], sep=", ", padding_space=False, pref_msg: str = "", *args, **kwargs):

        if isinstance(msg, Dict):
            msg_str = (
                pref_msg
                + " "
                + sep.join(f"{k} {round(v, 4) if isinstance(v, int) else v}" for k, v in msg.items())
            )
            if padding_space:
                msg_str = sep + msg_str + " " + sep
            if self.use_ddp:
                msg_str = f'\t[GPU{os.environ["LOCAL_RANK"]}]: {msg_str}'
            print(msg_str)
        else:
            if self.use_ddp:
                msg = f'\t[GPU{os.environ["LOCAL_RANK"]}]: {msg}'
            print(msg)

    def log_imgs(self, *args, **kwargs):
        pass

    def log_config(self, *args, **kwargs):
        pass

    def update_best_result(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


class MyLogging:
    def __init__(self, project_name):
        self.use_py_logger = False
        self.use_wandb = True
        self.use_ddp = False

        self.py_logger = None
        if self.use_wandb:
            init_wandb(project_name)

    def info(
        self,
        msg: Union[Dict, str],
        use_wandb=None,
        sep=", ",
        padding_space=False,
        pref_msg: str = "",
    ):

        if isinstance(msg, Dict):
            msg_str = (
                pref_msg
                + " "
                + sep.join(f"{k} {round(v, 4) if isinstance(v, int) else v}" for k, v in msg.items())
            )
            if padding_space:
                msg_str = sep + msg_str + " " + sep
            if self.use_ddp:
                msg_str = f'[GPU{os.environ["LOCAL_RANK"]}]: {msg_str}'

            if use_wandb:
                wandb.log(msg)

            if self.use_py_logger:
                self.py_logger.info(msg_str)
            else:
                print(msg_str)
        else:
            if self.use_py_logger:
                self.py_logger.info(msg)
            else:
                if self.use_ddp:
                    msg = f'[GPU{os.environ["LOCAL_RANK"]}]: {msg}'
                print(msg)

    def log_imgs(self, x, y, y_hat, classes, max_scores, name: str):
        columns = ["image", "pred", "label", "score", "correct"]
        data = []
        for j, image in enumerate(x, 0):
            # pil_image = Image.fromarray(image, mode="RGB")
            data.append(
                [
                    wandb.Image(image[:3]),
                    classes[y_hat[j].item()],
                    classes[y[j].item()],
                    max_scores[j].item(),
                    y_hat[j].item() == y[j].item(),
                ]
            )

        table = wandb.Table(data=data, columns=columns)
        wandb.log({name: table})

    def log_config(self, config):
        wandb.config.update(OmegaConf.to_container(config))  # , allow_val_change=True)

    def update_best_result(self, msg: str, metric, val, use_wandb=None):

        if self.use_py_logger:
            self.py_logger.info(msg)
        else:
            print(msg)
        if use_wandb:
            wandb.run.summary[metric] = val

    def finish(
        self,
        use_wandb=None,
        msg_str: str = None,
        model=None,
        model_name: str = "",
        # dummy_batch_x=None,
    ):
        if use_wandb:
            ## skip for now, not necessary, and takes too much space
            # if model_name:
            #     wandb.save(model_name)
            #     print(f"saved pytorch model {model_name}!")

            # if exists(model):
            #     try:
            #         # https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb#scrollTo=j64Lu7pZcubd
            #         if self.args.hardware.multi_gpus == "DataParallel":
            #             model = model.module
            #         torch.onnx.export(model, dummy_batch_x, "model.onnx")
            #         wandb.save("model.onnx")
            #         print("saved to model.onnx!")
            #     except Exception as e:
            #         print(e)
            wandb.finish()
