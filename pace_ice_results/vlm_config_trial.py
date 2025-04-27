
import argparse, time
import yaml
import copy
from vlm_model import vlm_model

import random, time, gc
import pickle
import torch
import transformers, trl, bitsandbytes, peft, accelerate, flash_attn
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# Seed setting
# Ref: https://qiita.com/north_redwing/items/1e153139125d37829d2d
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class vlm_config_trial:
    def __init__(self):
        # Ref: https://note.nkmk.me/python-if-conditional-expressions/#if-elif-else
        # Ref: https://pytorch.org/docs/stable/notes/mps.html
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        # Ref: https://huggingface.co/blog/4bit-transformers-bitsandbytes
        self.dtype =  torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Ref: CS7643 - Assignment 1 main.py
        self.parser = argparse.ArgumentParser(description='Final Project')
        self.parser.add_argument('--config', default='./config.yaml')

    def main(self):
        config_file, time_taken, log_history = self.run()

        print("config_file", config_file)
        print("time_taken", time_taken)
        print("log_history", log_history)

        return None

    def run(self):
        # Prepare argparse #################
        # Ref: CS7643 - Assignment 1 main.py
        global args
        args = self.parser.parse_args()
        config_file = str(args.config)
        with open(args.config) as f:
            # Ref: https://qiita.com/TakamiChie/items/fbdbfc1dc659efb24998
            config = yaml.load(f, Loader=yaml.SafeLoader)

        for key in config:
            for k, v in config[key].items():
                setattr(args, k, v)
        ####################################

        # Get data from pickle files
        # Ref: https://note.nkmk.me/python-pickle-usage/
        with open(args.data_path+'test_CQ_'+args.data_type+'.pkl', 'rb') as f:
            test_CQ = pickle.load(f)
        with open(args.data_path+'train_CL_'+args.data_type+'.pkl', 'rb') as f:
            train_CL = pickle.load(f)
        with open(args.data_path+'val_CL_'+args.data_type+'.pkl', 'rb') as f:
            val_CL = pickle.load(f)
        
        print("args.lora: ", args.lora)
        start_time = time.time()
        time.sleep(3)
        end_time = time.time()
        time_taken = (end_time - start_time)/60

        # Saving data
        log_history = "fake log_history"

        return config_file, time_taken, log_history

if __name__ == "__main__":
    instance = vlm_config_trial()
    instance.main()
