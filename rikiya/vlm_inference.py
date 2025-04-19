import argparse
import yaml
import copy
from vlm_model import vlm_model

import random, time, gc, re
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

class vlm_inference:
    def __init__(self, use_fine_tuned = True):
        # Ref: https://note.nkmk.me/python-if-conditional-expressions/#if-elif-else
        # Ref: https://pytorch.org/docs/stable/notes/mps.html
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        # Ref: https://huggingface.co/blog/4bit-transformers-bitsandbytes
        self.dtype =  torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Ref: CS7643 - Assignment 1 main.py
        self.parser = argparse.ArgumentParser(description='Final Project')
        self.parser.add_argument('--config', default='./config.yaml')

        # Inference using fine tuned model or not
        self.use_fine_tuned = use_fine_tuned
        # Length of output tokens
        self.max_new_tokens = 32


    def prepare_eval(self):
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

        self.test_data = test_CQ

        # Model loading path
        adapter_path = args.output_dir + "/final_model"

        # Prepare model
        if self.use_fine_tuned:
            model, processor = vlm_model(args.model_type, args.quantization,
                                         use_tuned=True, adapter_path=adapter_path)
        else:
            model, processor = vlm_model(args.model_type, args.quantization,
                                         use_fine_tuned=False, adapter_path=None)
            
        self.model = model
        self.processor = processor
        
        return None
    
    # Ref: https://huggingface.co/learn/cookbook/en/fine_tuning_smol_vlm_sft_trl#3-load-model-and-check-performance-
    def inference(self, sample):
        text_input = self.processor.apply_chat_template(sample[1:2], add_generation_prompt=True)
        image_inputs = [sample[1]['content'][0]['image']]
        model_inputs = self.processor(text=[text_input], images=image_inputs,return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
        trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text[0].strip()

    # Ref: helper function for the group project
    def normalize(self, input):
        """Clean & convert value → float | 'yes' | 'no' | None."""
        YES, NO = {"yes","y","true","correct"}, {"no","n","false","incorrect"}
        s = str(input).strip() # space on both ends
        m = re.fullmatch(r"\[['\"]?(.*?)['\"]?\]", s)
        if m:
            s = m.group(1)
        s = s.lower().replace(",","").replace("$","").replace("%","").strip()
        if s in YES:
            return "yes"
        elif s in NO:
            return "no"
        else:
            return s

    def evaluate_performance(self):
        score = 0
        num_samples = len(self.test_data)
        for i in range(num_samples):
            prediction  = (self.inference(self.test_data[i])).lower().replace('%','').removesuffix('.')
            ground_truth = (self.test_data[i][2]['content'][0]['text']).lower()

            # normalize the outputs
            prediction = self.normalize(prediction)
            ground_truth = self.normalize(ground_truth)

            # check if Yes/No question
            if ground_truth == 'yes' or ground_truth == "no":
                if prediction == ground_truth:
                    score += 1
            else:
                try:
                    prediction = float(prediction)
                    ground_truth = float(ground_truth)
                    relative_difference = abs((prediction - ground_truth)/ground_truth)
                    if relative_difference <= 0.05:
                        score+=1
                except:
                    pass

        accuracy = score / num_samples
        return accuracy
    
    def main(self):
        self.prepare_eval()
        score = self.evaluate_performance()
        return score

if __name__ == "__main__":
    instance = vlm_inference(use_fine_tuned = False)
    score = instance.main()
    print("relaxed accuracy", score)


    #instance2 = vlm_inference(use_fine_tuned = True)
    #score2 = instance2.main()
    #print("relaxed accuracy", score2)
