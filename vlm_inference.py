import argparse
import yaml
import copy
from vlm_model import vlm_model
import pandas as pd
from word2number import w2n

import random, time, gc, re, os, sys
from pathlib import Path
from functools import reduce
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

        self.json_output = os.path.join('./', "chartqa_evaluation_results_comparison.json")


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

        # For displaying parameters (for PACE record)
        print("r", args.r)
        print("lora_alpha", args.lora_alpha)
        print("lora_dropout", args.lora_dropout)
        print("target_modules", args.target_modules)

        # Get data from pickle files
        # Ref: https://note.nkmk.me/python-pickle-usage/
        with open(args.data_path+'test_CQ_'+args.data_type+'.pkl', 'rb') as f:
            test_CQ = pickle.load(f)

        self.model_name = args.model_type
        self.output_dir = args.output_dir
        self.test_data = test_CQ

        # Model loading path
        adapter_path = args.output_dir + "/final_model"

        # Prepare model
        if self.use_fine_tuned:
            model_instance = vlm_model(args.model_type, args.quantization,
                                       use_tuned=True, adapter_path=adapter_path)
            model, processor = model_instance.get_model()
        else:
            model_instance = vlm_model(args.model_type, args.quantization,
                                       use_tuned=False, adapter_path=None)
            model, processor = model_instance.get_model()            

        self.model = model
        self.processor = processor
        
        return None
    
    # Ref: https://huggingface.co/learn/cookbook/en/fine_tuning_smol_vlm_sft_trl#3-load-model-and-check-performance-
    def inference(self, sample):
        text_input = self.processor.apply_chat_template(sample[1:2], add_generation_prompt=True)
        image_inputs = [sample[1]['content'][0]['image']]
        model_inputs = self.processor(text=[text_input], images=image_inputs,return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens,
                                            # test_notebook_5 compatibility
                                            do_sample=False,
                                            pad_token_id=self.processor.tokenizer.pad_token_id,
                                            eos_token_id=self.processor.tokenizer.eos_token_id)
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
        rel_score = 0
        EM_score = 0
        EM_denominator = 0
        rel_denominator = 0
        num_samples = len(self.test_data)
        for i in range(num_samples):
            prediction  = (self.inference(self.test_data[i])).lower().replace('%','').removesuffix('.')
            ground_truth = (self.test_data[i][2]['content'][0]['text']).lower()

            # normalize the outputs
            prediction = self.normalize(prediction)
            ground_truth = self.normalize(ground_truth)

            # check if Yes/No question
            if (ground_truth == 'yes' or ground_truth == "no") and (prediction == 'yes' or prediction == "no"):
                EM_denominator +=1
                if prediction == ground_truth:
                    EM_score += 1
            else:
                try:
                    prediction = float(prediction)
                    ground_truth = float(ground_truth)
                    relative_difference = abs((prediction - ground_truth)/ground_truth)
                    EM_denominator +=1
                    rel_denominator +=1
                    if prediction == ground_truth:
                        EM_score+=1
                    if relative_difference <= 0.05:
                        rel_score+=1
                except:
                    pass

        EM = EM_score / EM_denominator
        rel_accuracy = rel_score / rel_denominator
        return EM, rel_accuracy, (EM_score, EM_denominator), (rel_score, rel_denominator)

    # --- Advanced accuracy metrics -------------------------------
    # 2.  Helpers -------------------------------------------------------------------
    def to_scalar(self, v):
        YES, NO      = {"yes","y","true","correct"}, {"no","n","false","incorrect"}

        """Clean & convert value -> float | 'yes' | 'no' | None."""
        if pd.isna(v): return None
        s = str(v).strip()
        m = re.fullmatch(r"\[['\"]?(.*?)['\"]?\]", s)
        if m: s = m.group(1)
        s = s.lower().replace(",","").replace("$","").replace("%","").strip()
        if s in YES: return "yes"
        if s in NO:  return "no"
        for fn in (float, w2n.word_to_num):
            try: return float(fn(s))
            except Exception: pass
        return None

    def relaxed(self, gt, pred):
        TOL, EPS     = 0.05, 1e-9 # +-5 %, tiny tolerance for 0s
        return abs(gt - pred) <= (abs(gt) * TOL or EPS)

    def save_json_and_output(self):
        results_list = []
        i=0
        for sample in self.test_data:
            entry = {"id": i,
                    "question": sample[1]['content'][1]['text'],
                    "ground_truth": sample[2]['content'][0]['text']} # Convert GT to string here
            i+=1
            # Basic validation of core sample data needed for processing
            #if not all([entry["id"] != "N/A", entry["question"], entry["ground_truth"] is not None, isinstance(sample[1]['content'][0]['image'], Image.Image)]):
            #    print(f"Warning: Skipping sample {entry['id']} due to missing/invalid core data.")
            #    continue # Skip this sample

            entry[f"predicted_answer_{self.model_name}"] = self.inference(sample)
            results_list.append(entry)
        # --- End Prediction Loop ---

        # --- Save results_df to JSON ---
        if results_list:
            results_df = pd.DataFrame(results_list)
            os.makedirs(os.path.dirname(self.json_output), exist_ok=True)
            results_df.to_json(self.json_output, orient="records", indent=2)

        df = results_df.copy()
        df["GT_proc"] = df["ground_truth"].map(self.to_scalar)
        keep = df["GT_proc"].isin(["yes","no"]) | df["GT_proc"].apply(lambda x: isinstance(x,(int,float)))
        df_filt = df[keep]
        if df_filt.empty:
            sys.exit("  No yes/no/numeric ground-truth rows after processing.")

        # Metrics per run -----------------------------------------------------------
        PRED_PREFIX  = "predicted_"                     # rename if your columns differ
        results = {}

        for col in [c for c in df.columns if c.startswith(PRED_PREFIX)]:
            proc = f"{col}_proc"
            df[proc] = df[col].map(self.to_scalar)
            valid   = df[["GT_proc", proc]].dropna()
            numeric = valid[valid["GT_proc"].apply(lambda x:isinstance(x,(int,float))) &
                            valid[proc].apply(lambda x:isinstance(x,(int,float)))]

            em  = 100 * (valid["GT_proc"] == valid[proc]).mean() if not valid.empty else 0
            rel = 100 * numeric.apply(lambda r: self.relaxed(r["GT_proc"], r[proc]), axis=1).mean() if not numeric.empty else 0

            run = col.replace(PRED_PREFIX, "").replace("answer_","")
            results[run] = {"EM (%)": round(em,2),
                            "Relaxed Num (%)": round(rel,2),
                            "# Numeric": len(numeric),
                            "# Valid": len(valid)}

            print(f"{run:<15} EM={em:6.2f}%  Relaxed={rel:6.2f}%  ({len(numeric)} num / {len(valid)} valid)")

        return None

    def main(self):
        self.prepare_eval()
        #score = self.evaluate_performance()
        #return score
        self.save_json_and_output()
        return None

if __name__ == "__main__":
    start_time = time.time()
    #instance = vlm_inference(use_fine_tuned = False)
    instance = vlm_inference(use_fine_tuned = True)
    #em, rel, EM_pair, rel_pair = instance.main()
    instance.main()
    end_time = time.time()
    time_taken = (end_time - start_time)/60
    
    #print("EM (%)", em)
    #print("relaxed accuracy (%)", rel)
    #print("EM - correct / # of valid", EM_pair)
    #print("rel. acc. - correct / # of valid", rel_pair)
    #print("EM -  # of valid", numeric)
    #print("rel.- # of valid", valid)
    print("inference time:", time_taken)