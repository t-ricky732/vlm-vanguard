import argparse
import yaml
import copy
from vlm_model import vlm_model
import pandas as pd

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

        self.json_output = "./chartqa_evaluation_results_comparison.json"


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
                                            eos_token_id=self.processor.tokenizer.eos_token_id))
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
    
    def save_json(self):
        results_list = []
        for sample in self.test_data:
            entry = {"id": sample.get("img_idx", "N/A"),
                    "question": sample.get("query"),
                    "ground_truth": str(sample.get("label"))} # Convert GT to string here

            # Basic validation of core sample data needed for processing
            if not all([entry["id"] != "N/A", entry["question"], entry["ground_truth"] is not None, isinstance(sample.get("image"), Image.Image)]):
                print(f"Warning: Skipping sample {entry['id']} due to missing/invalid core data.")
                continue # Skip this sample

            entry[f"predicted_answer_{self.model_name}"] = self.inferece(sample)
            results_list.append(entry)
        # --- End Prediction Loop ---   

        # --- Save results_df to JSON ---
        if results_list:
            results_df = pd.DataFrame(results_list)
            if self.json_output in locals() and self.json_output:
                try:
                    os.makedirs(os.path.dirname(self.json_output), exist_ok=True)
                    results_df.to_json(self.json_output, orient="records", indent=2)
                    print(f"\nEvaluation results (raw predictions) saved to: {self.json_output}")
                except Exception as e_save:
                    print(f"\nERROR saving evaluation results to {self.json_output}: {e_save}")
            else:
                print("\nWarning: EVAL_OUTPUT_FILE not defined, results not saved.")
        return None
    
    # --- Advanced accuracy metrics -------------------------------
    # 2.  Helpers -------------------------------------------------------------------
    def to_scalar(self, v):
        YES, NO      = {"yes","y","true","correct"}, {"no","n","false","incorrect"}
        TOL, EPS     = 0.05, 1e-9                  # ±5 %, tiny tolerance for 0

        """Clean & convert value → float | 'yes' | 'no' | None."""
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
        return abs(gt - pred) <= (abs(gt) * TOL or EPS)
    
    def aggregate(self):
        BASE_OUTPUT_DIR = Path('./')
        OUTPUT_FOLDER_PREFIX = self.output_dir
        EVAL_FILENAME = "chartqa_evaluation_results_comparison.json"

        search_pattern = f"{OUTPUT_FOLDER_PREFIX}*/{EVAL_FILENAME}"
        result_files = list(BASE_OUTPUT_DIR.glob(search_pattern))

        # if not result_files:
        #     raise FileNotFoundError("No evaluation result files found.")

        aggregated_dfs = []
        base_models_added = set()

        for file_path in result_files:
            run_label = file_path.parent.name.replace(f"{OUTPUT_FOLDER_PREFIX}-", "")
            parts = run_label.split('-')
            base_model_name = '-'.join(parts[:-1]) if len(parts) >= 2 else "UnknownBase"
            base_label = f"{base_model_name}-Original"

            df = pd.read_json(file_path)

            columns = ['id', 'question', 'ground_truth'] if not aggregated_dfs else ['id']
            rename_dict = {}

            if 'predicted_answer_finetuned' in df:
                columns.append('predicted_answer_finetuned')
                rename_dict['predicted_answer_finetuned'] = f"Pred_{run_label}"

            if base_model_name not in base_models_added and 'predicted_answer_base' in df:
                columns.append('predicted_answer_base')
                rename_dict['predicted_answer_base'] = f"Pred_{base_label}"
                base_models_added.add(base_model_name)

            df_subset = df[columns].rename(columns=rename_dict)
            aggregated_dfs.append(df_subset)

        if aggregated_dfs:
            final_comparison_df = reduce(lambda left, right: pd.merge(left, right, on='id', how='outer'), aggregated_dfs)
            final_comparison_df.ffill(inplace=True)
            final_comparison_df.bfill(inplace=True)
        else:
            print("No evaluation results found to aggregate.")
            final_comparison_df = pd.DataFrame()
        df = final_comparison_df.copy()
        df["GT_proc"] = df["ground_truth"].map(self.to_scalar)

        keep = df["GT_proc"].isin(["yes","no"]) | df["GT_proc"].apply(lambda x: isinstance(x,(int,float)))
        df_filt = df[keep]
        if df_filt.empty:
            sys.exit("  No yes/no/numeric ground‑truth rows after processing.")

        # Metrics per run -----------------------------------------------------------
        PRED_PREFIX  = "Pred_"                     # rename if your columns differ
        results = {}
        for col in [c for c in df.columns if c.startswith(PRED_PREFIX)]:
            proc = f"{col}_proc"
            df[proc] = df[col].map(self.to_scalar)

            valid   = df[["GT_proc", proc]].dropna()
            numeric = valid[valid["GT_proc"].apply(lambda x:isinstance(x,(int,float))) &
                            valid[proc].apply(lambda x:isinstance(x,(int,float)))]

            em  = 100 * (valid["GT_proc"] == valid[proc]).mean() if not valid.empty else 0
            rel = 100 * numeric.apply(lambda r: self.relaxed(r["GT_proc"], r[proc]), axis=1).mean() if not numeric.empty else 0

            run = col.replace(PRED_PREFIX, "")
            results[run] = {"EM (%)": round(em,2),
                            "Relaxed Num (%)": round(rel,2),
                            "# Numeric": len(numeric),
                            "# Valid": len(valid)}

            print(f"{run:<15} EM={em:6.2f}%  Relaxed={rel:6.2f}%  ({len(numeric)} num / {len(valid)} valid)")

        return em, rel, numeric, valid
    
    def main(self):
        self.prepare_eval()
        #score = self.evaluate_performance()
        #return score
        self.save_json()
        em, rel, numeric, valid = self.aggregate()
        return em, rel, numeric, valid
    

if __name__ == "__main__":
    start_time = time.time()
    #instance = vlm_inference(use_fine_tuned = False)
    instance = vlm_inference(use_fine_tuned = True)
    #em, rel, EM_pair, rel_pair = instance.main()
    em, rel, numeric, valid = instance.main()
    end_time = time.time()
    time_taken = (end_time - start_time)/60
    
    print("EM (%)", em)
    print("relaxed accuracy (%)", rel)
    #print("EM - correct / # of valid", EM_pair)
    #print("rel. acc. - correct / # of valid", rel_pair)
    print("EM -  # of valid", numeric)
    print("rel.- # of valid", valid)
    print("inference time:", time_taken)