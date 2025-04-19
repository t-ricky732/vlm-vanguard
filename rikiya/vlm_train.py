import argparse
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


class vlm_train:
    def __init__(self):
        # Ref: https://note.nkmk.me/python-if-conditional-expressions/#if-elif-else
        # Ref: https://pytorch.org/docs/stable/notes/mps.html
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        # Ref: https://huggingface.co/blog/4bit-transformers-bitsandbytes
        self.dtype =  torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Ref: CS7643 - Assignment 1 main.py
        self.parser = argparse.ArgumentParser(description='Final Project')
        self.parser.add_argument('--config', default='./config.yaml')

    # Ref: https://huggingface.co/learn/cookbook/en/fine_tuning_smol_vlm_sft_trl#3-load-model-and-check-performance-
    def clear_cuda(self):
        gc.collect()
        time.sleep(1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(1)
            gc.collect()
            time.sleep(1)
        return None

    def main(self):
        config_file, time_taken, log_history = self.run()

        return None

    def run(self):
        # Clear GPU
        self.clear_cuda()

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
        with open(args.data_path+'train_CL_'+args.data_type+'.pkl', 'rb') as f:
            train_CL = pickle.load(f)
        with open(args.data_path+'val_CL_'+args.data_type+'.pkl', 'rb') as f:
            val_CL = pickle.load(f)
        
        # Prepare model
        model_instance = vlm_model(args.model_type, args.quantization, use_tuned=False, adapter_path=None)
        model, processor = model_instance.get_model()

        # Ref: https://huggingface.co/learn/cookbook/en/fine_tuning_smol_vlm_sft_trl#3-load-model-and-check-performance-
        if args.lora:
            if args.quantization:
                model = prepare_model_for_kbit_training(model)

            peft_config = LoraConfig(
                r=args.r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.target_modules,
                # following parameters are fixed
                #init_lora_weights="gaussian", default is used by Ryan
                bias="none",
                task_type="CAUSAL_LM",
                )
            model = get_peft_model(model, peft_config)
        
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

        training_args = SFTConfig(
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            dataset_text_field="",
            packing=False,
            dataset_kwargs={"skip_prepare_dataset": True},

            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            
            optim=args.optim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,
            save_total_limit=2,
            save_only_model=(peft_config is not None),

            bf16=(self.dtype == torch.bfloat16),
            fp16=(self.dtype == torch.float16),
            seed=seed,
            remove_unused_columns=False,
            )
        
        def format_and_batch(examples):
            image_token_id = processor.tokenizer.additional_special_tokens_ids[
                processor.tokenizer.additional_special_tokens.index("<image>")]
            images = []
            texts = []
            for example in examples:
                images.append(example[1]["content"][0]["image"])
                texts.append(processor.apply_chat_template(example, add_generation_prompt=False, tokenize=False))

            batch = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                #padding="longest",
                padding=True,
                #truncation=True,
                #max_length=args.max_seq_length,
                )
            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            labels[labels == image_token_id] = -100
            batch["labels"] = labels

            return batch

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_CL,
            eval_dataset=val_CL,
            data_collator=format_and_batch,
            peft_config=peft_config if args.lora else None,
        )

        # Execute training
        start_time = time.time()
        train_result = trainer.train()
        end_time = time.time()
        time_taken = (end_time - start_time)/60

        # Saving data
        FINAL_MODEL_DIR = args.output_dir + "/final_model"
        FINAL_PROCESSOR_DIR = args.output_dir + "/final_processor"
        trainer.save_model(FINAL_MODEL_DIR)
        processor.save_pretrained(FINAL_PROCESSOR_DIR)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        log_history = trainer.state.log_history

        # Clear GPU
        self.clear_cuda()

        return config_file, time_taken, log_history

if __name__ == "__main__":
    instance = vlm_train()
    instance.main()
