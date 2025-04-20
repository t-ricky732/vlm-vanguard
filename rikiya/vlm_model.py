import random
import torch
import transformers
from transformers import Idefics3ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# Seed setting
# Ref: https://qiita.com/north_redwing/items/1e153139125d37829d2d
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class vlm_model():
    def __init__(self, model_type, quantization, use_tuned=False, adapter_path=None):
        # Ref: https://note.nkmk.me/python-if-conditional-expressions/#if-elif-else
        # Ref: https://pytorch.org/docs/stable/notes/mps.html
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        # Ref: https://huggingface.co/blog/4bit-transformers-bitsandbytes
        self.dtype =  torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # model loading
        self.pretrained_model = model_type # such as "HuggingFaceTB/SmolVLM-256M-Instruct" from yaml file
        self.quantization = quantization # if True -> do QLoRA, from yaml file
        self.use_tuned = use_tuned
        if self.use_tuned:
            self.adapter_path = adapter_path

        # Note: we decided to use only 4 bit in this project
        self.four_bit = True # True if 4 bit is used, False = 8 bit


    # Ref: https://huggingface.co/learn/cookbook/en/fine_tuning_smol_vlm_sft_trl#41-load-the-quantized-model-for-training-
    def get_model(self):
        # common parameters
        model_kwargs = {}
        model_kwargs['device_map'] = "auto"
        model_kwargs['torch_dtype'] = self.dtype
        model_kwargs['_attn_implementation'] = "flash_attention_2"

        # For bitsandbytes quantization
        if (self.quantization) and (not self.use_tuned): # Use quantization => use QLoRA
        #if self.quantization: # Use quantization => use QLoRA
            if self.four_bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype = self.dtype
                    )
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_use_double_quant=True,
                    bnb_8bit_quant_type="nf8",
                    bnb_8bit_compute_dtype = self.dtype
                    )
            # Put bitsandbytes config into kwargs
            model_kwargs["quantization_config"] = bnb_config

        # Get model
        model = Idefics3ForConditionalGeneration.from_pretrained(
            self.pretrained_model,
            **model_kwargs
        )

        # Get processor
        processor = AutoProcessor.from_pretrained(self.pretrained_model)
        # Ref: Ryan
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            if hasattr(processor, 'pad_token') and processor.pad_token is None:
                processor.pad_token = processor.tokenizer.eos_token

        if self.use_tuned:
            model.load_adapter(self.adapter_path)

        return model, processor

if __name__ == "__main__":
    pass
