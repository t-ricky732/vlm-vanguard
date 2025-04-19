import random, time, gc, os
import pickle
import torch
import datasets
from datasets import load_dataset
from PIL import Image

# Seed setting
# Ref: https://qiita.com/north_redwing/items/1e153139125d37829d2d
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class vlm_data:
    def __init__(self, selector):
        # Ref: https://note.nkmk.me/python-if-conditional-expressions/#if-elif-else
        # Ref: https://pytorch.org/docs/stable/notes/mps.html
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        # Ref: https://huggingface.co/blog/4bit-transformers-bitsandbytes
        self.dtype =  torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # system message
        self.selector = selector

        # chartllama image loading
        self.image_size = 384
        self.chartllama_path = "chartllama_data/"
        self.chartllama_train_frac = 0.8 # 0.8 => 80% train, 20% validation


    def system_message(self):
        if self.selector == 'long':
            #Ref: https://huggingface.co/learn/cookbook/en/fine_tuning_smol_vlm_sft_trl#3-load-model-and-check-performance-
            message = "You are a Vision Language Model specialized in interpreting visual data from chart images. \
                Your task is to analyze the provided chart image and respond to queries with concise answers, \
                    usually a single word, number, or short phrase. The charts include a variety of types \
                        (e.g., line charts, bar charts) and contain colors, labels, and text. \
                            Focus on delivering accurate, succinct answers based on the visual information. \
                                Avoid additional explanation unless absolutely necessary."
        elif self.selector == 'middle':
            message = "Please answer the question using only one word or a number based on the provided chart."
        elif self.selector == 'short':
            message = "Consider the input."
        elif self.selector == 'other':
            message = "Have fun!"
        else:
            raise RuntimeError("data selector is incorrect")
        return message
    

    #Ref: https://huggingface.co/learn/cookbook/en/fine_tuning_smol_vlm_sft_trl#3-load-model-and-check-performance-
    def format_dataset(self, sample):
        output = [
            {"role": "system","content": [{"type": "text", "text": self.system_message()}]},
            {"role": "user","content": [{"type": "image","image": sample["image"]},{"type": "text","text": sample["query"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": sample["label"][0]}]},
        ]
        return output


    def load_chartQA(self):
        # Ref: https://huggingface.co/datasets/HuggingFaceM4/ChartQA
        data_id = "HuggingFaceM4/ChartQA"
        #train_CQ, eval_CQ, test_CQ = load_dataset(data_id, split=["train[:10%]", "val[:10%]", "test[:10%]"])
        test_CQ = load_dataset(data_id, split="test")

        #Formatting
        # Ref: https://huggingface.co/learn/cookbook/en/fine_tuning_smol_vlm_sft_trl#41-load-the-quantized-model-for-training-
        #train_CQ = [self.format_dataset(sample) for sample in train_CQ]
        #eval_CQ = [self.format_dataset(sample) for sample in eval_CQ]
        test_CQ = [self.format_dataset(sample) for sample in test_CQ]

        #return train_CQ, eval_CQ, test_CQ
        return test_CQ


    # Ref: https://huggingface.co/learn/cookbook/en/fine_tuning_smol_vlm_sft_trl#41-load-the-quantized-model-for-training-
    def format_chartllama(self, sample):
        #print(sample)
        image_path = self.chartllama_path + sample['image']
        image_object = Image.open(image_path).resize((self.image_size, self.image_size)).convert('RGB')
        question = sample['conversations'][0]['value'].strip('<image>').strip('\n')
        answer = sample['conversations'][1]['value']
        
        output = [
            {"role": "system","content": [{"type": "text", "text": self.system_message()}]},
            {"role": "user","content": [{"type": "image", "image": image_object}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        return output


    # Ref: https://huggingface.co/learn/cookbook/en/fine_tuning_smol_vlm_sft_trl#41-load-the-quantized-model-for-training-
    def load_chartllma(self):
        #This function assumes that the data are stored locally in chartllama_data
        data_files=[self.chartllama_path + 'box_chart_100examples_simplified_qa.json',
                    self.chartllama_path + 'candlestick_chart_100examples_simplified_qa.json',
                    self.chartllama_path + 'funnel_chart_100examples_simplified_qa.json',
                    self.chartllama_path + 'gantt_chart_100examples_simplified_qa.json',
                    self.chartllama_path + 'heatmap_chart_100examples_simplified_qa.json',
                    self.chartllama_path + 'polar_chart_100examples_simplified_qa.json',
                    self.chartllama_path + 'scatter_chart_100examples_simplified_qa.json'
        ]
        # Loading json files
        dataset_CL = load_dataset('json', data_files=data_files)

        # Change data format
        CL_formated = [self.format_chartllama(sample) for sample in dataset_CL['train']]

        # train/val split, assume self.chartllama_train_frac => % of train data, with shuffle
        train_size = int(len(CL_formated)*self.chartllama_train_frac)
        val_size = len(CL_formated) - train_size
        train_CL, val_CL = torch.utils.data.random_split(CL_formated, [train_size, val_size])

        return train_CL, val_CL
    
    
    def load_data(self):
        test_CQ = self.load_chartQA()
        train_CL, val_CL = self.load_chartllma()
        return test_CQ, train_CL, val_CL


if __name__ == "__main__":
    load_long = vlm_data("long")
    test_CQ_long, train_CL_long, val_CL_long = load_long.load_data()

    load_middle = vlm_data("middle")
    test_CQ_middle, train_CL_middle, val_CL_middle = load_middle.load_data()

    load_short = vlm_data("short")
    test_CQ_short, train_CL_short, val_CL_short = load_short.load_data()

    load_other = vlm_data("other")
    test_CQ_other, train_CL_other, val_CL_other = load_other.load_data()

    # Make folder
    # Ref: https://khid.net/2019/12/python-check-exists-dir-make-dir/
    save_dir = 'data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # saving data in pickle
    # Ref: https://note.nkmk.me/python-pickle-usage/
    # For long system_message
    with open('data/test_CQ_long.pkl', 'wb') as f:
        pickle.dump(test_CQ_long, f)
    with open('data/train_CL_long.pkl', 'wb') as f:
        pickle.dump(train_CL_long, f)
    with open('data/val_CL_long.pkl', 'wb') as f:
        pickle.dump(val_CL_long, f)

    # For middle system_message
    with open('data/test_CQ_middle.pkl', 'wb') as f:
        pickle.dump(test_CQ_middle, f)
    with open('data/train_CL_middle.pkl', 'wb') as f:
        pickle.dump(train_CL_middle, f)
    with open('data/val_CL_middle.pkl', 'wb') as f:
        pickle.dump(val_CL_middle, f)

    # For short system_message
    with open('data/test_CQ_short.pkl', 'wb') as f:
        pickle.dump(test_CQ_short, f)
    with open('data/train_CL_short.pkl', 'wb') as f:
        pickle.dump(train_CL_short, f)
    with open('data/val_CL_short.pkl', 'wb') as f:
        pickle.dump(val_CL_short, f)

    # For other system_message
    with open('data/test_CQ_other.pkl', 'wb') as f:
        pickle.dump(test_CQ_other, f)
    with open('data/train_CL_other.pkl', 'wb') as f:
        pickle.dump(train_CL_other, f)
    with open('data/val_CL_other.pkl', 'wb') as f:
        pickle.dump(val_CL_other, f)
