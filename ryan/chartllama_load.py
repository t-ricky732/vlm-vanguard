import json
import zipfile
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import PIL components directly
from PIL import Image, ImageDraw, UnidentifiedImageError

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger("ChartDataset")

class ChartDataset(Dataset):
    """
    Chart QA Dataset Loader. Loads data, handles images, cleans answers.
    Skips examples if image loading fails, logs the reason.
    """

    def __init__(
        self,
        data_dir="./chartllama_data",
        processor=None,
        image_size=384,
        max_answer_length=128,
        prompt_format="USER: {question}\nASSISTANT:",
        sample_limit=None,
        cache_images=True,
    ):
        self.processor = processor
        self.img_size = image_size
        self.max_ans_len = max_answer_length
        self.prompt_fmt = prompt_format
        self.limit = sample_limit
        self.cache_img = cache_images
        self.img_cache = {} if cache_images else None

        self.stats = {
            'files_found': 0, 'read_errors': 0, 'examples_seen': 0,
            'examples_ok': 0, 'bad_format': 0, 'bad_qa': 0,
            'img_load_failed_or_missing': 0,
            'cleaned_ans': 0, 'added_img_token': 0,
        }

        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        self.image_dir = self._setup_images(data_path)

        self.data_list = self._load_all_data(data_path)

        self._log_summary() 


    def _setup_images(self, data_path):
        """Handle zip extraction"""
        zip_f = data_path / 'ours.zip'
        extract_to = data_path / 'extracted_images'
        marker = extract_to / '.extraction_complete'
        img_base = data_path

        if zip_f.exists():
            if not marker.exists():
                extract_to.mkdir(exist_ok=True)
                try:
                    with zipfile.ZipFile(zip_f, 'r') as z:
                        z.extractall(extract_to)
                    marker.touch()
                    img_base = extract_to
                except Exception as e:
                    logger.error(f"Zip extract failed: {e}. Using base dir.")
                    img_base = data_path
            else:
                 img_base = extract_to


        return img_base

    def _load_all_data(self, data_path):
        """Loop through jsons and process examples"""
        all_json_files = list(data_path.glob('*.json'))
        self.stats['files_found'] = len(all_json_files)
        if not all_json_files:
            logger.warning(f"No json files in {data_path}!")
            return []

        good_data = []
        for fpath in tqdm(all_json_files, desc="Reading JSONs"):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = json.load(f)
            except Exception as e:
                logger.warning(f"Couldn't read {fpath.name}: {e}")
                self.stats['read_errors'] += 1
                continue

            if not isinstance(content, list):
                logger.warning(f"Skipping {fpath.name}, not a list.")
                self.stats['read_errors'] += 1
                continue

            fname_stem = fpath.stem
            ctype = fname_stem.split('_')[0] if '_' in fname_stem else 'unknown'

            for idx, data_item in enumerate(content):
                self.stats['examples_seen'] += 1
                processed = self._process_one_item(data_item, ctype, fpath.name, idx)
                if processed:
                    good_data.append(processed)
                    if self.limit and len(good_data) >= self.limit:
                        logger.info(f"Hit sample limit ({self.limit}). Stopping data loading.")
                        self.stats['examples_ok'] = len(good_data)
                        return good_data

        self.stats['examples_ok'] = len(good_data)
        return good_data

    def _process_one_item(self, data_item, ctype, fname, fidx):
        """Validate, clean, load image for one json entry. Skips if image fails."""

        # --- Step 1: Validate text data first ---
        if not isinstance(data_item, dict) or 'conversations' not in data_item:
            self.stats['bad_format'] += 1
            return None
        convos = data_item['conversations']
        if not isinstance(convos, list) or len(convos) < 2:
            self.stats['bad_format'] += 1
            return None
        q_part = convos[0]
        a_part = convos[1]
        if not isinstance(q_part, dict) or 'value' not in q_part or \
           not isinstance(a_part, dict) or 'value' not in a_part:
             self.stats['bad_format'] += 1
             return None
        q = q_part['value']
        raw_a = a_part['value']
        if not isinstance(q, str) or not q.strip() or \
           not isinstance(raw_a, str) or not raw_a.strip():
             self.stats['bad_qa'] += 1
             return None

        # --- Step 2: Clean text data ---
        if "<image>" not in q:
            q += " <image>"
            self.stats['added_img_token'] += 1
        lines = raw_a.strip().split('\n')
        a = lines[0].strip()
        if len(lines) > 1 and "Question:" in raw_a:
            self.stats['cleaned_ans'] += 1

        # --- Step 3: Prepare image path and ID ---
        img_path = data_item.get('image', '')
        if not isinstance(img_path, str): img_path = ''
        id_str = f"{fname}_{fidx}_{q[:20]}"
        ex_id = hashlib.md5(id_str.encode()).hexdigest()[:10]

        # --- Step 4: Load the image itself ---
        img_obj = self._load_img(img_path, ctype, ex_id)

        # --- Step 5: Check if image loading failed ---
        if img_obj is None:
            self.stats['img_load_failed_or_missing'] += 1
            return None 
        # --- Step 6: Return the valid example data ---
        return {
            'id': ex_id,
            'original_id': data_item.get('id', None),
            'image_path': img_path,
            'image': img_obj,
            'question': q,
            'answer': a,
            'chart_type': ctype,
            'source_file': fname,
            'source_index': fidx,
        }

    def _load_img(self, img_path_str: str, ctype: str, ex_id: str) -> Optional[Image.Image]:
        """
        Loads image, maybe from cache. Returns None on failure and logs reason.
        """
        if self.cache_img and self.img_cache is not None and ex_id in self.img_cache:
            return self.img_cache[ex_id]

        img = None
        found_path = None
        error_reason = "No image path provided"

        if img_path_str:
            error_reason = f"Image path not found or invalid: {img_path_str}"
            p = Path(img_path_str)
            paths_to_try = [ self.image_dir / p, self.image_dir / p.name, p ]
            for img_file_path in paths_to_try:
                if img_file_path.is_file():
                    found_path = img_file_path
                    try:
                        loaded = Image.open(img_file_path).convert('RGB')
                        resized = loaded.resize((self.img_size, self.img_size), Image.LANCZOS)
                        img = resized
                        error_reason = None # Clear error on success
                        break
                    except UnidentifiedImageError:
                        error_reason = f"Cannot identify image file (corrupt?): {found_path}"
                        break
                    except Exception as e:
                        error_reason = f"Error loading/resizing {found_path}: {e}"
                        break # Stop trying on error

        if img is None:
            logger.warning(f"Image load failed for example ID {ex_id}. Reason: {error_reason}")
            return None # Return None on failure

        if self.cache_img and self.img_cache is not None:
            self.img_cache[ex_id] = img

        return img

    def __len__(self):
        """How many items in the dataset"""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Gets one item's raw data for the collator"""
        try:
            data_item = self.data_list[idx]
            prompt = self.prompt_fmt.format(question=data_item['question'])
            return {
                "prompt": prompt,
                "answer": data_item['answer'],
                "image": data_item['image'],
                "id": data_item.get('id', f'unknown_{idx}')
            }
        except IndexError:
            logger.error(f"Index {idx} too high for dataset len {len(self.data_list)}")
            return None
        except Exception as e:
            logger.error(f"Problem getting item {idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _log_summary(self):
        """Logs a summary of the data loading statistics."""
        logger.info("--- Loading Summary ---")
        logger.info(f"JSON files found: {self.stats['files_found']}")
        logger.info(f"JSON read errors: {self.stats['read_errors']}")
        logger.info(f"Raw examples processed: {self.stats['examples_seen']}")
        logger.info(f"Valid examples loaded: {self.stats['examples_ok']}")
        logger.info(f"Skipped (invalid format): {self.stats['bad_format']}")
        logger.info(f"Skipped (missing Q/A): {self.stats['bad_qa']}")
        logger.info(f"Skipped (image missing/error): {self.stats['img_load_failed_or_missing']}")
        logger.info(f"Multi-part answers cleaned: {self.stats['cleaned_ans']}")
        logger.info(f"<image> tokens added: {self.stats['added_img_token']}")
        logger.info("------------------------")