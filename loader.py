import os
import gc
import torch
import numpy as np
import logging
from utils import Tools
from concurrent.futures import ThreadPoolExecutor

class Loader():
    def __init__(self, size):
        self.size = size

    def _read(self, name, classes, train_dataset, val_dataset, test_dataset, size):
        """Read and form the data from dataset list."""
        def get_data_dict(dataset_list):
            x, y = Tools.readFromList(dataset_list)
            x_tensor = torch.tensor(x, dtype=torch.float).view(-1, size[0], size[1], size[2])
            y_tensor = torch.tensor(y, dtype=torch.long).view(-1, size[1], size[2])
            return {'x': x_tensor, 'y': y_tensor}

        data_train = {
            'name': name, 
            'classes': classes,
            'train': get_data_dict(train_dataset),
            'val': get_data_dict(val_dataset)
        }
        data_test = {
            'name': name, 
            'classes': classes,
            'test': get_data_dict(test_dataset)
        }
        gc.collect()
        return data_train, data_test

    def _save_compressed(self, data_dict, path):
        """Save a dictionary of tensors to a compressed npz file."""
        save_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):
                for sub_k, sub_v in value.items():
                    if isinstance(sub_v, dict):
                        for tri_k, tri_v in sub_v.items():
                            combined_key = f"{key}|{sub_k}|{tri_k}"
                            save_dict[combined_key] = tri_v.numpy() if torch.is_tensor(tri_v) else tri_v
                    else:
                        combined_key = f"{key}|{sub_k}"
                        save_dict[combined_key] = sub_v.numpy() if torch.is_tensor(sub_v) else sub_v
            else:
                save_dict[key] = value.numpy() if torch.is_tensor(value) else value
        
        np.savez_compressed(path, **save_dict)
        del save_dict
        gc.collect()

    @staticmethod
    def load_compressed(path):
        """Load a dictionary of tensors from a compressed npz file."""
        data = np.load(path, allow_pickle=True)
        res = {}
        for key in data.files:
            val = data[key]
            parts = key.split('|')
            curr = res
            
            for i, part in enumerate(parts):
                k = int(part) if part.isdigit() else part
                
                if i == len(parts) - 1:
                    if isinstance(val, np.ndarray) and val.ndim > 0:
                        if val.dtype.kind in {'U', 'S', 'O'}:
                            curr[k] = val.tolist()
                        else:
                            curr[k] = torch.from_numpy(val).clone()
                    else:
                        curr[k] = val.item() if hasattr(val, 'item') else val
                else:
                    if k not in curr or not isinstance(curr[k], dict):
                        curr[k] = {}
                    curr = curr[k]
        return res

    def make(self, path, seed, fourTasks):
        """Create all dataset and save to the disk."""
        order = Tools.order(seed, fourTasks)
        
        if not os.path.exists("./zip_data"):
            os.mkdir("./zip_data")

        task_config = {
            0: ("Lumbar", 6, 'lumbar_train', 'lumbar_val', 'lumbar_test'),
            1: ("RV", 2, 'RV_train', 'RV_val', 'RV_test'),
            2: ("Tumor", 2, 'tumor_train', 'tumor_val', 'tumor_test'),
            3: ("Lumbar_MR", 8, 'lumbar_MR_train', 'lumbar_MR_val', 'lumbar_MR_test')
        }

        with ThreadPoolExecutor() as executor:
            for i, d in enumerate(order):
                name, classes, tr_p, val_p, te_p = task_config[d]
                dt_train, dt_test = self._read(name, classes, path[tr_p], path[val_p], path[te_p], self.size)
                
                # Save individual files for both Train and Test
                executor.submit(self._save_compressed, dt_train, f"./zip_data/{name.lower()}_train.npz")
                executor.submit(self._save_compressed, dt_test, f"./zip_data/{name.lower()}_test.npz")

                gc.collect()


    def load_all(self, seed, fourTasks):
        """Load all dataset from disk at once."""
        order = Tools.order(seed, fourTasks)
        name_map = {0: "lumbar", 1: "rv", 2: "tumor", 3: "lumbar_mr"}
        task_classes = {0: 6, 1: 2, 2: 2, 3: 8}
        warning = False
        
        data = {}
        
        def load_pair(idx, task_id):
            t_name = name_map[task_id]
            train = self.load_compressed(f"./zip_data/{t_name}_train.npz")
            test = self.load_compressed(f"./zip_data/{t_name}_test.npz")
            # Merge them so data_all[idx] contains both 'train', 'val', and 'test'
            train.update(test)
            return idx, train

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(load_pair, i, d) for i, d in enumerate(order)]
            for f in futures:
                idx, result = f.result()
                data[idx] = result

        # Dynamic metadata reconstruction
        train_order = [name_map[d].upper() for d in order]
        INFO = [(i, task_classes[d]) for i, d in enumerate(order)]
        n_classes = sum(task_classes[d] for d in order)
        
        data.update({'classes': n_classes, 'order': train_order, 'info': INFO})
        return data, INFO

    def load_metadata(self, seed, fourTasks):
        """Generates metadata dynamically for both lazy modes."""
        order = Tools.order(seed, fourTasks)
        
        # Source of truth for task mapping
        task_map = {
            0: {"name": "Lumbar", "classes": 6},
            1: {"name": "RV", "classes": 2},
            2: {"name": "Tumor", "classes": 2},
            3: {"name": "Lumbar_MR", "classes": 8}
        }
        
        train_order = [task_map[d]["name"] for d in order]
        info = [(i, task_map[d]["classes"]) for i, d in enumerate(order)]
        
        metadata = {
            'order': train_order,
            'info': info,
            'classes': sum(task_map[d]["classes"] for d in order)
        }
        return metadata, info

    def load_task(self, task_name, split='train'):
        """Loads a single task's data from disk on demand."""
        path = f"./zip_data/{task_name.lower()}_{split.lower()}.npz"
        return self.load_compressed(path)

    def check_all_data_exists(self, path, fourTasks):
        """
        Checks if all required .npz files exist in ./zip_data/.
        Returns True if all clear, or False if items are missing.
        """
        order = Tools.order(0, fourTasks)

        # Map IDs to filenames
        task_map = {
            0: ("Lumbar", 'lumbar_train', 'lumbar_val', 'lumbar_test'),
            1: ("RV", 'RV_train', 'RV_val', 'RV_test'),
            2: ("Tumor", 'tumor_train', 'tumor_val', 'tumor_test'),
            3: ("Lumbar_MR", 'lumbar_MR_train', 'lumbar_MR_val', 'lumbar_MR_test')
        }
        
        splits = ["train", "test"]
        
        for d in order:
            task_name, tr_p, val_p, te_p = task_map[d]
            for split in splits:
                file_path = f"./zip_data/{task_name.lower()}_{split.lower()}.npz"
                if not os.path.exists(file_path):
                    return False
                tr_n = Tools.readlines(path[tr_p])
                val_n = Tools.readlines(path[val_p])
                te_n = Tools.readlines(path[te_p])
                data_train = self.load_task(task_name)
                data_test = self.load_task(task_name, 'test')
                tr_x = data_train['train']['x'].shape[0]
                val_x = data_train['val']['x'].shape[0]
                te_x = data_test['test']['x'].shape[0]

                if tr_n != tr_x or val_n != val_x or te_n != te_x:
                    return False

        return True