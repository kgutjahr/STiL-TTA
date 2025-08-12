from typing import Dict
import torch

class AugmentSummarizer():
    def __init__(self):
        self.multi_aug_rate = []
        self.image_aug_rate = []
        self.table_aug_rate = []
        
        self.multi_aug_epoch_rate = []
        self.image_aug_epoch_rate = []
        self.table_aug_epoch_rate = []
        
    def register_rate(self, modality: str, A: torch.Tensor, B: torch.Tensor) -> None:
        assert modality in ("image", "tabular", "multimodal")
        with torch.no_grad():
            rate = self.__get_augment_rate(A=A, B=B)

            # multi
            if modality == "multimodal":
                self.multi_aug_epoch_rate.append(rate)
            # image
            if modality == "image":
                self.image_aug_epoch_rate.append(rate)
            # tabular
            if modality == "tabular":
                self.table_aug_epoch_rate.append(rate)
            
    def summarize(self) -> Dict[str, float]:
        def safe_mean(values):
            return sum(values) / len(values) if values else 0.0

        with torch.no_grad():
            # get current epoch averages
            multi_epoch_avg = safe_mean(self.multi_aug_epoch_rate)
            image_epoch_avg = safe_mean(self.image_aug_epoch_rate)
            table_epoch_avg = safe_mean(self.table_aug_epoch_rate)

            # add to previous averages
            self.multi_aug_rate.append(multi_epoch_avg)
            self.image_aug_rate.append(image_epoch_avg)
            self.table_aug_rate.append(table_epoch_avg)

            # calculate current averages over the whole training
            multi_rate = safe_mean(self.multi_aug_rate)
            image_rate = safe_mean(self.image_aug_rate)
            table_rate = safe_mean(self.table_aug_rate)

        return {"multi_rate": multi_rate, "image_rate": image_rate, "table_rate": table_rate}
    
    def reset(self) -> None:
        # clear epoch lists without replacing them (keeps references intact)
        with torch.no_grad():
            self.multi_aug_epoch_rate.clear()
            self.image_aug_epoch_rate.clear()
            self.table_aug_epoch_rate.clear()
    
    @staticmethod
    def __get_augment_rate(A: torch.Tensor, B: torch.Tensor):
        assert len(A) == len(B)
        diff = (A != B)
        rows_with_diff = diff.any(dim=1)  # True if any element in row is True
        return (rows_with_diff.sum() / A.size()[0]).item()