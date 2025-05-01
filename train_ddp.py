from data_loader import build_loaders
from datetime import datetime
from pathlib import Path
import argparse
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from sklearn.utils.class_weight import compute_class_weight
import random
import os
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class LitTimm(pl.LightningModule):
    def __init__(self, model_name, num_classes: int, lr: float, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )
        self.class_weights = class_weights
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1   = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        imgs, labels = batch
        logits = self(imgs)
        # loss   = F.cross_entropy(logits, labels)
        loss   = F.cross_entropy(logits, labels, weight=self.class_weights.to(self.device))
        self.train_f1(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_f1",   self.train_f1, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        imgs, labels = batch
        logits = self(imgs)
        # loss = F.cross_entropy(logits, labels)
        loss   = F.cross_entropy(logits, labels, weight=self.class_weights.to(self.device))
        self.val_f1(logits, labels)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_f1",   self.val_f1, prog_bar=True,
                on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-8
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"},
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus",        type=int, default=1)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--img_size",    type=int, default=224)
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--seed",        type=int, default=777)
    args = parser.parse_args()

    # 1) 재현성
    pl.seed_everything(args.seed, workers=True)
    seed_everything(args.seed)

    # 2) Data 로드
    train_loader, val_loader, le = build_loaders(
        data_dir="./data/train",
        test_size=0.2,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
        mean = (0.38771973, 0.39787053, 0.40713646), # 전역분산
        std = (0.2130759, 0.21581389, 0.22090413),
    )

    # 2-2) Class Weighs 계산(필요시)
    train_labels = train_loader.dataset.labels

    balanced = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )  # shape (num_classes,)

    sqrt_balanced = np.sqrt(balanced)
    class_weights = torch.tensor(sqrt_balanced, dtype=torch.float)


    # 3) 모델
    model = LitTimm(
        model_name="vit_xsmall_patch16_clip_224",
        # model_name='vit_so150m2_patch16_reg1_gap_448.sbb_e200_in12k_ft_in1k',
        num_classes=len(le.classes_),
        lr=args.lr,
        class_weights=class_weights
        )

    # 4) 콜백
    timestamp = datetime.now().strftime("%m%d_%H%M")  # 예: 0419_1532
    save_dir = Path("model_weights")
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = ModelCheckpoint(
        dirpath=save_dir,
        filename=f"{timestamp}_{{epoch:02d}}_{{val_f1:.4f}}",
        monitor="val_f1", # val_loss
        mode="max",
        save_top_k=1
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Window에서는 ddp 방식이 불가함
    if os.name == "nt":  # Windows
        # 1GPU → 'auto', 2+GPU → ddp_spawn
        strategy = 'auto' if args.gpus == 1 else "ddp_spawn"
    else:  # Linux/macOS
        strategy = 'auto' if args.gpus == 1 else "ddp"

    # 5) Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        accumulate_grad_batches=1,  # 필요할때 사용 (최종 batch size는 512에 맞추기)
        # 최종 batch size = num_gpus * batch_size * accumulate_grad_batches
        strategy=strategy,
        max_epochs=args.epochs,
        precision='16-mixed',    
        callbacks=[ckpt, lr_monitor],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()

    # 예시
    # python train_ddp.py --gpus 1 --epochs 40 --img_size 224 --batch_size 512 --seed 777 --num_workers 4 --prefetch_factor 1
    # python train_ddp.py --gpus 2 --epochs 20 --img_size 448 --batch_size 64 --num_workers 10
    # CUDA_VISIBLE_DEVICES=0 python train_ddp.py --gpus 1 --epochs 30 --img_size 256 --batch_size 128
    # CUDA_VISIBLE_DEVICES=0 python train_ddp.py --gpus 1 --epochs 40 --img_size 224 --num_workers 32 --batch_size 512
