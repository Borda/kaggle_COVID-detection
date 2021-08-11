import os

import fire
import flash
import torch
from flash.image import ObjectDetectionData, ObjectDetector


def main(
    path_dataset: str,
    image_size: int = 512,
    head: str = "efficientdet",
    backbone: str = "tf_d0",
    learn_rate: float = 1.5e-5,
    batch_size: int = 12,
    num_epochs: int = 30
) -> None:
    # 1. Create the DataModule
    dm = ObjectDetectionData.from_coco(
        train_folder=os.path.join(path_dataset, "images", 'train'),
        train_ann_file=os.path.join(path_dataset, "covid_train.json"),
        val_split=0.1,
        batch_size=batch_size,
        image_size=image_size,
    )

    # 2. Build the task
    model = ObjectDetector(
        head=head,
        backbone=backbone,
        optimizer=torch.optim.AdamW,
        learning_rate=learn_rate,
        num_classes=dm.num_classes,
        image_size=image_size,
    )

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(
        max_epochs=num_epochs,
        gpus=torch.cuda.device_count(),
        precision=16,
        accumulate_grad_batches=24,
        val_check_interval=0.5,
    )
    trainer.finetune(model, datamodule=dm, strategy="freeze_unfreeze")

    # 3. Save the model!
    trainer.save_checkpoint("object_detection_model.pt")


if __name__ == '__main__':
    fire.Fire(main)
