from rfdetr import RFDETRBase

if __name__ == "__main__":
    model = RFDETRBase(pretrain_weights="pretrained_models/rf-detr-base-coco.pth")
    print("running")
    model.train(
        dataset_dir="dataset",
        epochs=50,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="RFDETResults"
    )