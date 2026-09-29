from ultralytics import YOLO
import albumentations as A

if __name__ == "__main__":
    # Model 26, x indicates extra large sixe ~10-15 ms per image, cle sets to classify
    print("Loading model")
    model = YOLO('yolo26x.pt')
    # Add other augmentations here, these set the bounding box out of bounds so don't use them
    # custom_transforms = [
    #     A.Blur(blur_limit=7, p=0.5),
    #     A.CLAHE(clip_limit=4.0, p=0.5),
    # ]
    print("Training Model")
    # Do not change image size, epochs is passes over the data
    model.train(
        data="dataset",
        epochs=5,
        # augmentations=custom_transforms,
        imgsz=640,
    )
    print("Model Statistics")
    # Print statistics
    metrics = model.val(data="dataset")
    print(metrics.box.map)
    print("Saving model")

    # fix file path to run yourself
    model = YOLO('/scratch/user/tylersprin/runs/detect/train-5/weights/best.pt')
    model.save('/OxyFire.pt')