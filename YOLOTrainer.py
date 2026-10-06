from ultralytics import YOLO
# import albumentations as A
import shutil

if __name__ == "__main__":
    # Cleaning out folder from previous runs
    print("cleaning out folder")
    shutil.rmtree("runs/detect")

    print("Loading model")
    models = [YOLO('pretrained_models/yolo26n.pt'), YOLO('pretrained_models/yolo26s.pt'), YOLO('pretrained_models/yolo26m.pt'), YOLO('pretrained_models/yolo26l.pt'), YOLO('pretrained_models/yolo26x.pt')]
    names = ['Nano', 'Small', 'Medium', 'Large', 'Extra Large']
    # custom_transforms = [
    #     A.Blur(blur_limit=7, p=0.5),
    #     A.CLAHE(clip_limit=4.0, p=0.5),
    # ]

    # imgsz cannot be changed, epochs is iterations over data for training
    count = 0
    for model in models:
        print("Training " + names[count] + " Model")
        model.train(
            data="dataset",
            epochs=200,
            imgsz=640,
            optimizer="AdamW"
        )

        print(names[count] + " Model Statistics")
        metrics = model.val(data="dataset", plots=True)
        print(metrics.box.maps)

        print("Saving model")
        model = YOLO('runs/detect/train/weights/best.pt')
        model.save(names[count] + 'YOLOOxyFire.pt')
        count += 1