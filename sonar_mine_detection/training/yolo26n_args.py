import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_yolo26n_augmentation(augmentation_key, config_path):
    augmentations = load_yaml(config_path)

    if augmentation_key not in augmentations:
        raise ValueError(f"Unknown augmentation setting: {augmentation_key}")

    values = augmentations[augmentation_key].copy()
    framework = values.pop("framework", None)

    if framework != "ultralytics":
        raise ValueError(f"Not a YOLO26n augmentation: {augmentation_key}")

    return values


def build_yolo26n_train_args(
    data_yaml,
    output_dir,
    run_name,
    image_size,
    epochs,
    batch_size,
    learning_rate,
    patience,
    seed,
    augmentation_key,
    augmentation_config,
    optimizer="SGD",
):
    train_args = {
        "data": str(data_yaml),
        "project": str(output_dir),
        "name": run_name,
        "imgsz": image_size,
        "epochs": epochs,
        "batch": batch_size,
        "lr0": learning_rate,
        "patience": patience,
        "seed": seed,
        "optimizer": optimizer,
        "exist_ok": True,
        "plots": False,
        "verbose": False,
        "workers": 4,
    }

    train_args.update(
        load_yolo26n_augmentation(
            augmentation_key=augmentation_key,
            config_path=augmentation_config,
        )
    )

    return train_args
