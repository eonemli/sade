import glob
import os


def clean(x):
    return x.strip().replace("_", "")


def get_image_files_list(dataset_name, dataset_dir, splits_dir):
    filenames = {}

    if dataset_name.lower() in ["abcd", "ibis-finetune"]:
        for split in ["train", "val", "test"]:
            with open(
                os.path.join(splits_dir, f"{ dataset_name.lower()}-{split}_keys.txt"), "r"
            ) as f:
                filenames[split] = [clean(x) for x in f.readlines()]

        train_file_list = [
            {"image": os.path.join(dataset_dir, f"{x}.nii.gz")} for x in filenames["train"]
        ]

        val_file_list = [
            {"image": os.path.join(dataset_dir, f"{x}.nii.gz")} for x in filenames["val"]
        ]

        test_file_list = [
            {"image": os.path.join(dataset_dir, f"{x}.nii.gz")} for x in filenames["test"]
        ]

    elif dataset_name.lower() == "ibis":
        train_file_list = val_file_list = None
        with open(os.path.join(splits_dir, "ibis_inlier_keys.txt"), "r") as f:
            filenames["inlier"] = [x.strip() for x in f.readlines()]

        test_file_list = [
            {"image": os.path.join(dataset_dir, f"IBIS_{x}.nii.gz")}
            for x in filenames["inlier"]
        ]
    elif dataset_name.lower() in ["ds-sa"]:
        train_file_list = val_file_list = None
        fname = os.path.join(splits_dir, f"{dataset_name}_outlier_keys.txt")
        assert os.path.exists(fname), "File for {dataset_name} does not exist at {fname}"
        with open(fname, "r") as f:
            filenames["outlier"] = [x.strip() for x in f.readlines()]

        test_file_list = [
            {"image": os.path.join(dataset_dir, f"IBIS{x}.nii.gz")}
            for x in filenames["outlier"]
        ]
    elif "lesion" in dataset_name:
        train_file_list = val_file_list = None
        test_file_list = [
            {"image": p, "label": p.replace(".nii.gz", "_label.nii.gz")}
            for p in glob.glob(f"{dataset_dir}/*/*.nii.gz")
            if "label" not in p  # very lazy, i know :)
        ]
    else:
        NotImplementedError

    # Sort the file list by image name
    if train_file_list is not None:
        train_file_list = sorted(train_file_list, key=lambda x: x["image"])

    if val_file_list is not None:
        val_file_list = sorted(val_file_list, key=lambda x: x["image"])

    if test_file_list is not None:
        test_file_list = sorted(test_file_list, key=lambda x: x["image"])

    return train_file_list, val_file_list, test_file_list
