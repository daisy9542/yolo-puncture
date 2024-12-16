models = [
    "ela-t-v3",
    "eca-v4",
    "caa-v10",
    "ema-v13",
    "mca-v14",
    "cbam-v16",
    "srm-v18",
    "se-v20",
    "ca-v22",
    "sa-v24",
    "diy-v26",
    "diy-v27",
    "diy-v28",
    "diy-acam-v29",
    "diy-dcim-v30",
    "diy-lsam-v31",
    "diy-mstm-v32",
    "diy-cpem-v33",
    "diy-fsfm-v34",
    "diy-capm-v35",
    "diy-lham-v36",
    "diy-eem-v37",
    "diy-nafm-v38",
    "diy-isrm-v39",
    "diy-ssrm-v40",
    "diy-csrm-v41",
    "diy-gnsrm-v42",
    "diy-msrm-v43",
    "diy-nlsrm-v44",
    "diy-ressrm-v45",
    "diy-fssrm-v46",
    "diy-gcsrm-v47",
    "diy-dcsrm-v48"
]
cuda_device = 2
version = "yolo11"
# Carparts-seg, Crack-seg, Package-seg
CARPARTS = "Carparts-seg"
CRACK = "Crack-seg"
PACKAGE = "Package-seg"
dataset = CRACK
project = "model-improve"
epochs = 100
imgsz = 640
batch = 64
split = "val"

script_path = "../x.bash"
output_path = f"{dataset}-result.txt"
if dataset == CARPARTS:
    dataset_name = "carparts"
elif dataset == CRACK:
    dataset_name = "crack"
elif dataset == PACKAGE:
    dataset_name = "package"


def gen_train():
    with open(script_path, "w") as f:
        for model in models:
            command = (
                f"CUDA_VISIBLE_DEVICES={cuda_device} "
                f"yolo segment train "
                f"data=/mnt/zwp/{dataset}/data.yaml "
                f"model={version}-{model}.yaml "
                f"epochs={epochs} "
                f"imgsz={imgsz} "
                f"batch={batch} "
                f"project={project} "
                f"name={version}-{dataset_name}-{model} "
                f"pretrained=False "
                f"exist_ok=True "
                f"seed=42 "
                f"device={cuda_device}"
                f"\n"
            )
            f.write(command)


def gen_test():
    with open(script_path, "w") as f:
        for model in models:
            f.write(f"echo \"model {version}-{dataset_name}-{model} "
                    f"-------------------------------------------------\\n\" >> {output_path}\n")
            for weights in ["best", "last"]:
                command = (
                    f"CUDA_VISIBLE_DEVICES={cuda_device} "
                    f"yolo segment val "
                    f"data=/mnt/zwp/{dataset}/data.yaml "
                    f"model={project}/{version}-{dataset_name}-{model}/weights/{weights}.pt "
                    f"split={split} "
                    f"imgsz={imgsz} "
                    f"device={cuda_device} "
                    f">> {output_path}\n"
                )
                f.write(command)
        
        f.write(f"echo \"file written in file {output_path}\"\n")


if __name__ == "__main__":
    # gen_train()
    gen_test()
