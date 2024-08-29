import subprocess
import uuid
import yaml
import time
import argparse
import itertools
from typing import Any
from dataclasses import dataclass

@dataclass
class Model:
    name: str
    bucket_path: str
    max_input_tokens: int
    replicas: int

    def __str__(self) -> str:
        return self.name

@dataclass
class Test:
    # for use in grafana dashboard
    run_name: str
    model: Model
    # seconds
    duration: int
    # device
    device: str
    # "embedding", "sentence_similarity", or "rerank"
    task: str
    # "http" or "grpc"
    interface: str
    # number of simulated users for llm-load-test
    concurrency: int
    # batch sizes in llm-load-test
    batch_size: int

def run(cmd: str, print_stdout=False, print_stderr=False) -> dict[str, Any]:
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result = {
        "rc": 1 if p.returncode is None else int(p.returncode),
        "stdout": "" if p.stdout is None else str(p.stdout.read().decode("utf-8")),
        "stderr": "" if p.stderr is None else str(p.stderr.read().decode("utf-8")),
    }

    if print_stdout:
        print(result["stdout"])

    if print_stderr:
        print(result["stderr"])

    return result

def run_print(cmd: str) -> dict[str, Any]:
    return run(cmd, print_stdout=True, print_stderr=True)

def cleanup_jobs():
    run_print("oc delete jobs --all")

def deploy_model(model: Model, device: str="cpu"):
    sr = {}
    isvc = {}
    with open("template_servingruntime.yaml", "r") as f:
        sr = yaml.safe_load(f)
    with open("template_isvc.yaml", "r") as f:
        isvc = yaml.safe_load(f)

    if device == "gpu":
        sr["spec"]["containers"][0]["resources"]["requests"]["nvidia.com/gpu"] = 1
        sr["spec"]["containers"][0]["resources"]["limits"] = {}
        sr["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = 1

    with open("sr.yaml", "w+") as f:
        yaml.dump(sr, f)
    run_print(f"oc apply -f sr.yaml")

    isvc["spec"]["predictor"]["model"]["storageUri"] = model.bucket_path
    isvc["spec"]["predictor"]["minReplicas"] = model.replicas
    isvc["spec"]["predictor"]["maxReplicas"] = model.replicas

    with open("isvc.yaml", "w+") as f:
        yaml.dump(isvc, f)

    run_print(f"oc apply -f isvc.yaml")
    run_print(f"oc wait -f isvc.yaml --for=condition=Ready=True --timeout=5m")
    run_print("rm sr.yaml isvc.yaml")

def cleanup_models():
    run_print(f"oc delete isvc --all")
    run_print(f"oc delete servingruntime --all")

def start_test(test_settings: Test) -> str:

    test_uuid = str(uuid.uuid4())
    print(f"Started test {test_settings.run_name} ({test_uuid})")

    cleanup_jobs()
    cleanup_models()
    deploy_model(
        test_settings.model,
        device=test_settings.device,
    )

    metadata = {
        "platform": "metal",
        "sdnType": "OVNKubernetes",
        "name": "caikit-embeddings",
        "totalNodes": 1,
        "runName": test_settings.run_name,
        "device": test_settings.device,
        "concurrency": test_settings.concurrency,
        "batch_size": test_settings.batch_size,
    }

    with open("metadata.yml", "w+") as f:
        yaml.dump(metadata, f)

    llt_config = {}
    with open("template_config.yaml", "r") as f:
        llt_config = yaml.safe_load(f)

    llt_config["load_options"]["duration"] = test_settings.duration
    llt_config["load_options"]["concurrency"] = test_settings.concurrency
    llt_config["plugin_options"]["interface"] = test_settings.interface
    llt_config["plugin_options"]["task"] = test_settings.task
    llt_config["plugin_options"]["model_name"] = test_settings.model.name
    llt_config["plugin_options"]["model_max_input_tokens"] = test_settings.model.max_input_tokens
    llt_config["plugin_options"]["batch_size"] = test_settings.batch_size

    with open("llt_config.yaml", "w+") as f:
        yaml.dump(llt_config, f)
    run_print(f"oc cp llt_config.yaml model-store-pod:/pv/llt_config.yaml")

    run_print(f"mkdir {test_uuid}")
    start_time = int(time.time()) - 30
    run_print(f"kube-burner init --uuid={test_uuid} --user-metadata=metadata.yml --skip-tls-verify -c caikit-embedding.yml")
    end_time = int(time.time())
    run_print(f"mv kube-burner-{test_uuid}.log {test_uuid}/.")
    run_print(f"kube-burner index --uuid={test_uuid} -e metrics-endpoints.yml -s 5s --user-metadata metadata.yml --start={start_time} --end={end_time}")
    run_print(f"oc cp --retries=-1 model-store-pod:/pv/output/output.json {test_uuid}/output-{test_uuid}.json")
    run_print(f"python3 import.py -p {test_uuid}/output-{test_uuid}.json --uuid={test_uuid}")

    cleanup_jobs()
    run_print("rm metadata.yml")
    print(f"Finished test {test_settings.run_name} ({test_uuid})")
    return test_uuid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="path to config yaml file", required=True)
    args = parser.parse_args()

    config = {}
    with open(args.c, "r") as f:
        config = yaml.safe_load(f)

    make_run_name = lambda xs: "-".join(str(x) for x in xs)
    prefix = []
    if "prefix" in config and config["prefix"] != "":
        prefix.append(config["prefix"])
    models: list[Model] = [Model(model["name"], model["bucket_path"], model["max_input_tokens"], model["replicas"]) for model in config["model"]]
    test_combos = itertools.product(models, config["duration"], config["device"], config["task"], config["interface"], config["concurrency"], config["batch_size"])
    tests: list[Test] = [Test(make_run_name(prefix + list(combo)), combo[0], combo[1], combo[2], combo[3], combo[4], combo[5], combo[6]) for combo in test_combos]
    finished_tests: list[str] = []
    for test in tests:
        finished_uuid = start_test(test)
        finished_tests.append(finished_uuid)

    print("==========================")
    print("Tests executed:")
    for test_uuid in finished_tests:
        print(test_uuid)

if __name__ == "__main__":
    main()
