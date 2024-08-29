from typing import Any, Optional
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk, BulkIndexError
from functools import reduce 
import json
import yaml
import os
import sys
import datetime
from datetime import timezone
import uuid
import hashlib
import argparse

def get_from_path(d: dict[Any, Any], path: str, delim: str=".") -> Optional[Any]:
    try:
        segments = path.split(delim)
        return reduce(lambda td, seg: td[seg], segments, d)
    except KeyError:
        return None

def extract_payloads(obj: dict[str, Any], use_uuid: Optional[str]=None, metadata_path: Optional[str]=None) -> list[dict[str, Any]]:
    """
    Takes llm-load-test output and converts it to a
    kube-burner metric payload to be index in OpenSearch
    along side other kube-burner metrics
    """
    new_uuid = use_uuid if use_uuid is not None else str(uuid.uuid4())
    start_time = datetime.datetime.now(timezone.utc).isoformat()
    if "results" in obj:
        start_time = datetime.datetime.fromtimestamp(min(res["start_time"] for res in obj["results"]), tz=timezone.utc).isoformat()
    elif "start_time" in obj["summary"]:
        start_time = datetime.datetime.fromtimestamp(obj["summary"]["start_time"], tz=timezone.utc).isoformat()

    metadata = {}
    if metadata_path is not None:
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)

    payloads = []
    metrics = {
        "total_requests": "LLMLoadTestTotalRequests",
        "total_tokens": "LLMLoadTestTotalTokens",
        "throughput": "LLMLoadTestThroughput",
        "response_time.median": "LLMLoadTestResponseTimeMedian",
        "response_time.percentile_99": "LLMLoadTestResponseTime99th",
        "throughput_per_object": "LLMLoadTestThroughputPerObject",
        "throughput_tokens_per_document_per_second": "LLMLoadTestThroughputTokensPerDocumentPerSecond",
        "input_objects.median": "LLMLoadTestInputObjectsMedian",
        "input_objects.percentile_99": "LLMLoadTestInputObjects99th"
    }
    for metric_pair in metrics.items():
        payload = {
            "timestamp": start_time,
            "labels": {
                "task": obj["config"]["plugin_options"]["task"],
                "model_name": obj["config"]["plugin_options"]["model_name"],
                "interface": obj["config"]["plugin_options"]["interface"],
                "model_max_input_tokens": obj["config"]["plugin_options"]["model_max_input_tokens"],
                "concurrency": obj["config"]["load_options"]["duration"],
                "duration": obj["config"]["load_options"]["duration"],
                "replicas": obj["config"]["extra_metadata"]["replicas"],
                "host": obj["config"]["plugin_options"]["host"]
            },
            "metadata": metadata,
            "metricName": metric_pair[1],
            "value": get_from_path(obj["summary"], metric_pair[0]),
            "uuid": new_uuid,
            "query": "llm-load-test",
        }
        payloads.append(payload)
    return payloads

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="path to llm-load-test output", required=True)
    parser.add_argument("-m", help="optional path to a metadata file to attach to results (yaml)", default="metadata.yml")
    parser.add_argument("--uuid", help="optional uuid to use", default=None)
    args = parser.parse_args()


    file_to_import = args.p

    file_obj = {}
    with open(file_to_import, "r") as f:
        file_obj = json.load(f)

    if args.uuid is not None:
        payloads = extract_payloads(file_obj, use_uuid=args.uuid, metadata_path=args.m)
    else:
        payloads = extract_payloads(file_obj, metadata_path=args.m)

    host = os.environ.get("OPENSEARCH_HOST", "www.example.org")
    port = 443
    username = os.environ.get("OPENSEARCH_USER", "psap_admin")
    password = os.environ.get("OPENSEARCH_PASS", "password")
    auth = (username, password) # For testing only. Don't store credentials in code.
    index = os.environ.get("OPENSEARCH_INDEX")
    if index is None:
        sys.exit("Error: no OpenSearch index provided by environ OPENSEARCH_INDEX")

    # Create the client with SSL/TLS enabled, but hostname verification disabled.
    client = OpenSearch(
        hosts = [{'host': host, 'port': port}],
        http_compress = True,
        http_auth = auth,
        use_ssl = True,
        verify_certs = False,
        ssl_assert_hostname = False,
        ssl_show_warn = False
    )

    # import into opensearch index
    print(f"Uploading {len(payloads)} data points into index {index} in {host}")
    payloads_with_metadata = [{"_index": index, "_id": hashlib.sha256(str(p).encode()).hexdigest()} | p for p in payloads]
    try:
        bulk(client, payloads_with_metadata, max_retries=3)
    except BulkIndexError as e:
        print(e)
        sys.exit("Error: couldn't perform bulk insert")
    sys.exit(0)

if __name__ == "__main__":
    main()
