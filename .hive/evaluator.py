import logging
import subprocess
import json
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    logger.info("Compiling Dynamo...")

    ROOT = os.getenv("DYNAMO_HOME","/app/repo")
    VENV = ".venv"
    DATA = f"{ROOT}/mooncake_trace.jsonl"
    MOCKER_OUTPUT = f"{ROOT}/replay-report.json"
    NUM_WORKERS = 8
    
    # Re-compile Dynamo
    COMPILE_CMD = (
        f"cd {ROOT} && "
        f". {VENV}/bin/activate && "
        f"cd {ROOT}/lib/bindings/python && "
        "maturin develop --release --uv --strip && "
        f"cd {ROOT} && "
        "uv pip install -e ."
    )
    subprocess.run(COMPILE_CMD, shell=True, check=True, capture_output=True)

    # Run the Mocker on data set
    logger.info("Run Mocker Trace Replay")

    MOCKER_CMD = (
        f"cd {ROOT} && "
        f"{VENV}/bin/python -m dynamo.replay {DATA} "
        f"--num-workers {NUM_WORKERS} "
        "--replay-mode offline "
        "--router-mode kv_router "
        "--trace-block-size 512 "
        f"--report-json {MOCKER_OUTPUT} "
        """--extra-engine-args '{"block_size":64}' """
    )

    subprocess.run(MOCKER_CMD, shell=True, check=True, capture_output=True)

    # Parse the mocker output
    with open(MOCKER_OUTPUT) as f:
        mocker_output = json.load(f)

    # Form Hive output. Must contain fitness
    hive_output = mocker_output
    mean_ttft_ms = mocker_output["mean_ttft_ms"]
    mean_itl_ms = mocker_output["mean_itl_ms"]

    # Multi-objective
    hive_output["fitness"] = {
        "-TTFT": -mean_ttft_ms,
        "-ITL": -mean_itl_ms
    }
    hive_output["signature"] = [float(mean_ttft_ms), float(mean_itl_ms)]

    print(json.dumps({"output": hive_output, "metainfo": "Success"}))

    