import argparse
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import build_adversarial_pool as sampler
from check_local_model import Tee
from post_warmup_data import prepare_context
from round1_data import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    args.config = Path(__file__).resolve().parents[1] / "configs/post-warmup-round1.json"
    config = read(args.config)
    args.diagnostic_run = args.assets / "evidence" / config["train_probe"]
    args.samples_per_problem = config["samples_per_problem"]
    for key, value in {"HF_HOME": str(args.assets / "cache/huggingface"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
        "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp")}.items():
        os.environ[key] = value
    folder = args.assets / "evidence" / (("post-warmup-pool-preflight-" if args.preflight else "post-warmup-adversary-") + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    folder.mkdir(parents=True, exist_ok=False)
    sources = folder / "source"
    sources.mkdir()
    for name in (Path(__file__).name, "post_warmup_data.py", "build_adversarial_pool.py", "restricted_python.py", "dataset_tasks.py", "evaluate_model.py", "round1_data.py", "check_local_model.py"):
        shutil.copy2(Path(__file__).with_name(name), sources / name)
    shutil.copy2(args.config, sources / args.config.name)
    result = {"started_at": datetime.now().astimezone().isoformat(), "status": "started", "training": False, "rl_training": False,
              "test_set_evaluation": False, "preflight_only": args.preflight, "curriculum_round": "post_warmup_round1",
              "source_hashes": {path.name: sampler.digest(path) for path in sources.iterdir()},
              "context_adapter": "Same frozen sampler, restricted interpreter and gate; validated 48-case SFT Train context injected in memory"}
    previous = sampler.prepare_context
    out, err = sys.stdout, sys.stderr
    code = 0
    with (folder / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(out, log), Tee(err, log)
        try:
            print("Evidence:", folder, flush=True)
            sampler.prepare_context = prepare_context
            sampler.run(args, folder, result)
            print("STATUS:", result["status"], flush=True)
        except KeyboardInterrupt:
            result["status"], code = "interrupted", 130
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error))
            traceback.print_exc()
            code = 1
        finally:
            sampler.prepare_context = previous
            result["finished_at"] = datetime.now().astimezone().isoformat()
            sampler.write_json(folder / "result.json", result)
            print("Saved evidence:", folder, flush=True)
            sys.stdout, sys.stderr = out, err
    return code


if __name__ == "__main__":
    raise SystemExit(main())
