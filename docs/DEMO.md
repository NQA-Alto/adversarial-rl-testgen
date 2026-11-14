# Demo guide

The Gradio application combines saved examples, result summaries and local inference. It runs on the original Windows environment and requires external dataset, model and run files. See [setup and training](APPS_TRAINING.md) before launching it.

For a quick look at scoring without those assets, run `python examples/verify_test.py` from the repository root.

## Start and stop

On the configured machine, run `start_ui.cmd`, then open <http://127.0.0.1:7860/?__theme=light>. TensorBoard is served at <http://127.0.0.1:6006>. Run `stop_ui.cmd` to stop the services started by the launcher.

The launchers currently contain absolute paths to the original machine. They are not a portable installer.

## Application tabs

| Tab | Purpose |
|---|---|
| Demo | Inspect a problem, faulty code and saved tests; optionally generate four new tests |
| Results | Compare recorded Validation and held-out Test results |
| Data | Inspect dataset sources, split sizes and supported input domains |
| Experiments | Run dataset checks, a GRPO trial or adapter evaluation; inspect logs |

The interface uses English labels. Saved examples use Validation with generation seed 42. New inference uses the GPU and can differ from those examples because generation is sampled.

## Reading a generated test

Each row shows the input, generated expected output and the faulty function's output. When expected is wrong, the reference output is shown as well. A correct test that misses the bug is separate from an invalid test. The raw JSON view preserves the original output.

For the middle-character example, `bAA` should return `A`, while `bb` should return `bb`. The allowed input domain is displayed alongside the code.

## Interactive training

The GRPO control starts from SFT840 with a new optimizer and uses fixed Train code. It accepts 1-96 training steps. Interactive runs save new adapters, logs and results under `evidence/`; they do not replace the checkpoint used in the recorded final evaluation.

Stopping a job preserves files already written. The experiment logs include the run directory so saved outputs can be inspected later.
