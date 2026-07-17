# OpenRL docs — reading order

| Read this | When |
| --- | --- |
| [quickstart.md](quickstart.md) | You want a running system today (LoRA first, then FFT + vLLM). |
| [configuration.md](configuration.md) | You need a knob: every env var, with the long-context and time-slicing mechanics explained. |
| [journey.md](journey.md) | You want the condensed story: traces→SFT→GRPO, the memory war, the garble taxonomy — what broke and why the code looks like this. |
| [fft/fft.md](fft/fft.md) | Deep architecture of the FFT subsystem: queues, worker lifecycle, time-slicer, cluster topology. |
| [fft/single-h100-long-context.md](fft/single-h100-long-context.md) | Measured memory ceilings and step timings behind the long-context knobs. |
| [fft/rl-with-gemma4.md](fft/rl-with-gemma4.md) | Gemma 4 renderer/formatting background (the LAB recipe ships the maintained renderer). |
| [architecture.md](architecture.md) | General system overview (gateway, store, trainer, checkpoints). |
| [setup/local-setup.md](setup/local-setup.md), [setup/gke-setup.md](setup/gke-setup.md), [setup/gke-fft-timeslice.md](setup/gke-fft-timeslice.md) | Environment provisioning: single VM, basic GKE, and the FFT time-slice cluster. |
| [tinker-client-compatibility.md](tinker-client-compatibility.md) | Which tinker SDK surface the gateway implements (stale; regenerate before relying on it). |
| [../examples/harvey_labs/README.md](../examples/harvey_labs/README.md) | The Harvey LAB recipe itself: setup, LoRA/FFT runs, reward and renderer details. |
| [blog/from-mac-to-gke.md](blog/from-mac-to-gke.md) | Narrative blog post about the project. |
