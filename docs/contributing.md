# How to contribute

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Development environment

This repository uses uv projects for isolated dependency sets:

```bash
uv sync --only-dev                    # repository formatting and pre-commit tooling
uv sync --package open-rl-client      # examples and client scripts
uv sync --project src/server --extra cpu
uv sync --project src/server --extra vllm
```

The root workspace lock covers repository tooling, examples, and developer tools. The server keeps its own lockfile because its CPU, GPU, and vLLM extras intentionally select different Torch indexes.

Run formatting before sending a pull request:

```bash
make fmt
```

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.
