# Acknowledgements

HIVE is built on the following open-source work:

- [verl](https://github.com/volcengine/verl): distributed RL training, FSDP workers, rollout integration, reward utilities, and checkpoint tooling. Upstream copyright and Apache 2.0 notices are retained in vendored source files.
- [GRESO](https://github.com/Infini-AI-Lab/GRESO): history-based prompt selection and the experimental preprocessing protocol. The Gaokao input is referenced at a fixed upstream commit rather than redistributed here.
- [vLLM](https://github.com/vllm-project/vllm): efficient rollout generation.

The boxed-answer preprocessing helper and FSDP merger are adapted from the local archived verl/GRESO source. HIVE-specific changes align selection and batch handling with the accompanying manuscript.

Model and dataset providers are identified in [the data guide](docs/data.md) and [model profiles](configs/models.json). Their licenses and access conditions are independent of this repository's Apache 2.0 code license. Paper figures in `assets/` are supplied by the HIVE authors.
