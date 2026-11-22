---
name: kaggle-workflow
description: Use Kaggle CLI and Kaggle MCP for notebook, dataset, and training workflows. Apply when operating Kaggle resources or choosing between CLI, MCP, and local execution.
---

# Kaggle Workflow

Use Kaggle's official CLI and available Kaggle MCP tools to operate notebooks and datasets. Keep source data and full run artifacts in Kaggle when the project requires it. Do not use a browser for routine API, quota, notebook, or dataset operations.

## Choose the interface

- Prefer a Kaggle MCP tool when it provides the exact required operation and its arguments preserve the needed metadata. Discover available tools from the current tool inventory; do not assume the MCP server has a quota endpoint.
- Use the official CLI for quota checks, notebook pushes that require metadata or accelerator settings, kernel status, and logs. In this environment run it as `uvx --from kaggle --with six kaggle ...` so it reads the configured `~/.kaggle/access_token`.
- The available Kaggle MCP `kernel_push` accepts notebook text and basic visibility/language fields, but not a metadata file or accelerator choice. Use CLI `kernels push -p <directory>` for configured notebook runs, especially GPU runs.
- MCP and CLI may use separate authentication contexts. A working Kaggle MCP connection does not prove local CLI authentication, and vice versa. Verify the interface being used with a harmless read operation; never print or inspect credential contents. Do not ask for browser login if CLI authentication already works.

## GPU-first execution

Before starting or falling back from any GPU-compatible mainline task:

1. Query the live official quota with `uvx --from kaggle --with six kaggle quota`. Record the timestamp, GPU used/remaining/total and refresh time. This is the preferred quota check; do not use a browser for it.
2. Check known active or queued Kaggle GPU kernels with `uvx --from kaggle --with six kaggle kernels status <owner>/<kernel-slug>` and the available MCP session/status tools. Do not claim there are no reservations if the available interfaces cannot establish that.
3. If quota remains, run meaningful GPU-compatible mainline work on Kaggle GPU. Configure the notebook metadata and explicitly request an accelerator in the CLI where supported, then verify the actual run status and CUDA/device evidence in Kaggle logs. A metadata label alone does not prove the GPU was allocated.
4. Continue with valid GPU-compatible work until the official rolling quota reaches zero. Only then may unfinished GPU-compatible work fall back to Kaggle CPU. CPU fallback is also allowed when Kaggle explicitly refuses GPU allocation or the run proves CUDA unavailable; record that evidence. Unknown quota or an inconvenient runtime estimate is not grounds for CPU fallback.
5. Keep CPU-native tasks on Kaggle CPU: examples include Rust archive validation, label matrix generation, and traditional rules or logistic-regression baselines. Do not repeat an experiment solely to consume quota or exceed the official limit.

Record quota before/after, requested and actual device, GPU hours, and any fallback reason with the task's results. Where a roadmap exists, update its evidence after each major task, then commit and push the completed change as the user requested.

## Safe Kaggle operations

- Prefer MCP read/list/metadata operations when they return the needed facts. Use Kaggle CLI `kernels status` for a notebook's execution status and `kernels logs` for run logs.
- `kernels output` downloads notebook output files. Avoid it when the workflow requires artifacts to remain in Kaggle; use bounded logs, Kaggle-side summaries, or a read-only index notebook instead.
- Before pushing, inspect the notebook's `kernel-metadata.json` for privacy, dataset sources, Internet, and accelerator settings. Push from that directory with `uvx --from kaggle --with six kaggle kernels push -p <directory>`.
- Credentials are secret material. Never read, copy, log, or include `~/.kaggle/access_token` or any API key in notebook code, tool output, commits, or documentation.
