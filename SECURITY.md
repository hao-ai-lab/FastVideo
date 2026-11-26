# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities privately. Do not open a public issue, and do
not post details in Discussions or Slack.

Email the maintainers at wlsaidhi@gmail.com with the subject
`FastVideo security report`.

Include as much of this as you can:

- The affected component, with the file, CLI command or server endpoint.
- Steps to reproduce, using the smallest example you can give.
- The FastVideo version or commit you tested. For a source checkout, run
  `git rev-parse HEAD` for the commit. Include `python collect_env.py` output for
  the Python, PyTorch and CUDA versions.
- Any proof of concept, logs or screenshots.

## What we do with a report

We will acknowledge the report and tell you whether we treat it as a security
issue. We cannot promise a fix date. We will keep you updated, coordinate the
disclosure with you, and credit you in the release notes unless you prefer
otherwise. Please give us time to release a fix before you write about the
issue.

## Supported versions

We apply security fixes on `main`, and they ship in the next release.

## Scope

Reports we care about include:

- Code execution, path traversal or unsafe deserialization while loading
  models, preprocessing data or serving requests.
- Authentication or authorization bypass in the OpenAI-compatible server or the
  job runner.
- Credentials, tokens or private file contents exposed in logs, generated
  artifacts or error messages.
- Remote code execution or privilege escalation in the kernel build path.

Crashes, model quality, prompt handling and hardware-specific failures without a
security impact are regular bugs. Use the bug report template for those.
