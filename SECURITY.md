# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities privately. Do not open a public issue, and do
not post details in Discussions or Slack.

Email the maintainers at wlsaidhi@gmail.com with the subject
`FastVideo security report`. If this repository has GitHub private
vulnerability reporting enabled, you can use the Security tab and choose
"Report a vulnerability" instead.

Include as much of this as you can:

- The affected component, with the file, CLI command or server endpoint.
- Steps to reproduce, using the smallest example you can give.
- The FastVideo version or commit you tested. `python collect_env.py` prints
  the FastVideo, Python, PyTorch and CUDA versions.
- Any proof of concept, logs or screenshots.

## What we do with a report

We will acknowledge the report and tell you whether we treat it as a security
issue. We cannot promise a fix date. We will keep you updated, coordinate the
disclosure with you, and credit you in the release notes unless you prefer
otherwise. Please give us time to release a fix before you write about the
issue.

## Supported versions

FastVideo is pre-1.0 and moves quickly. We fix security issues on `main` and
ship them in the next release; we do not backport fixes to earlier tags. If you
report against an older version, we may ask you to reproduce on the latest
release.

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
