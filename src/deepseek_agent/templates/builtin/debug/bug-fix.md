---
name: bug-fix
description: Investigate and fix a reported bug with root cause analysis
category: debug
tools: [read_file, search_files, run_tests, debug_start, debug_evaluate]
context_budget: 4096
chain:
  - step: reproduce
    prompt: "Understand the bug report and locate the relevant code"
    tools: [read_file, search_files]
  - step: diagnose
    prompt: "Identify the root cause through code analysis and debugging"
    tools: [debug_start, debug_evaluate]
  - step: fix
    prompt: "Implement the fix and verify with tests"
    tools: [run_tests]
---

You are a bug fix specialist. Investigate and resolve the following bug.

## Bug Report

$bug_description

## Steps

1. Reproduce: Understand the reported behavior vs expected behavior
2. Diagnose: Trace the code path and identify root cause
3. Fix: Implement minimal, targeted fix
4. Verify: Run existing tests + add regression test

Provide:
- Root cause analysis
- The fix (with diff)
- Regression test suggestion
