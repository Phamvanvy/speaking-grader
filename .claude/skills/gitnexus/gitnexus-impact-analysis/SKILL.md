---
description: Analyze blast radius of code changes — identify callers, callees, affected processes, and risk before editing
---

# Impact Analysis

**Goal:** Understand what breaks before you break it. Run impact analysis on any function, class, method, or module BEFORE making changes, so you know the blast radius.

> **RULE:** Never edit a symbol without first checking its impact. A 5-line impact query takes 2 seconds; fixing unintended regressions takes 2 hours.

## Workflow

### 1. Identify the target

Decide what you will change — a function name, a class, a config key, a shared utility.

### 2. Run impact upstream

```
impact({
  target: "functionName",
  direction: "upstream",
  maxDepth: 3,
  limit: 50
})
```

This returns every symbol that **depends on** the target, grouped by depth:

| Depth | Meaning |
|-------|---------|
| d=1 | **WILL BREAK** — direct callers / importers |
| d=2 | **LIKELY AFFECTED** — indirect callers |
| d=3 | **MAY NEED TESTING** — transitive dependents |

It also returns:

- **risk:** `LOW` / `MEDIUM` / `HIGH` / `CRITICAL`
- **affected_processes:** execution flows that pass through this symbol
- **affected_modules:** functional areas hit (direct vs indirect)

### 3. Interpret the risk

| Risk | Action |
|------|--------|
| **LOW** (0-3 consumers) | Proceed with change; update d=1 callers |
| **MEDIUM** (4-9 consumers, or mismatches) | Review all d=1 + d=2; write tests first |
| **HIGH** (10+ consumers, or mismatches with 4+) | **Warn the user.** Consider alternative approach. |
| **CRITICAL** (hub symbol, base class, core config) | **Stop.** Refactor with extreme care. Migrate in small steps. |

### 4. Drill into high-risk symbols

For each d=1 symbol that concerns you:

```
context({name: "highRiskSymbol"})
```

This shows every caller, callee, method, property, and process membership — full 360° view.

### 5. Check downstream too (optional)

If you want to know what the target **depends on** (libraries it imports, base classes it extends):

```
impact({
  target: "functionName",
  direction: "downstream",
  maxDepth: 2
})
```

Useful for: "If this library changes, what happens inside my function?"

## Quick Reference

| Question | Tool | Parameters |
|----------|------|------------|
| "What calls this function?" | `impact` | `direction: "upstream"` |
| "What does this class import?" | `impact` | `direction: "downstream"` |
| "Which processes use this?" | `context` | `name: "symbol"` |
| "How risky is this change?" | `impact` | look at `risk` field |
| "Show me only top-level callers" | `impact` | `maxDepth: 1` |
| "Hub symbol — too many results" | `impact` | `summaryOnly: true` |
| "Field read/write tracking" | `impact` | `relationTypes: ["ACCESSES"]` |
| "Method override chain" | `impact` | `relationTypes: ["HAS_METHOD", "METHOD_OVERRIDES"]` |

## Tips

- **Hub symbols** (base errors, shared utilities, config loaders) have hundreds of d=1 dependents. Use `summaryOnly: true` first to see counts, then drill into specific depths with `limit`/`offset`.
- **`relationTypes`** defaults to `CALLS`/`IMPORTS`/`EXTENDS`/`IMPLEMENTS`. Add `ACCESSES` for field-level tracking, `HAS_METHOD`/`HAS_PROPERTY` for class members.
- **Group mode:** set `repo: "@groupName"` for cross-repo impact. The tool fans out through the contract bridge automatically.
- **Before committing:** always run `detect_changes()` to verify your edits only affected expected symbols.

## Example

You want to rename `validate_user_input` → `validate_request`:

```
# 1. Impact upstream — who calls it?
impact({target: "validate_user_input", direction: "upstream", maxDepth: 2})

# Result: risk=MEDIUM, 7 direct callers, 2 processes affected

# 2. Check a high-risk caller
context({name: "api_handler_process"})

# 3. Coordinated rename (uses call graph, NOT find-and-replace)
rename({symbol_name: "validate_user_input", new_name: "validate_request"})

# 4. Verify only expected symbols changed
detect_changes()
```

## Anti-patterns

- ❌ **Find-and-replace** for renames → misses imports, breaks string literals
- ❌ **Editing without impact check** → silent regressions in callers you forgot
- ❌ **Ignoring CRITICAL risk** → hub symbol changes need migration strategy
- ❌ **Skipping `detect_changes()` before commit** → untracked side effects