# Requirements Writing Guide

**Document**: REQUIREMENTS_WRITING_GUIDE.md
**Version**: 2.0
**Last Updated**: 2025-10-26
**Purpose**: Universal standards and principles for writing technical requirements
**Scope**: Project-neutral, applicable to any software project

---

## ⚠️ CRITICAL: Precision and Conciseness

**The fundamental requirement for all requirements documentation**:

### Requirements must be BOTH precise AND concise

- **Precise**: Exact contracts, clear specifications, zero ambiguity
- **Concise**: Minimal words, no redundancy, maximum clarity per line

**This is non-negotiable**. Requirements that are:
- ❌ Verbose and precise → Too long, hard to maintain
- ❌ Concise but vague → Ambiguous, incomplete
- ✅ **Precise and concise** → **Ideal**

**Example of precise AND concise**:
```markdown
**Input**: `file_path` (Path), `mode` (Optional[str])
**Output**: File handle
**Logic**: Open file in read mode if mode is None, else use specified mode
**Reference**: `io/file_handler.py` for implementation patterns
```

**Not this (verbose)**:
```markdown
The function should take two parameters: file_path which is a Path object
pointing to the file location, and mode which is an optional string representing
the file opening mode we want to use. If mode is None, then we should open the
file in read mode. Otherwise, we need to use the specified mode parameter...
```

---

## 1. Purpose

This guide establishes consistent practices for writing technical requirements that are:
- **Precise**: Exact specifications with zero ambiguity
- **Concise**: Minimal words, maximum information density
- **Clear**: Describe WHAT to build, not HOW to build it
- **Actionable**: Sufficient detail for implementation decisions
- **Maintainable**: Easy to update as code evolves

**Priority Order**: Precision > Conciseness > Everything Else

---

## 2. Core Principles

### 2.0 Precision and Conciseness ⚠️ CRITICAL

**Every requirement must achieve BOTH precision AND conciseness.**

**Precision** = Exact, unambiguous, complete specification of contract
**Conciseness** = Minimal words, no redundancy, maximum information per line

**Formula**: `Information Density = Precision / Word Count` → **Maximize this**

**Techniques for Precision + Conciseness**:

1. **Use structured formats** (tables, lists, code blocks)
2. **Omit obvious details** (assume competent developer)
3. **Reference instead of duplicate** (single source of truth)
4. **Use domain terminology** (assume technical knowledge)
5. **Show contracts, not implementation** (inputs, outputs, behavior)

**Example - Comparing approaches**:

| Approach | Word Count | Precision | Information Density |
|----------|------------|-----------|---------------------|
| Verbose prose | 150 words | Medium | Low ❌ |
| Code dump | 80 lines | High | Low ❌ |
| **Structured spec** | **25 words** | **High** | **High ✅** |

**Structured spec example** (25 words):
```
Function: transform_data
Input: data (array [N, D]), params ([[offset, scale], ...])
Output: transformed (array [N, D])
Formula: (data - offset) / scale
Reference: utils/transform.py
```

**Verbose prose example** (150 words):
```
This function is designed to transform input data using the provided parameter
values. It takes two arguments: first, the data which should be an array with
shape [N, D] where N is the number of samples and D is the number of dimensions;
second, the parameters which is a list of lists containing offset and scale
pairs for each dimension. The function will return a transformed version of
the input data with the same shape. The transformation is performed by
subtracting the offset from each dimension and then dividing by the scale
value. This is a standard linear transformation approach. For the complete
implementation details, please refer to the transform.py file in the utils
directory.
```

**Assessment**:
- Verbose: 150 words, medium precision → Information density = 0.67
- Structured: 25 words, high precision → Information density = 4.00
- **Structured is 6× more efficient**

### 2.1 Separation of Concerns

**Requirements describe WHAT. Source code shows HOW.**

✅ **Good** - Requirement describes the contract:
```
File Retrieval: Locate file based on identifier.
- If identifier is None: Use default file
- Otherwise: Find file matching identifier
Reference: utils/file_manager.py for naming convention and implementation
```

❌ **Bad** - Requirement duplicates source code:
```python
def find_file(directory: Path, identifier: Optional[str] = None) -> Path:
    if identifier is None:
        default_path = directory / "default_file.dat"
        if not default_path.exists():
            raise FileNotFoundError(f"Default file not found: {default_path}")
        return default_path
    else:
        file_path = directory / f"file_{identifier}.dat"
        ...
```

**Why**: Source code changes frequently. Requirements should remain stable.

---

### 2.2 Use Pseudocode for Algorithms

When describing complex logic or algorithms, use pseudocode instead of implementation code.

✅ **Good** - High-level algorithm:
```
JIT Model Creation Algorithm:
1. Load checkpoint to CPU
2. Create model instance from config
3. Set eval mode and move to CPU
4. Create example input tensor
5. Compile: try script(), fallback to trace()
6. Verify output and save
```

❌ **Bad** - Full implementation:
```python
checkpoint = torch.load(checkpoint_path, map_location='cpu')
config = load_config(config_path)
model = create_model_from_config(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.to('cpu')
example_input = torch.randn(1, config['input_size']).to('cpu')
try:
    jit_model = torch.jit.script(model)
except Exception:
    jit_model = torch.jit.trace(model, example_input)
```

**Why**: Pseudocode is language-agnostic and focuses on logic, not syntax.

---

### 2.3 Reference Source Code, Don't Duplicate It

Always point developers to existing implementations rather than copying code into requirements.

✅ **Good** - Reference pattern:
```
Parameter Extraction: Extract configuration parameters in JSON format.
- FormatA: array-based format
- FormatB: object-based format
Reference: config/parser.py lines 100-250 (FormatA), 300-450 (FormatB)
```

❌ **Bad** - Code duplication:
```python
# From config/parser.py line 150
def load_params(self):
    with open(self.config_path, 'rb') as f:
        params = pickle.load(f)
    if isinstance(params, ConfigObject):
        return params.values, params.types
    else:
        return params.tolist()
```

**Why**: Single source of truth. Code in requirements becomes stale immediately.

---

### 2.4 Describe Contracts, Not Implementation Details

Focus on inputs, outputs, and behavior contracts.

✅ **Good** - Contract specification:
```
Function: find_resource
Input: resource_dir (Path), resource_id (Optional[int])
Output: Path to resource file
Behavior:
  - If resource_id is None: return default resource
  - Otherwise: return resource matching ID
  - Raise FileNotFoundError if not found
Reference: utils/resource_manager.py
```

❌ **Bad** - Implementation details:
```
The function should use Path.exists() to check if the file is there,
then use the / operator to construct paths, and format strings with
f-strings using the resource ID number...
```

**Why**: Implementation details constrain developers unnecessarily.

---

### 2.5 Concise Error Guidance

Provide error handling principles, not exhaustive error handling code.

✅ **Good** - Error principles:
```
Error Handling:
- Fail-fast: Raise exceptions immediately
- Full context: Include what failed, why, and how to fix
- Traceback: Never swallow exceptions
Reference: See ERROR_HANDLING.md for complete error handling requirements
```

❌ **Bad** - Detailed error handling code:
```python
try:
    resource = load_resource(path)
except FileNotFoundError as e:
    logger.error(f"Resource not found: {e}")
    raise FileNotFoundError(
        f"Resource not found: {path}\n"
        f"Expected pattern: resource_{{id}}.dat\n"
        f"Check utils/resource_manager.py"
    ) from e
except Exception as e:
    logger.error(f"Failed to load: {e}", exc_info=True)
    raise
```

**Why**: Error handling patterns should be consistent across codebase. Define principles once.

---

## 3. Documentation Structure

### 3.1 Requirement Document Template

```markdown
## X. Component Name

### X.1 Purpose
Brief description of component purpose (1-2 sentences).

### X.2 Workflow
High-level data flow or process steps (pseudocode or diagram).

### X.3 Core Requirements

#### X.3.1 Requirement Name

**Input**: Input parameters with types
**Output**: Output with type
**Behavior**: Contract specification (what it does, not how)

**Algorithm** (if complex):
1. High-level step 1
2. High-level step 2
3. High-level step 3

**Reference**: path/to/source_module.ext for implementation

#### X.3.2 Next Requirement
...

### X.4 Interface Specification (if applicable)
Complete interface specification: arguments, types, defaults, constraints.

### X.5 Error Handling
Error categories and handling principles.

**Reference**: See implementation for detailed error handling.
```

---

### 3.2 When to Include Code Examples

Include code ONLY when:
1. **No existing implementation exists** - You're defining something new
2. **Format specification** - Exact JSON/API format required
3. **Interface contract** - Function signatures, type definitions
4. **Common mistakes** - Brief anti-pattern examples (2-3 lines max)

Never include:
- ❌ Complete function implementations
- ❌ Error handling boilerplate
- ❌ Detailed business logic
- ❌ Copy-paste from source code

---

### 3.3 Reference Format

**Good references include**:
1. **Module path**: `utils/resource_manager.py`
2. **Line numbers** (optional): Lines 100-150
3. **What to look for**: "for naming convention", "for validation patterns"
4. **Context**: "See complete implementation" or "for implementation details"

**Example**:
```
Reference: utils/resource_manager.py (lines 100-150) for resource naming convention
```

---

## 4. Error Handling Principles

### 4.1 Fail-Fast Philosophy

**Principle**: Surface errors immediately with full context.

**Requirements should specify**:
- Errors must raise exceptions (not return error codes)
- Exceptions must include full Python traceback
- Error messages must include:
  - What failed
  - Why it failed (root cause)
  - How to fix it (actionable guidance)

**Example Requirement**:
```
Error Handling:
- Fail-fast: All errors raise exceptions immediately
- Full traceback: Never catch exceptions without re-raising
- Actionable messages: Include diagnostic information and remediation steps
Reference: See ERROR_HANDLING.md for detailed guidelines
```

---

### 4.2 Common Error Categories

Specify error categories, not individual error messages:

```
Error Categories:
1. Configuration errors: Invalid config format, missing required fields
2. Not found errors: Resource doesn't exist
3. Validation errors: Data validation failures
4. Runtime errors: Unexpected internal errors

Each error must:
- Use appropriate error code/status (if applicable)
- Include full traceback in logs
- Return actionable error message
```

---

### 4.3 Error Documentation Anti-Pattern

❌ **Don't do this**:
```
Common Errors:
| Error | Cause | Solution |
| "Resource not found: .../resource.xml" | Wrong extension | Use .json not .xml |
| "'str' object has no attribute 'method'" | Type handling | Check type conversion |
```

✅ **Do this instead**:
```
Error Handling:
- Configuration errors: Check source code for actual file naming patterns
- Type errors: Verify type conversions before accessing type-specific methods
- Not found errors: Reference implementation for expected patterns
Reference: utils/ for standard patterns and naming conventions
```

**Why**: Specific error messages become stale. General principles remain valid.

---

## 5. Referencing Source Code

### 5.1 When to Reference vs. When to Specify

**Reference existing code when**:
- ✅ Implementation already exists and is stable
- ✅ Pattern is well-established in codebase
- ✅ Details are complex and likely to change
- ✅ Multiple valid implementation approaches exist

**Specify in requirements when**:
- ✅ New functionality being defined
- ✅ API contract that must remain stable
- ✅ Cross-component interface
- ✅ Critical architectural decision

---

### 5.2 Source Code Reference Template

```markdown
**Requirement**: [Brief description of what needs to be done]

**Details**: [High-level approach or algorithm in pseudocode]

**Reference**: [path/to/source_module.ext] for [what to look for]
- Lines X-Y: [specific functionality]
- Lines A-B: [related pattern]

**Implementation Notes**:
- [Key constraint or consideration]
- [Important edge case]
```

---

### 5.3 Example: Good Reference Pattern

```markdown
#### Resource Selection

**Requirement**: Locate resource file based on identifier or use default.

**Logic**:
- If identifier is None: use default resource
- Otherwise: find resource matching identifier
- Raise FileNotFoundError if not found

**Reference**: `utils/resource_manager.py` (lines 100-150) for:
- Resource naming patterns
- File existence validation
- Error message formatting

**Implementation Note**: Always verify file existence before returning path.
```

---

## 6. CLI Specification Standards

### 6.1 Interface Argument Documentation

Command-line and API interfaces must be **fully specified** in requirements (not referenced):

**Required Information**:
1. Argument/parameter name
2. Type
3. Default value
4. Environment variable (if applicable)
5. Description
6. Valid values/constraints

**Format**:
```markdown
**Arguments**:

| Argument | Type | Default | Env Var | Description |
|----------|------|---------|---------|-------------|
| `--config` | path | - | `CONFIG_PATH` | Path to configuration file (required) |
| `--port` | int | 8000 | `APP_PORT` | Application port number (1-65535) |
| `--mode` | str | auto | `RUN_MODE` | Run mode: auto, manual, or test |

**Examples**:
```bash
# Basic usage
application --config config.json

# Custom port
application --port 9000
```
```

**Why**: Interfaces are user-facing contracts. They must be fully documented, not discovered by reading code.

---

## 7. Pseudocode Guidelines

### 7.1 When to Use Pseudocode

Use pseudocode for:
- Complex algorithms with multiple steps
- Control flow logic
- Data transformation pipelines
- State machine transitions

Don't use pseudocode for:
- Simple assignments
- Single function calls
- Obvious operations

---

### 7.2 Pseudocode Format

**Good pseudocode**:
- Uses indentation for structure
- Focuses on logic, not syntax
- Includes decision points
- Shows high-level flow

**Example**:
```
Data Denormalization Algorithm:
1. Load scalers (input_scaler, output_scaler)
2. For each prediction:
   a. Identify target indices
   b. Extract corresponding scaler parameters
   c. Apply inverse transform: value * std + mean
3. Return denormalized predictions

Reference: See trainer/transforms/denormalize.py for implementation
```

---

## 8. Common Anti-Patterns

### Anti-Pattern 1: Code-as-Requirements

❌ **Problem**: Pasting source code into requirements
```markdown
## Implementation
```python
def process_data(data):
    transformed = (data - offset) / scale
    return transformed
```
```

✅ **Solution**: Describe contract and reference implementation
```markdown
## Data Processing
Transform input data using linear transformation.
Reference: utils/transform.py
```

---

### Anti-Pattern 2: Over-Specification

❌ **Problem**: Constraining implementation unnecessarily
```markdown
Use a dictionary to store the cache, with string keys and model objects as values.
Implement a manual LRU eviction policy by tracking access times in a separate list.
When cache is full, iterate through the list to find the least recently used item.
```

✅ **Solution**: Specify behavior, allow implementation flexibility
```markdown
Model Pool: LRU cache for loaded models (max size configurable).
Reference: Standard LRU cache implementation patterns
```

---

### Anti-Pattern 3: Outdated Examples

❌ **Problem**: Specific error messages or file extensions
```markdown
Common Errors:
- "File not found: best_model.pth" → Use best_model.pt instead
- "Config error: missing field" → Add required field to config
```

✅ **Solution**: General error handling principles
```markdown
Error Handling:
- File not found: Reference trainer source for naming conventions
- Config errors: Validate against schema, show full traceback
```

---

### Anti-Pattern 4: Missing References

❌ **Problem**: No guidance on where to find implementation details
```markdown
Implement parameter extraction for both config types.
```

✅ **Solution**: Always include reference
```markdown
Parameter Extraction: Extract configuration parameters from both config types.
Reference: config/parser.py for JSON and YAML parser implementations
```

---

## 9. Maintenance Guidelines

### 9.1 Keeping Requirements Current

**When source code changes**:
1. ✅ Update references if file paths change
2. ✅ Update algorithm pseudocode if logic fundamentally changes
3. ✅ Keep contracts stable (only update on breaking changes)
4. ❌ Don't update implementation details (they're in source code)

**When requirements drift**:
1. Review references for accuracy
2. Update high-level descriptions
3. Add new sections for new functionality
4. Archive obsolete sections

---

### 9.2 Version Control

**Requirements documents should**:
- Include version number and last updated date
- Track major changes in version history
- Reference stable source code (commit hash or version tag when possible)

**Example**:
```markdown
**Version**: 1.3
**Last Updated**: 2025-10-26
**Source Code Reference**: v2.1+ or commit abc123def
```

---

### 9.3 Revision Summary Tracking ⚠️ MANDATORY

**FUNDAMENTAL RULE**: Every requirement document set MUST maintain a `REVISION_SUMMARY.md` file.

**Purpose**:
- Complete audit trail of all changes
- Document corrections and their root causes
- Maintain transparency and accountability
- Enable traceability for all modifications

**Update Frequency**: **EVERY TIME** any requirement document is modified

**Required Content**:
1. **Initial Revision**: Document creation, version, principles applied
2. **Subsequent Corrections**: Each change with:
   - Date
   - Issue description (what was wrong)
   - Root cause (why the error occurred)
   - Corrections made (what changed)
   - Files affected (with occurrence counts)
   - Impact assessment

**Format**:
```markdown
# Requirements Revision Summary

**Date**: 2025-11-02
**Revised According to**: REQUIREMENT_WRITING_GUIDE.md
**Version**: 2.0

## Key Changes
[Document initial revision details]

## Subsequent Corrections (YYYY-MM-DD)

### Correction N: [Brief Title]
**Issue**: [What was incorrect in the documentation]

**Root Cause**: [Why the error occurred]

**Fix**: [What was corrected]
- Specific change 1
- Specific change 2

**Files Updated**:
- file1.md (N occurrences: [details])
- file2.md (N occurrences: [details])

**Impact**: [Severity and effect of correction]
```

**Example**:
```markdown
## Subsequent Corrections (2025-11-03)

### Correction 2: Indicator Name and Field Name
**Issue**: Documentation referenced wrong indicator name and field name.

**Corrections**:
1. **Indicator name**: `ZZefClose_09262025MG` → `ZZefClose_09262025RMG`
2. **Field name**: `close` → `_close` (primary input field)

**Files Updated**:
- 01-overview.md (3 occurrences)
- 02-data-sources.md (13 occurrences: headers, tables, sv_object class, examples)
- 03-feature-engineering.md (1 occurrence)
- 04-implementation-guide.md (4 occurrences)
- README.md (3 occurrences)

**Impact**: Critical corrections ensuring accurate implementation guidance.
```

**Benefits**:
- ✅ Complete change history
- ✅ Root cause documentation prevents repeated errors
- ✅ Transparency for all stakeholders
- ✅ Easy to understand evolution of requirements
- ✅ Accountability for corrections

**Anti-Pattern** ❌:
```markdown
❌ Don't do this:
- Update requirements without documenting changes
- Document "what changed" without "why it was wrong"
- Skip updating revision summary for "minor" changes
```

**Best Practice** ✅:
```markdown
✅ Always do this:
- Update REVISION_SUMMARY.md for EVERY change
- Include root cause analysis
- List all affected files
- Assess impact of correction
- Keep chronological order
```

**Enforcement**: Requirements without maintained REVISION_SUMMARY.md are incomplete and do not meet documentation standards.

---

## 10. Review Checklist

Before finalizing requirements, verify:

**Precision and Conciseness ⚠️ CRITICAL**:
- [ ] **High information density** - Every word adds value
- [ ] **Zero ambiguity** - Only one interpretation possible
- [ ] **No redundancy** - Each fact stated once
- [ ] **Structured format** - Tables, lists, blocks (not prose)
- [ ] **Complete contracts** - All inputs, outputs, behaviors specified

**Content**:
- [ ] Describes WHAT to build, not HOW to build it
- [ ] Uses pseudocode for algorithms, not implementation code
- [ ] Includes references to existing source code
- [ ] Specifies contracts (inputs, outputs, behavior)
- [ ] CLI fully documented with all arguments

**Clarity**:
- [ ] No copy-pasted source code (except brief examples)
- [ ] Error principles defined, not error instances
- [ ] References include file paths and what to look for
- [ ] Pseudocode is readable and high-level

**Maintenance**:
- [ ] Won't become outdated when code changes
- [ ] Single source of truth (requirements for contract, source for implementation)
- [ ] Version number and date included
- [ ] **REVISION_SUMMARY.md updated** (if modifying existing requirements)

**Self-Test**:
- [ ] Can I remove any word without losing precision? → **Remove it**
- [ ] Can I make it more precise without adding words? → **Do it**
- [ ] Can I use a table/list instead of prose? → **Use it**
- [ ] Am I duplicating source code? → **Reference it instead**

---

## 11. Example: Before and After

### Before (Anti-Pattern)

```markdown
#### Checkpoint Selection

**CRITICAL**: Checkpoint file extensions and naming patterns

From trainer/utils/checkpoint.py (lines 112, 127):
- Best model: best_model.pt (NOT .pth)
- Epoch checkpoints: checkpoint_epoch_{epoch}.pt

**Implementation**:
```python
def find_checkpoint(checkpoint_dir: Path, desired_step: Optional[int] = None) -> Path:
    if desired_step is None:
        best_path = checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(
                f"Best model not found: {best_path}\n"
                f"Expected file: best_model.pt (note the .pt extension)\n"
            )
        return best_path
    else:
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{desired_step}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path
```

**Common Error**: Looking for best_model.pth instead of best_model.pt
```

### After (Following Guide)

```markdown
#### Checkpoint Selection

**Requirement**: Locate checkpoint file based on step number or use best model.

**Input**: `checkpoint_dir` (Path), `desired_step` (Optional[int])
**Output**: Path to checkpoint file

**Logic**:
- If desired_step is None: return best model checkpoint
- Otherwise: return checkpoint for specific epoch
- Raise FileNotFoundError if checkpoint doesn't exist

**Reference**: `trainer/utils/checkpoint.py` for:
- Checkpoint naming patterns and file extensions
- File validation and error handling

**Implementation Note**: Verify file exists before returning path.
```

**Reduction**: ~30 lines → 10 lines
**Maintainability**: ✅ Won't break when file extension or error messages change
**Clarity**: ✅ Focuses on contract, not implementation

---

## Summary

**Golden Rules** (in priority order):

### ⚠️ CRITICAL - Rule #0:
**Requirements must be PRECISE and CONCISE. No exceptions.**
- Maximize information density: `Precision / Word Count`
- Use structured formats: tables, lists, code blocks
- Reference source code, don't duplicate it
- Every word must add value

### Core Rules:
1. **Requirements = WHAT** (contracts), Source Code = HOW (implementation)
2. **Reference, don't duplicate** - Single source of truth
3. **Pseudocode for algorithms** - High-level logic only
4. **Principles for errors** - Not specific messages
5. **Full specs for CLIs** - Complete argument tables
6. **Contracts remain stable** - Implementations evolve
7. **REVISION_SUMMARY.md mandatory** - Update for EVERY change

**Quality Metrics**:
- ✅ **Information Density**: High (3+ precision points per 25-30 words)
- ✅ **Precision Score**: Complete (all contracts specified)
- ✅ **Conciseness Score**: Minimal (no redundant words)
- ✅ **Maintainability**: High (won't become stale)

**Result**: Requirements that are **precise**, **concise**, **clear**, **maintainable**, and **actionable**.

---

**Application**:

This guide is project-neutral and can be applied to any software development project:
- Web applications, APIs, microservices
- Desktop applications, CLI tools
- Libraries, frameworks, SDKs
- Data pipelines, ML systems
- Embedded systems, firmware

The principles remain the same across all domains: maximize information density while maintaining precision.

---
