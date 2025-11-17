# CLAUDE.md - WOS Documentation & Project Generator (EGG)

**Document**: CLAUDE.md
**Version**: 1.0
**Last Updated**: 2025-11-14
**Purpose**: Guide for maintaining WOS documentation base and project generator
**Audience**: Claude Code AI assistant and maintainers

---

## Project Overview

### What This Project Is

This is the **WOS-EGG (Wolverine Operating System - Environment Generation & Guidance)** project, which serves as:

1. **Documentation Base**: Comprehensive WOS system documentation (12 chapters in wos/ directory)
2. **Project Generator**: CLI tool (`create_project.py`) that creates new WOS projects
3. **Template System**: Project templates and boilerplate code
4. **Development Guide**: Instructions for users creating indicators and strategies

**Purpose**: Enable users to quickly create and develop WOS trading indicators and strategies with complete documentation and tooling.

**This is NOT**:
- A WOS indicator project
- A trading strategy implementation
- An executable trading system

**This IS**:
- The source of truth for WOS documentation
- The tool that generates WOS projects
- The maintenance base for all WOS guidance

---

## Project Structure

```
money-engineering/
├── wos/                           # WOS documentation (12 chapters)
│   ├── 01-overview.md
│   ├── 02-data-sources.md
│   ├── 03-feature-engineering.md
│   ├── ...
│   └── 12-appendix.md
├── organic/                       # Organic amendment proposals
│   ├── fix001.md                 # Amendment proposal 1
│   ├── fix002.md                 # Amendment proposal 2 (future)
│   └── ...                       # Additional amendment proposals
├── templates/                     # Project templates
│   └── template_system.md
├── create_project.py             # CLI tool for project generation
├── README.md                     # User-facing documentation
├── CLI_USAGE.md                  # CLI tool user guide
├── REQUIREMENT_WRITING_GUIDE.md  # Documentation standards
├── UPDATE_LOG.md                 # Mandatory revision tracking
├── CLAUDE.md                     # This file
└── .devcontainer/                # Container configuration
```

---

## Core Principles (From REQUIREMENT_WRITING_GUIDE.md)

### Critical Rule: Precision AND Conciseness

**All documentation must be**:
- **Precise**: Exact specifications, zero ambiguity
- **Concise**: Minimal words, maximum information density
- **Structured**: Tables, lists, code blocks (not prose)
- **Referenced**: Point to source code, don't duplicate it

**Information Density Formula**: `Precision / Word Count` → **Maximize this**

### Documentation Standards

1. **Describe WHAT, not HOW**: Requirements specify contracts, not implementation
2. **Use pseudocode**: High-level algorithms only
3. **Reference source**: Always include file paths
4. **No code duplication**: Single source of truth
5. **Maintain UPDATE_LOG.md**: Document ALL changes

---

## Regular Workflows

### Workflow 1: Update WOS Documentation

**When**: Source code changes, features added, bugs fixed, corrections needed

**Steps**:

1. **Identify affected chapters**:
   ```
   Grep for related concepts across wos/*.md files
   Read affected sections
   ```

2. **Update documentation following REQUIREMENT_WRITING_GUIDE.md**:
   - Maintain precision AND conciseness
   - Use structured formats (tables, lists, blocks)
   - Reference source code, don't duplicate
   - Update pseudocode if logic fundamentally changed
   - Keep contracts stable

3. **Update UPDATE_LOG.md** (MANDATORY):
   ```markdown
   ## Correction N: [Brief Title] (YYYY-MM-DD)
   **Issue**: [What was incorrect]
   **Root Cause**: [Why the error occurred]
   **Fix**: [What was corrected]
   **Files Updated**: [List with occurrence counts]
   **Impact**: [Severity assessment]
   ```

4. **Verify consistency**:
   - Check cross-references between chapters
   - Ensure examples align with current code
   - Validate all file path references
   - Review related chapters for consistency

5. **Update version numbers**:
   - Increment version in affected files
   - Update "Last Updated" date

**Anti-Patterns to Avoid**:
- ❌ Pasting source code into docs
- ❌ Verbose prose instead of structured specs
- ❌ Updating docs without UPDATE_LOG.md
- ❌ Specific error messages (use principles instead)

---

### Workflow 2: Update CLI Tool (create_project.py)

**When**: Template changes, new project types, configuration updates

**Steps**:

1. **Understand the change**:
   - Read create_project.py to understand current logic
   - Identify what needs modification
   - Check if templates need updates

2. **Modify CLI tool**:
   - Update argument parsing if needed
   - Modify template generation logic
   - Update validation rules
   - Ensure backward compatibility where possible

3. **Update templates**:
   - Modify templates in create_project.py
   - Ensure generated files follow WOS standards
   - Test template generation

4. **Update CLI_USAGE.md**:
   - Document new options/features
   - Add examples for new functionality
   - Update option tables
   - Keep concise and structured

5. **Test thoroughly**:
   ```bash
   # Test basic project creation
   ./create_project.py --name TestIndicator --market DCE --securities i

   # Test multi-market
   ./create_project.py --name TestMulti --market DCE,SHFE --securities-DCE i --securities-SHFE cu

   # Test composite
   ./create_project.py --name TestComp --type composite

   # Test interactive mode
   ./create_project.py --interactive
   ```

6. **Verify generated projects**:
   - Check all files created correctly
   - Validate JSON configurations
   - Ensure symlinks work (wos/ directory)
   - Test debug configurations

---

### Workflow 3: Add New Documentation Chapter

**When**: New framework features, new tiers, new concepts

**Steps**:

1. **Plan chapter structure**:
   ```markdown
   ## X. Chapter Name

   ### X.1 Purpose
   Brief description (1-2 sentences)

   ### X.2 Workflow
   High-level process steps (pseudocode)

   ### X.3 Core Requirements
   #### X.3.1 Requirement Name
   **Input**: Parameters with types
   **Output**: Output with type
   **Behavior**: Contract specification
   **Reference**: path/to/source_module

   ### X.4 Interface Specification
   Complete interface docs

   ### X.5 Error Handling
   Error principles
   ```

2. **Write following REQUIREMENT_WRITING_GUIDE.md**:
   - Use structured formats
   - Maximum information density
   - Reference existing code
   - No implementation details

3. **Add to index**:
   - Update README.md to reference new chapter
   - Add to table of contents
   - Update cross-references in related chapters

4. **Create UPDATE_LOG.md entry**:
   - Document as "Initial Revision"
   - Include version, date, principles applied

5. **Review checklist** (from REQUIREMENT_WRITING_GUIDE.md Section 10):
   - [ ] High information density
   - [ ] Zero ambiguity
   - [ ] No redundancy
   - [ ] Structured format
   - [ ] Complete contracts
   - [ ] References included
   - [ ] Won't become stale

---

### Workflow 4: Fix Documentation Errors

**When**: User reports issues, inconsistencies found, outdated information

**Steps**:

1. **Validate the error**:
   - Read current documentation
   - Check against source code
   - Verify user's report
   - Identify root cause

2. **Search for all occurrences**:
   ```bash
   # Use grep to find all instances
   grep -r "incorrect_pattern" wos/

   # Check related terms
   grep -r "related_concept" wos/
   ```

3. **Correct ALL occurrences**:
   - Fix each instance systematically
   - Maintain consistency across files
   - Verify context remains correct

4. **Document in UPDATE_LOG.md** (MANDATORY):
   ```markdown
   ## Correction N: [Error Type] (YYYY-MM-DD)
   **Issue**: Documentation referenced X instead of Y

   **Root Cause**: [Why the error occurred]

   **Corrections**:
   1. Changed X to Y in all references
   2. Updated related examples

   **Files Updated**:
   - 01-overview.md (3 occurrences)
   - 02-data-sources.md (13 occurrences)
   - 04-implementation-guide.md (4 occurrences)

   **Impact**: Critical - affects implementation guidance
   ```

5. **Verify consistency**:
   - Check cross-references
   - Validate examples
   - Test any code snippets

---

### Workflow 5: User Requests New Feature/Template

**When**: User asks for new project type, template, or capability

**Steps**:

1. **Clarify requirements**:
   - Ask user for specific needs
   - Understand use case
   - Identify constraints
   - Check existing templates

2. **Design solution**:
   - Plan template structure
   - Identify required files
   - Design configuration format
   - Consider edge cases

3. **Implement in create_project.py**:
   - Add new project type option
   - Create template generation logic
   - Add validation rules
   - Test thoroughly

4. **Document in CLI_USAGE.md**:
   - Add new examples
   - Update option tables
   - Document new workflows
   - Keep concise and structured

5. **Create documentation** (if needed):
   - Add chapter to wos/ if new concept
   - Update existing chapters if extension
   - Follow REQUIREMENT_WRITING_GUIDE.md
   - Update UPDATE_LOG.md

6. **Test end-to-end**:
   - Generate project with new template
   - Verify all files correct
   - Test in container
   - Validate with user

---

## Organic Amendment Process

### Overview

**Organic amendments** are user-submitted documentation corrections and improvements stored in the `organic/` directory. This provides a structured way for users to propose changes without directly editing documentation.

**Purpose**:
- Capture user-identified issues in documentation
- Provide clear amendment proposals
- Maintain audit trail of user feedback
- Enable systematic application of corrections

### Amendment File Format

**Location**: `organic/fix###.md` (numbered sequentially)

**Structure**:
```markdown
# WOS Documentation Issues

[Brief list of issues]

## Issue 1: [Title]

### Problem
[Description of the problem]

### Proposed Fix
[Detailed fix with code examples if applicable]

### Affected Files
- file1.md (section X.Y)
- file2.md (section Z.W)
```

### Workflow: Processing Organic Amendments

**When**: User provides amendment file in `organic/` directory

**Steps**:

1. **Read the amendment file**:
   ```bash
   # User provides path
   /Users/.../organic/fix001.md
   ```

2. **Validate the amendments**:
   - Check if issues are valid
   - Verify against source code
   - Identify affected documentation files

3. **Apply amendments systematically**:
   - Update each affected file
   - Follow REQUIREMENT_WRITING_GUIDE.md principles
   - Maintain precision and conciseness
   - Cross-check related sections

4. **Document in UPDATE_LOG.md** (MANDATORY):
   ```markdown
   ## Correction N: [From organic/fix###.md] (YYYY-MM-DD)

   **Source**: organic/fix###.md

   **Issues Fixed**:
   1. [Issue 1 summary]
   2. [Issue 2 summary]

   **Root Cause**: [Analysis of why errors occurred]

   **Corrections Applied**:
   - [File 1]: [What changed]
   - [File 2]: [What changed]

   **Files Updated**:
   - wos/04-structvalue-and-sv_object.md (N changes)
   - wos/05-stateless-design.md (M changes)
   - wos/07-tier1-indicator.md (K changes)

   **Impact**: [Severity and effect]
   ```

5. **Update version numbers**:
   - Increment versions in affected files
   - Update "Last Updated" dates

6. **Verify consistency**:
   - Check cross-references
   - Validate examples
   - Test any code patterns

7. **Respond to user**:
   - Confirm amendments applied
   - List files updated
   - Note UPDATE_LOG.md entry

### Amendment Categories

Common types of organic amendments:

1. **Technical Corrections**:
   - Incorrect API usage
   - Wrong function signatures
   - Outdated patterns

2. **Clarifications**:
   - Ambiguous descriptions
   - Missing context
   - Unclear examples

3. **Additions**:
   - Missing documentation
   - New patterns
   - Best practices

4. **Structural Improvements**:
   - Better organization
   - Improved information density
   - Enhanced readability

### Example: Processing fix001.md

**Issues in fix001.md**:
- `copy_to_sv`/`from_sv` inverse relationship unclear
- `_on_cycle_pass` data leakage prevention missing
- `bar_since_start` initialization tracking undocumented
- Package import patterns need clarification
- Time format conversions need documentation

**Process**:
1. Read fix001.md thoroughly
2. Identify affected chapters (04, 05, 07)
3. Apply fixes following REQUIREMENT_WRITING_GUIDE.md
4. Create comprehensive UPDATE_LOG.md entry
5. Update version numbers
6. Verify all cross-references

### Best Practices

**Do**:
- ✅ Keep amendment files concise and structured
- ✅ Include code examples for clarity
- ✅ Reference specific sections/line numbers
- ✅ Explain root cause when known
- ✅ Apply ALL occurrences of each issue

**Don't**:
- ❌ Skip documenting in UPDATE_LOG.md
- ❌ Apply partial fixes (fix only some occurrences)
- ❌ Ignore cross-references to updated sections
- ❌ Forget to update version numbers

### Retention Policy

**Amendment files**:
- Keep permanently in `organic/` directory
- Serve as historical record
- Enable traceability
- Reference from UPDATE_LOG.md

**Benefits**:
- Complete audit trail
- User feedback loop
- Continuous improvement
- Documentation quality assurance

---

## Common Tasks

### Task: Search Documentation for Concept

**When**: User asks "where is X documented?" or "how do I do Y?"

**Approach**:
```bash
# Search for keywords
grep -r "concept_name" wos/

# Search for patterns
grep -r "sv_object" wos/

# Search for specific implementations
grep -r "on_bar" wos/

# Use multiple keywords
grep -r "composite.*strategy" wos/
```

**Then**: Direct user to specific chapter and section.

---

### Task: Validate Documentation Consistency

**When**: After updates, before releases, periodic reviews

**Checklist**:
- [ ] All file path references valid
- [ ] Cross-references between chapters correct
- [ ] Examples align with current framework
- [ ] Version numbers updated
- [ ] UPDATE_LOG.md current
- [ ] No code duplication
- [ ] Structured format maintained
- [ ] Information density high

**Tools**:
```bash
# Check for broken references
grep -r "Reference:" wos/ | # extract paths | # verify exist

# Check for code blocks (should be minimal)
grep -rn "```python" wos/ | wc -l  # Should be low

# Check for verbose prose (flag long paragraphs)
# Should favor tables/lists/blocks
```

---

### Task: Generate Test Project

**When**: Testing CLI tool, validating templates, user support

**Commands**:
```bash
# Basic indicator
./create_project.py --name TestBasic --market DCE --securities i

# Multi-market
./create_project.py --name TestMulti \\
    --market DCE,SHFE \\
    --securities-DCE i,j \\
    --securities-SHFE cu,al \\
    --granularity 300,900

# Composite
./create_project.py --name TestComp --type composite

# Interactive
./create_project.py --interactive
```

**Validation**:
```bash
cd TestBasic/
# Check files exist
ls -la
# Verify symlink
ls -la wos/
# Check JSON validity
python -m json.tool uin.json
python -m json.tool uout.json
# Check Python syntax
python -m py_compile TestBasic.py
```

---

### Task: Update Market/Securities Information

**When**: New exchanges added, securities listed, codes changed

**Files to Update**:
1. **create_project.py**: Market definitions, securities lists
2. **CLI_USAGE.md**: Market tables, examples
3. **wos/02-data-sources.md**: Market descriptions
4. **README.md**: Market references

**Process**:
1. Update market definitions in create_project.py
2. Test project generation with new markets
3. Update documentation
4. Add examples to CLI_USAGE.md
5. Document in UPDATE_LOG.md

---

## Error Handling

### User Reports Documentation Bug

**Response Pattern**:
1. **Acknowledge**: "I'll check the documentation and fix this"
2. **Search**: Find all occurrences
3. **Validate**: Verify against source code
4. **Fix**: Update all instances
5. **Document**: Update UPDATE_LOG.md
6. **Confirm**: "Fixed in [files]. Updated UPDATE_LOG.md"

### User Confused by Documentation

**Response Pattern**:
1. **Identify issue**: What's unclear?
2. **Check documentation**: Is it precise? Concise?
3. **Improve**: Restructure if needed (table vs prose)
4. **Add example**: If helpful (keep minimal)
5. **Update**: Follow revision process

### CLI Tool Generates Invalid Project

**Response Pattern**:
1. **Reproduce**: Generate project with same parameters
2. **Identify issue**: What's wrong?
3. **Fix CLI tool**: Update create_project.py
4. **Fix template**: If template issue
5. **Test**: Regenerate and validate
6. **Document**: Update CLI_USAGE.md if needed

---

## Quality Standards

### Documentation Quality Metrics

**High Quality**:
- ✅ Information density: 3+ precision points per 25-30 words
- ✅ Structured format: 80%+ tables/lists/blocks
- ✅ Zero code duplication
- ✅ All references valid
- ✅ UPDATE_LOG.md current

**Low Quality** (needs improvement):
- ❌ Verbose prose paragraphs
- ❌ Code snippets from source
- ❌ Vague specifications
- ❌ Missing references
- ❌ Outdated UPDATE_LOG.md

### CLI Tool Quality Metrics

**High Quality**:
- ✅ All project types generate correctly
- ✅ Generated projects pass validation
- ✅ Interactive mode user-friendly
- ✅ Error messages actionable
- ✅ Documentation complete

---

## Don'ts - Critical Anti-Patterns

### Documentation Don'ts

1. ❌ **Don't duplicate source code** in documentation
   - Reference it instead
   - Use pseudocode for algorithms

2. ❌ **Don't write verbose prose** when structure works
   - Use tables, lists, code blocks
   - Maximize information density

3. ❌ **Don't skip UPDATE_LOG.md** updates
   - Required for EVERY change
   - Include root cause analysis

4. ❌ **Don't specify implementation details**
   - Describe contracts (what)
   - Not implementation (how)

5. ❌ **Don't create stale documentation**
   - Reference source code
   - Keep high-level
   - Verify before committing

### CLI Tool Don'ts

1. ❌ **Don't break existing projects**
   - Maintain backward compatibility
   - Test thoroughly before changes

2. ❌ **Don't generate invalid configurations**
   - Validate all JSON
   - Check array alignment
   - Verify file paths

3. ❌ **Don't skip testing**
   - Test all project types
   - Verify generated files
   - Check container setup

---

## Reference Files

### Key Documentation Files

**REQUIREMENT_WRITING_GUIDE.md**:
- Section 2.0: Precision and Conciseness (CRITICAL)
- Section 3.1: Document Template
- Section 9.3: Revision Summary Tracking (MANDATORY)
- Section 10: Review Checklist

**CLI_USAGE.md**:
- Project creation examples
- Market and securities reference
- Workflow after creation

**README.md**:
- User-facing overview
- Quick start guide
- Framework rules

### Framework Source References

**When users ask about implementation**:
- Point to wos/ documentation first
- Reference source code paths
- Don't duplicate code
- Use pseudocode for explanation

---

## Development Environment

### Working in This Project

**Setup**:
```bash
# This project doesn't need container
# Work directly on host

# Make CLI executable
chmod +x create_project.py

# Test CLI
./create_project.py --help
```

**Testing Changes**:
```bash
# Test documentation readability
# Read through updated chapters

# Test CLI tool
./create_project.py --name TestProj --market DCE --securities i
cd TestProj/
ls -la
python -m json.tool uin.json
cd ..
rm -rf TestProj/

# Test in container
cd TestProj/
code .
# Reopen in container
# Test debug configurations
```

---

## Summary

**This project maintains**:
1. WOS documentation (wos/ directory)
2. Project generator CLI tool (create_project.py)
3. User guides (README.md, CLI_USAGE.md)
4. Documentation standards (REQUIREMENT_WRITING_GUIDE.md)

**Regular workflows**:
1. Update WOS documentation
2. Update CLI tool
3. Add new chapters
4. Fix documentation errors
5. Add new features/templates

**Critical rules**:
- ✅ Precision AND conciseness (non-negotiable)
- ✅ Update UPDATE_LOG.md (mandatory)
- ✅ Reference code, don't duplicate
- ✅ Structured formats over prose
- ✅ Test CLI tool thoroughly

**Quality goal**: Maximum information density, zero ambiguity, easy maintenance.

---

**When in doubt**: Consult REQUIREMENT_WRITING_GUIDE.md and follow its principles religiously.
