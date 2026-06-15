---
name: add-tool
description: Add new MCP tools following standard patterns and guidelines
user-invocable: true
---

# Add New MCP Tool

Add a new MCP tool to this server following established patterns and guidelines.

## Process Overview

This command guides you through a systematic process to:
1. Analyze API behavior to understand the operation
2. Study existing codebase patterns
3. Implement consistent code following project guidelines
4. Review for compliance with project standards

---

## Phase 1: Understand the Project

### Study Project Guidelines
- Read the project's CLAUDE.md file to understand architectural patterns
- Identify the distinction between operation types (read-only vs async)
- Review parameter ordering conventions
- Understand error handling patterns

### Study Existing Implementation
- Search for similar existing tools in the services layer
- Examine how existing tools are registered in the MCP server
- Review how existing tools handle errors
- Identify polling and retry patterns for async operations

**Action:** Use file search tools to locate similar implementations before writing any code.

---

## Phase 2: Gather Requirements

### Understand the Operation
Ask for or determine:
- HTTP method and endpoint (or equivalent operation)
- Request/response structure
- Whether response includes async operation indicators
- Business purpose of the operation

### Determine Operation Type

| Type | Characteristics | Template |
|------|----------------|----------|
| **Read-only query** | GET/query, no side effects | No retry params, return directly |
| **Async with polling** | Create/delete/update, returns tracking ID | `maxRetries`, `pollIntervalMs` |
| **Conditional async** | Multi-step, some steps conditional | Spread operator with conditional |
| **Retrieve-then-update** | GET current state, merge, PUT back | Preserve all existing fields |
| **Type-based conditional** | Different payloads per resource type | Switch/if on type |
| **Type-based early return** | Entirely different flows per type | Early return pattern |
| **Optional payload** | Empty body support for PUT/POST | Conditional body construction |

**Action:** Confirm operation type before proceeding.

---

## Phase 3: Pattern Analysis

### Find the Most Similar Existing Tool
Search the codebase for:
- Functions with the same operation type
- Functions calling similar API endpoints
- Functions with matching patterns (polling, query, CRUD)

### Copy the Pattern Exactly
From the similar tool, note:
- Parameter order and types
- Default values
- API URL construction
- Request/response handling
- Error handling structure
- Polling logic (if async)

**Action:** Read the complete implementation of the most similar tool to use as a template.

---

## Phase 4: Implementation Planning

### Create Checklist
Plan the implementation:
1. Implement service layer function
2. Register tool in MCP server
3. Implement request handler
4. Review against guidelines
5. Review tool description for AI agent clarity

### Identify Potential Issues
Before coding, consider:
- Unique aspects not covered by existing patterns
- Data transformation needs
- Pagination requirements
- Authentication differences

**Action:** Stop and ask questions if the operation reveals patterns not seen in existing code.

---

## Phase 5: Service Layer Implementation

### Implement Core Function
Follow the pattern from the most similar existing tool:
- Use exact parameter ordering from the pattern
- Use exact default values from the pattern
- Copy API URL construction logic
- Copy request structure
- Copy error handling approach

### For Async Operations
- Copy polling loop exactly from existing async tool
- Do not modify retry defaults without approval
- Do not modify polling interval without approval
- Standard defaults: `maxRetries = 5`, `pollIntervalMs = 2000`

### For Read-Only Operations
- Do not add retry parameters
- Do not add polling logic
- Return response data directly
- Follow query/filter patterns if applicable

---

## Phase 6: MCP Tool Registration

### Register Tool Schema
- Define tool name following project naming conventions (snake_case)
- Write clear description (see Phase 9.5)
- Define input schema matching function parameters
- Mark required vs optional parameters correctly
- Add retry parameters only for async operations

---

## Phase 7: Request Handler Implementation

### Implement Handler
- Add case for the new tool
- Destructure parameters with correct defaults
- Obtain authentication using existing pattern
- Call service function with parameters in correct order
- Return response in standard format
- Copy error handling exactly from similar handler

---

## Phase 8: Compliance Review

### Verify Against Project Standards
- Parameter order matches conventions
- Default values match conventions
- Error handling matches existing patterns
- Response structure matches existing patterns
- No hardcoded values that should be configurable
- Naming conventions followed

---

## Phase 9: Tool Description Review for AI Agent Clarity

### Ensure Tool Description is Actionable

AI agents need clear descriptions to use tools correctly. Follow this structure:

```
[Action verb] [what it does]. [Additional context].
PREREQUISITE: [condition] (use [tool_name]).
REQUIRED: [param1] (use [tool_name] to get [param1]) + [param2].
[Special notes].
```

**Parameter descriptions should include tool references:**
```
'ID of the resource to delete (use query_resources to find resource ID)'
'Array of venue IDs (use get_venues to get venue IDs)'
```

**Checklist:**
- [ ] Description starts with clear action and purpose
- [ ] Prerequisites listed with tool references
- [ ] Required parameters explain how to obtain IDs
- [ ] Destructive operations include warnings
- [ ] Batch vs single operation scope is clear

---

## Phase 10: Summary

### Report
- Files modified and functions added
- Pattern used as basis
- Operation type and characteristics
- Any deviations from standard patterns (with justification)

### Testing Guidance
- How to manually test the new tool
- Expected input and output
- Edge cases to verify
- Suggest: `/ci-testcase` to generate automated tests

---

## Key Principles

1. **Consistency First**: Copy existing patterns exactly rather than inventing new approaches
2. **No Assumptions**: Stop and ask when unclear
3. **No Modifications**: Do not modify retry defaults, polling intervals, or error handling without approval
4. **Pattern Matching**: Find the most similar existing tool and follow its structure
5. **Question Everything**: If something seems inconsistent, investigate and ask
