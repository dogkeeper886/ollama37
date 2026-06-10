# Create User Story

Create a user story file from user input.

```
{{input}}

## PURPOSE

Capture the user's **need** as a story file saved to `docs/stories/`. A story is a
**goal, not a spec** — it states what the user needs and why, and leaves the *how*
open. Implementation detail (affected files, APIs, design) is worked out later on the
GitHub issue, not here. See `docs/stories/README.md`.

---

## AGENT WORKFLOW

### Step 1: Determine Story ID

Check `docs/stories/` for existing story files. Generate the next sequential ID:
- Format: `STORY-XXX` (e.g., STORY-001, STORY-002)
- If no stories exist, start with STORY-001

### Step 2: Clarify the Need

If the user input is vague, ask questions that sharpen the **need** — not the
implementation:
- Who is the user, and what are they trying to achieve?
- Why does it matter — what's the benefit or the problem behind the request?
- What would success look like from their point of view?

Do **not** ask about affected files, edge cases, or design here — that gets worked
out on the issue. If the need is clear enough, proceed directly.

### Step 3: Write Story File

Create `docs/stories/STORY-XXX.md` with this template:

```markdown
# STORY-XXX: [Title]

## User Story

As a [role],
I want to [action],
So that [benefit].

## The Need

[The problem behind the request — what the user is trying to achieve and why, in
their terms.]

## Success Looks Like

[The outcome that means this is done, described as observable user-facing results —
not implementation steps.]

## Open Questions

[What still has to be figured out — resolved later on the GitHub issue via research,
proof of concept, or clarification. The "how" goes here, not above.]

## Status

- Created: [date]
- Issues: none
```

### Step 4: Confirm

Show the user the created story and suggest next steps:
- `/dw-review-story STORY-XXX` to check it's complete and still a goal, not a spec
- `/dw-plan STORY-XXX` to research the approach and write the plan issue (the agreed
  *how*), reviewed before it's broken into tasks
- Trivial / single-task work can skip the plan — `/dw-tasks STORY-XXX` straight away
- `/qw-plan` to plan what to test (if QA-related)

---

## OUTPUT

The path to the created story file.
```
