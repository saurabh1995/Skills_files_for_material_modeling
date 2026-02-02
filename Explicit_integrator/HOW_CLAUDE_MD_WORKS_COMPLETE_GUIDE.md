# Complete Guide: How CLAUDE.md Works and How to Use It

## What is CLAUDE.md?

CLAUDE.md is a **mistake prevention document** that gets included in skills to prevent Claude from repeating known errors. It works like a checklist and pattern library that Claude consults when generating code.

Think of it as:
- ❌ A "do NOT do this" reference
- ✅ A "do THIS instead" pattern library  
- 📋 A quality checklist
- 🧠 Accumulated knowledge from mistakes

---

## How It Works

### The Skill Loading Sequence

When you use a skill that includes CLAUDE.md:

```
1. User makes a request
   ↓
2. Skill triggers (based on description match)
   ↓
3. Claude reads SKILL.md (main instructions)
   ↓
4. Claude reads CLAUDE.md (mistake prevention) ← THIS IS KEY
   ↓
5. Claude reads references/*.md as needed
   ↓
6. Claude generates code following ALL guidance
```

### Why It Prevents Mistakes

**Without CLAUDE.md:**
```
User: "Generate viscoplastic model with damage"
Claude: Reads SKILL.md → Generates code → ❌ Makes rate sequencing mistake
```

**With CLAUDE.md:**
```
User: "Generate viscoplastic model with damage"
Claude: Reads SKILL.md → Reads CLAUDE.md (sees pattern to avoid)
→ Generates code → ✅ Uses correct forward Euler pattern
```

---

## Where to Put CLAUDE.md

### Option 1: In the Skill Root (RECOMMENDED)

```
your-skill/
├── SKILL.md
├── CLAUDE.md              ← Put it here
└── references/
    ├── code-template.md
    └── ...
```

**Advantages:**
- Always loaded when skill is used
- Part of the skill package
- Can't be forgotten

**How to implement:**
```bash
# Add CLAUDE.md to your skill directory
cp CLAUDE.md /path/to/your-skill/

# Reference it in SKILL.md
echo "## ⚠️ CRITICAL: Read This First" >> SKILL.md
echo "**BEFORE generating code, ALWAYS read CLAUDE.md**" >> SKILL.md

# Repackage the skill
python3 /mnt/skills/examples/skill-creator/scripts/package_skill.py \
    /path/to/your-skill \
    /output/directory
```

### Option 2: As a Reference File

```
your-skill/
├── SKILL.md
└── references/
    ├── code-template.md
    ├── CLAUDE.md          ← Or here
    └── ...
```

**Advantages:**
- Organized with other references
- Can be selectively loaded

**How to implement:**
Add to SKILL.md:
```markdown
## Critical Implementation Notes

**ALWAYS read `references/CLAUDE.md` before generating code** to avoid common mistakes.
```

### Option 3: User Preferences (Global)

For claude.ai users, add key patterns to user preferences:

1. Click profile → Settings → User Preferences
2. Add:
```
When generating explicit integration schemes:
- Always use rates from PREVIOUS step for increments
- Update ALL state before computing new rates
- Compute new rates from UPDATED state, not trial
- See explicit-integration-scheme skill CLAUDE.md for details
```

---

## How to Update a Skill with CLAUDE.md

### Step-by-Step Process

**Step 1: Copy CLAUDE.md to skill directory**
```bash
cp /path/to/CLAUDE.md /path/to/your-skill/
```

**Step 2: Reference it in SKILL.md**

Add at the beginning of SKILL.md:
```markdown
---
name: your-skill-name
description: Your skill description
---

# Your Skill Name

## ⚠️ CRITICAL: Read This First

**BEFORE generating any code, ALWAYS read `CLAUDE.md`** for critical implementation patterns.

This file documents common mistakes and corrections, including:
- Forward Euler rate computation sequencing
- Proper operand structure with previous-step rates
- State update ordering
- Flow direction consistency

## Your Existing Content
...
```

**Step 3: Repackage the skill**
```bash
python3 /mnt/skills/examples/skill-creator/scripts/package_skill.py \
    /path/to/your-skill \
    /path/to/output
```

This creates `your-skill.skill` with CLAUDE.md included.

**Step 4: Verify it was included**
```bash
# .skill files are ZIP files
unzip -l your-skill.skill | grep CLAUDE.md
```

Should show:
```
  1234  2024-02-02 12:34   your-skill/CLAUDE.md
```

---

## Structure of an Effective CLAUDE.md

### Template Structure

```markdown
# Common Mistakes and Corrections for [Topic]

## ❌ CRITICAL ERROR 1: [Error Name]

### The Mistake
[Clear description of what NOT to do]

### ❌ Wrong Pattern (DO NOT USE):
```python
# Code example showing the WRONG way
```

**Why this is wrong:**
- [Bullet points explaining the problem]
- [Consequences of this mistake]

### ✅ Correct Pattern (ALWAYS USE):
```python
# Code example showing the RIGHT way
```

**Why this is correct:**
- [Bullet points explaining why it works]
- [Benefits of this approach]

---

## ❌ CRITICAL ERROR 2: [Another Error]
[Same structure...]

---

## [Key Principles Section]
[Fundamental rules and patterns]

---

## Checklist
Before delivering code, verify:
- [ ] Requirement 1
- [ ] Requirement 2
...

---

## Summary
**The Golden Rule:**
> [Core principle stated clearly]

**Never:**
- [List of things to avoid]

**Always:**
- [List of things to do]
```

### Key Elements

1. **Visual Markers**: Use ❌ for wrong, ✅ for correct
2. **Code Examples**: Show exact wrong vs right patterns
3. **Explanations**: Why it's wrong, why it's right
4. **Context**: Reference equations, algorithms, papers
5. **Checklist**: Actionable verification steps
6. **Summary**: Concise golden rule

---

## Real-World Example: Your Case

### The Mistake You Found

```python
# ❌ WRONG: Computing p_dot from trial state
def _plastic_update(operand):
    x = fy / params.K  # fy from trial
    p_dot = jnp.power(bracket, params.m)
    dp = p_dot * dt  # Use immediately
    # ... update state ...
    fy_new = ...  # From updated state
    # p_dot and fy_new now INCONSISTENT
```

### How CLAUDE.md Prevents It

1. Documents the exact wrong pattern with code
2. Explains why it's wrong (temporal inconsistency)
3. Shows the correct pattern
4. Includes in checklist: "fy and p_dot from same state?"

### How Claude Uses It

```
Step 1: User requests integration scheme
Step 2: Skill triggers
Step 3: Claude reads SKILL.md: "Generate integration scheme"
Step 4: Claude reads CLAUDE.md: "⚠️ DON'T compute p_dot from trial!"
Step 5: Claude checks pattern while generating
Step 6: Claude generates code with correct pattern
Step 7: Claude verifies against checklist before delivering
```

---

## Benefits of Using CLAUDE.md

### 1. Prevents Repeated Mistakes
Once documented, Claude won't make the same mistake again.

### 2. Accumulates Knowledge
Build up domain expertise over time by adding new patterns.

### 3. Portable
The skill carries its own corrections - works anywhere.

### 4. Specific
More targeted than general instructions or system prompts.

### 5. Examples-Based
Shows exact code patterns, not vague descriptions.

### 6. Evolving
Easy to update as you discover new patterns.

### 7. Self-Documenting
Future users see what mistakes to avoid.

---

## When to Update CLAUDE.md

Add to CLAUDE.md when you discover:

### ❌ Mistakes
- Claude generated wrong code
- Pattern that causes subtle bugs
- Common error in complex workflows

### ✅ Best Practices  
- Particularly elegant solution
- Optimization that should be standard
- Pattern that works well

### 🔧 Subtle Details
- Easy-to-miss implementation detail
- Counter-intuitive requirement
- Edge case handling

### 🚨 Critical Errors
- Mistakes that cause silent failures
- Violations of mathematical correctness
- Numerical stability issues

---

## Maintenance Workflow

### When You Find a Mistake

1. **Document it immediately**
   ```markdown
   ## ❌ ERROR: [Name]
   ### The Mistake
   [What went wrong]
   
   ### Wrong Pattern
   ```code```
   
   ### Correct Pattern
   ```code```
   ```

2. **Add to CLAUDE.md**
   ```bash
   # Edit CLAUDE.md
   vim /path/to/skill/CLAUDE.md
   ```

3. **Repackage skill**
   ```bash
   python3 package_skill.py /path/to/skill /output
   ```

4. **Test with Claude**
   - Upload new .skill file
   - Make same request that caused error
   - Verify correct code is generated

### Best Practices for Updates

- **Be specific**: Show exact code, not descriptions
- **Explain why**: Include reasoning
- **Keep organized**: Group related mistakes
- **Update checklist**: Add verification items
- **Test it**: Verify it prevents the mistake

---

## Integration with Skills

### For the Explicit Integration Scheme Skill

Your skill now has:

```
explicit-integration-scheme/
├── SKILL.md
│   └── References CLAUDE.md at top
├── CLAUDE.md
│   ├── Error 1: Rate sequencing
│   ├── Error 2: Operand structure
│   ├── Error 3: Flow direction
│   ├── Forward Euler pattern
│   ├── Checklist
│   └── Summary
└── references/
    ├── code-template.md
    ├── equation-mapping.md
    ├── example-models.md
    └── plane-stress-extension.md
```

### Loading Sequence

```
User: "Generate viscoplastic model with damage and kinematic hardening"
  ↓
Skill triggers: explicit-integration-scheme
  ↓
Claude reads: SKILL.md
  → Sees: "⚠️ CRITICAL: Read CLAUDE.md first"
  ↓
Claude reads: CLAUDE.md
  → ❌ Error 1: Don't compute p_dot from trial
  → ✅ Correct: Use previous step rates
  → ❌ Error 2: Include all rates in operand
  → ✅ Correct: Add D_dot, R_dot, X_dot
  → ❌ Error 3: Don't use trial for flow direction
  → ✅ Correct: Use updated sig_eff_new
  → 📋 Checklist: [all items]
  ↓
Claude generates code:
  → Uses previous step rates ✅
  → Updates all state first ✅
  → Computes new rates from updated state ✅
  → Includes all rates in operand ✅
  → Uses updated state for flow direction ✅
  ↓
Claude verifies against checklist ✅
  ↓
Delivers correct code
```

---

## Advanced Usage

### Multiple CLAUDE.md Files

For complex skills, you can have:

```
your-skill/
├── SKILL.md
├── CLAUDE.md  ← General mistakes
└── references/
    ├── CLAUDE-numerical.md  ← Numerical issues
    ├── CLAUDE-jax.md  ← JAX-specific patterns
    └── CLAUDE-plasticity.md  ← Plasticity-specific
```

Reference them in SKILL.md:
```markdown
## Critical Reading

1. **CLAUDE.md** - General mistake patterns
2. **references/CLAUDE-numerical.md** - Numerical stability
3. **references/CLAUDE-jax.md** - JAX compatibility
```

### Skill-Specific vs. Global

**Skill-Specific CLAUDE.md:**
- Lives in the skill
- Domain-specific patterns
- Technical implementation details

**Global User Preferences:**
- Applies to all conversations
- High-level principles
- Personal workflow preferences

Use both for maximum effectiveness.

---

## Troubleshooting

### "Claude still makes the mistake"

**Check:**
1. Is CLAUDE.md in the skill package?
   ```bash
   unzip -l your-skill.skill | grep CLAUDE.md
   ```

2. Is it referenced in SKILL.md?
   ```bash
   grep "CLAUDE.md" your-skill/SKILL.md
   ```

3. Is the pattern clearly documented?
   - Clear code examples?
   - Explanation of why it's wrong?
   - Correct alternative shown?

4. Did you repackage the skill after adding CLAUDE.md?

### "CLAUDE.md is too long"

**Solutions:**
1. Split into multiple files (CLAUDE-topic.md)
2. Keep only critical errors in main CLAUDE.md
3. Move detailed explanations to references/
4. Use concise code examples

### "Unclear which pattern to follow"

**Fix:**
- Make the ✅ correct pattern very clear
- Add "ALWAYS USE" label
- Put correct pattern after wrong pattern
- Include in checklist

---

## Summary

### What CLAUDE.md Is
- Mistake prevention document
- Pattern library
- Quality checklist
- Accumulated domain knowledge

### How It Works
- Loaded when skill triggers
- Consulted before/during code generation
- Verified against before delivery

### Where It Goes
- Recommended: Skill root directory
- Alternative: references/ subdirectory
- Also: User preferences for global patterns

### When to Update
- When Claude makes a mistake
- When you discover a best practice
- When you find a subtle requirement
- When you encounter an edge case

### The Result
- Prevents repeated mistakes
- Improves code quality
- Builds domain expertise
- Makes skills more robust

**Use CLAUDE.md to make your skills smarter over time!**
