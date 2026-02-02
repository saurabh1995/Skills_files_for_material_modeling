# Quick Start: Using CLAUDE.md to Prevent Mistakes

## What Happened

The code generator made a **critical mistake** in the `_plastic_update` function:
- ❌ Computed `p_dot` from TRIAL state
- ❌ Used it immediately in same step  
- ❌ Then recalculated `fy` from UPDATED state
- **Result:** `p_dot` and `fy` were inconsistent → accumulating errors

## The Fix

✅ Use rates from PREVIOUS step → Update ALL state → Compute NEW rates from updated state

## What I Created for You

### 1. **CLAUDE.md** - The Mistake Prevention File
Contains:
- ❌ Wrong patterns (what NOT to do)
- ✅ Correct patterns (what TO do)
- 📋 Verification checklist
- Summary of golden rules

### 2. **HOW_CLAUDE_MD_WORKS_COMPLETE_GUIDE.md** - Full Documentation
Explains:
- What CLAUDE.md is
- How it works
- Where to put it
- How to update skills with it
- When to add new patterns

### 3. **analysis_of_mistakes.md** - Detailed Comparison
Shows:
- Line-by-line comparison of wrong vs correct code
- Why the mistake happened
- Impact of the error

## How to Use CLAUDE.md Right Now

### Quick Method (5 minutes)

**If you already have the explicit-integration-scheme skill:**

1. The skill I packaged earlier already includes CLAUDE.md
2. Just upload `explicit-integration-scheme.skill` to claude.ai
3. When you use it, Claude will automatically avoid the mistake

**If you're creating a new skill or updating an existing one:**

1. Copy CLAUDE.md to your skill directory:
   ```bash
   cp CLAUDE.md /path/to/your-skill/
   ```

2. Add this to the top of your SKILL.md:
   ```markdown
   ## ⚠️ CRITICAL: Read This First
   **BEFORE generating code, ALWAYS read CLAUDE.md**
   ```

3. Repackage:
   ```bash
   python3 package_skill.py /path/to/your-skill /output
   ```

Done! Your skill now prevents the mistake.

## The Three Critical Errors Documented

### Error 1: Rate Sequencing ⭐ MOST IMPORTANT
```python
# ❌ WRONG
p_dot = compute_from_trial_state()
use_p_dot_immediately()

# ✅ CORRECT  
p_dot = from_previous_step
update_all_state()
p_dot_new = compute_from_updated_state()
```

### Error 2: Incomplete Operand
```python
# ❌ WRONG
operand = (..., p_dot, eps_I_dot)  # Missing rates

# ✅ CORRECT
operand = (..., p_dot, eps_I_dot, D_dot, R_dot, X_dot)  # ALL rates
```

### Error 3: Flow Direction Reference
```python
# ❌ WRONG
flow_dir = from_trial_state()

# ✅ CORRECT
flow_dir = from_updated_state()
```

## The Golden Rule

> **In forward Euler, rates are computed at the END of each step from UPDATED state, then stored and used at the BEGINNING of the NEXT step.**

## Verification Checklist

Before delivering integration scheme code, verify:
- [ ] Previous step rates used for current increments
- [ ] ALL state updated BEFORE computing new rates
- [ ] New rates computed from UPDATED state (not trial)
- [ ] ALL rates in operand (p_dot, eps_I_dot, D_dot, R_dot, X_dot)
- [ ] Flow direction from updated stress
- [ ] fy and p_dot from same state

## Files You Have

| File | Purpose | Use It To |
|------|---------|-----------|
| **CLAUDE.md** | Mistake prevention | Add to skills to prevent errors |
| **HOW_CLAUDE_MD_WORKS_COMPLETE_GUIDE.md** | Full documentation | Understand the system |
| **analysis_of_mistakes.md** | Detailed comparison | See exactly what was wrong |

## Next Steps

### Immediate (Do Now)
1. Read CLAUDE.md to understand the patterns
2. Use it as reference when reviewing generated code
3. Add it to your skills

### Short Term (This Week)
1. Test with the corrected prompt (generate viscoplastic model)
2. Verify Claude doesn't make the mistake
3. Update any existing skills

### Long Term (Ongoing)
1. When you find new mistakes, add to CLAUDE.md
2. Build up pattern library over time
3. Share with team/colleagues

## Key Insight

The mistake wasn't obvious! It's a subtle **temporal inconsistency** in the forward Euler integration. CLAUDE.md ensures:
- Claude knows about this subtle pattern
- Claude checks for it while generating
- The mistake never happens again

## Questions?

See **HOW_CLAUDE_MD_WORKS_COMPLETE_GUIDE.md** for:
- Detailed examples
- Troubleshooting
- Advanced usage
- Best practices

---

**Bottom line:** Add CLAUDE.md to your skills → Claude won't make the same mistakes again.
