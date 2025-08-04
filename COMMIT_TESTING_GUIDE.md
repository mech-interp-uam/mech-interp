# Commit-by-Commit Testing Guide for Mixed Precision SAE Implementation

## Testing Strategy Overview

This guide provides a systematic approach to test each commit in the mixed precision implementation. Each commit is ranked by risk level and includes specific testing recommendations.

**IMPORTANT:** Mixed precision training (fp32 → fp16/bf16) is expected to be unstable/broken until the final comprehensive scaling commit (`7521c21`). The commits build up the normalization infrastructure progressively:
- **Before input scaling**: Mixed precision will likely overflow/underflow
- **After input scaling**: Partial improvement but still incomplete
- **After comprehensive scaling**: Full mixed precision should work

**Base Command for Testing:**
```bash
python book/sae-plan-de-investigacion.py --early-exit <steps> --seed 42 --deterministic
```

**Risk Levels:**
- 🟢 **LOW**: Infrastructure/CLI changes, minimal model impact
- 🟡 **MEDIUM**: Model architecture changes, numerical modifications
- 🔴 **HIGH**: Core scaling logic, gradient computation, precision changes

---

## Commit Analysis (Starting from Testing/Reproducibility Base)

### BASE: `46f8789` - Add testing and reproducibility cli options
**Risk Level: 🟢 LOW**  
**Test Steps: 10**  
**Testing Command:**
```bash
python book/sae-plan-de-investigacion.py --early-exit 10 --seed 42 --deterministic --log-timing
```

**Intention:** Add infrastructure for reproducible testing with seed control, deterministic behavior, epoch timing, and compile settings.

**What to Verify:**
- Script runs with all new CLI options
- Timing logs are created
- Deterministic behavior works
- Early exit functions properly

**Risks:**
- CLI parsing errors
- Deterministic algorithms compatibility issues
- Timing infrastructure bugs

---

### `7d00eb9` - rename d_in to d_model
**Risk Level: 🟢 LOW**  
**Test Steps: 10**  
**Testing Command:**
```bash
python book/sae-plan-de-investigacion.py --early-exit 10 --seed 42 --deterministic
```

**Intention:** Refactor variable naming for clarity - rename d_in to d_model to be more descriptive and consistent with standard terminology.

**What to Verify:**
- Script runs without errors
- No functionality changes
- Variable names are consistent

**Risks:**
- Refactoring errors causing undefined variables
- Inconsistent renaming leading to bugs

---

### `be1dc49` - fix typos
**Risk Level: 🟢 LOW**  
**Test Steps: 10**  
**Testing Command:**
```bash
python book/sae-plan-de-investigacion.py --early-exit 10 --seed 42 --deterministic
```

**Intention:** Fix spelling and grammatical errors in comments and strings.

**What to Verify:**
- Script runs without errors
- No functionality changes

**Risks:**
- Accidental code changes while fixing typos

---

### `6eda098` - dissable loggin when frequency is zero
**Risk Level: 🟢 LOW**  
**Test Steps: 10**  
**Testing Command:**
```bash
python book/sae-plan-de-investigacion.py --early-exit 10 --seed 42 --deterministic
```

**Intention:** Disable logging operations when frequency is set to zero, improving performance by avoiding unnecessary computation.

**What to Verify:**
- Logging is properly disabled when frequency=0
- No performance overhead from disabled logging
- Logging still works when frequency>0

**Risks:**
- Logic errors in conditional checks
- Logging being disabled when it shouldn't be

---

### `ff3780a` - remove unecessary on logging conditionals
**Risk Level: 🟢 LOW**  
**Test Steps: 10**  
**Testing Command:**
```bash
python book/sae-plan-de-investigacion.py --early-exit 10 --seed 42 --deterministic
```

**Intention:** Clean up redundant conditional checks in logging code to improve readability and performance.

**What to Verify:**
- Logging still works correctly
- No changes in logging behavior
- Code is cleaner

**Risks:**
- Removing necessary conditionals
- Logic errors in simplified code

---

### `287633f` - remove unecessary detach() on tensorboard's writer calls
**Risk Level: 🟢 LOW**  
**Test Steps: 10**  
**Testing Command:**
```bash
python book/sae-plan-de-investigacion.py --early-exit 10 --seed 42 --deterministic
```

**Intention:** Remove unnecessary .detach() calls when logging to TensorBoard, since TensorBoard writer automatically handles tensor detachment.

**What to Verify:**
- TensorBoard logging still works
- No gradient tracking issues
- Performance improvements from fewer detach() calls

**Risks:**
- Gradient computation graph issues
- Memory leaks from undetached tensors

---

### `e0f0ebd` - cleaner loggin: single no_grad context: remove unecessary detach()s
**Risk Level: 🟢 LOW**  
**Test Steps: 10**  
**Testing Command:**
```bash
python book/sae-plan-de-investigacion.py --early-exit 10 --seed 42 --deterministic
```

**Intention:** Consolidate logging operations under a single no_grad context and remove unnecessary detach() calls for cleaner, more efficient code.

**What to Verify:**
- All logging works correctly
- Single no_grad context covers all logging operations
- No gradient tracking issues

**Risks:**
- Gradient context management errors
- Missing no_grad for some operations

---

### `713f3c2` - log params and grads
**Risk Level: 🟢 LOW**  
**Test Steps: 10**  
**Testing Command:**
```bash
python book/sae-plan-de-investigacion.py --early-exit 10 --seed 42 --deterministic
```

**Intention:** Add comprehensive logging of gradient and parameter statistics (norms, means, histograms) for monitoring training dynamics and debugging.

**What to Verify:**
- Script runs without errors
- Gradient logging appears in TensorBoard/logs
- Parameter logging works correctly
- No performance degradation from logging

**Risks:**
- Logging overhead affecting performance
- Gradient synchronization issues from logging operations
- Logging frequency causing storage issues

---

### `ff1ab1f` - use None for gradient stuff
**Risk Level: 🟢 LOW**  
**Test Steps: 10**  
**Testing Command:**
```bash
python book/sae-plan-de-investigacion.py --early-exit 10 --seed 42 --deterministic
```

**Intention:** Clean up gradient handling by using None instead of zero gradients, improving memory efficiency and gradient computation clarity.

**What to Verify:**
- Gradient computation works correctly
- No changes in training dynamics
- Memory usage improvements (if measurable)
- No gradient accumulation bugs

**Risks:**
- Gradient computation logic errors
- Unintended changes to optimizer behavior
- Memory management issues

---

### `a46bfdc` - skip long running eval at start for cleaner profiling
**Risk Level: 🟢 LOW**  
**Test Steps: 10**  
**Testing Command:**
```bash
python book/sae-plan-de-investigacion.py --early-exit 10 --seed 42 --deterministic
```

**Intention:** Skip expensive evaluation operations at the start of training to get cleaner profiling data and faster startup.

**What to Verify:**
- Training starts faster
- Profiling data is cleaner
- Evaluation still works after initial steps
- No impact on training dynamics

**Risks:**
- Breaking evaluation logic
- Missing important early evaluation data

---

### `236e480` - Use JumpReLU fused operation for step(x,threshold) * x
**Risk Level: 🟡 MEDIUM**  
**Test Steps: 64k**  
**Testing Command:**
```bash
python book/sae-plan-de-investigacion.py --early-exit 64000 --seed 42 --deterministic
```

**Intention:** Implement JumpReLU as a fused operation that combines the step function and multiplication (step(x,threshold) * x) for better performance and cleaner gradients.

**What to Verify:**
- JumpReLU produces correct forward outputs
- Gradients are computed correctly for both x and threshold
- Training dynamics remain similar to separate operations
- No numerical issues with the fused implementation

**Risks:**
- Incorrect gradient computation in custom autograd function
- Different numerical behavior from separate operations
- Performance regressions from custom function overhead
- Edge cases in gradient computation (zero gradients, boundary conditions)

---

### `177ee5d` - Fix Step function to use torch.where instead of bool.to(dtype)
**Risk Level: 🟡 MEDIUM**  
**Test Steps: 64k**  
**Testing Command:**
```bash
python book/sae-plan-de-investigacion.py --early-exit 64000 --seed 42 --deterministic
```

**Intention:** Replace `bool_tensor.to(dtype) * thing` pattern with cleaner `torch.where` implementation for better performance and code clarity.

**What to Verify:**
- Step function gradients are computed correctly
- No numerical differences in threshold updates
- Training proceeds normally without NaN/inf values
- Sparse activation patterns remain consistent

**Risks:**
- Different gradient computation behavior
- Numerical precision differences
- Performance regression from torch.where vs. boolean multiplication

---

### `f2b97e4` - Adam reformulation as RMSprop with a cli toggle
**Risk Level: 🟡 MEDIUM**  
**Test Steps: 64k**  
**Testing Command:**
```bash
# Test Adam
python book/sae-plan-de-investigacion.py --early-exit 64000 --optimizer adam --seed 42
# Test RMSprop
python book/sae-plan-de-investigacion.py --early-exit 64000 --optimizer rmsprop --seed 42
```

**Intention:** Add RMSprop option that mathematically reproduces Adam with beta1=0 through bias correction scaling. Provides optimizer flexibility while maintaining equivalent dynamics.

**What to Verify:**
- Both optimizers produce similar loss curves
- RMSprop bias correction works correctly
- Learning rate scheduling applies to both optimizers
- No numerical instabilities with either optimizer

**Risks:**
- Incorrect bias correction formula
- Learning rate interaction bugs
- Optimizer state management issues

---

### `2c4ee44` - Add dtype and lower-dtype CLI arguments with validation
**Risk Level: 🟢 LOW**  
**Test Steps: 10**  
**Testing Command:**
```bash
python book/sae-plan-de-investigacion.py --early-exit 10 --dtype fp32 --lower-dtype fp32
python book/sae-plan-de-investigacion.py --early-exit 10 --dtype fp32 --lower-dtype bf16
# Test invalid combination (should error)
python book/sae-plan-de-investigacion.py --dtype fp16 --lower-dtype bf16 || echo "Expected error occurred"
```

**Intention:** Establish proper CLI foundation for mixed precision with explicit dtype control and validation of supported combinations.

**What to Verify:**
- Valid combinations work
- Invalid combinations raise clear errors
- Dtype variables are set correctly
- No model behavior changes with fp32/fp32

**Risks:**
- CLI validation logic errors
- Missing edge cases in combination validation
- Incorrect dtype mapping

---

### `7d091e3` - Fix dtype validation to support only valid combinations
**Risk Level: 🟢 LOW**  
**Test Steps: 10**  
**Testing Command:**
```bash
# Test all valid combinations
python book/sae-plan-de-investigacion.py --early-exit 10 --dtype fp32 --lower-dtype fp32
python book/sae-plan-de-investigacion.py --early-exit 10 --dtype fp32 --lower-dtype bf16  
python book/sae-plan-de-investigacion.py --early-exit 10 --dtype fp32 --lower-dtype fp16
python book/sae-plan-de-investigacion.py --early-exit 10 --dtype fp16 --lower-dtype fp16
python book/sae-plan-de-investigacion.py --early-exit 10 --dtype bf16 --lower-dtype bf16
```

**Intention:** Refine validation to only allow tested, supported dtype combinations that make mathematical and hardware sense.

**What to Verify:**
- All listed combinations work
- Unsupported combinations fail with clear messages
- bf16 is properly added as dtype choice

**Risks:**
- Overly restrictive validation
- Missing valid combinations
- Incorrect error messages

---

### `621d36d` - Add autocast support for mixed precision combinations
**Risk Level: 🟡 MEDIUM**  
**Test Steps: 64k**  
**Testing Command:**
```bash
# Test same precision first (safer)
python book/sae-plan-de-investigacion.py --early-exit 64000 --dtype fp32 --lower-dtype fp32 --seed 42
python book/sae-plan-de-investigacion.py --early-exit 64000 --dtype fp16 --lower-dtype fp16 --seed 42

# NOTE: Mixed precision may be unstable until scaling is implemented
# These may fail with overflow/underflow - that's expected:
python book/sae-plan-de-investigacion.py --early-exit 1000 --dtype fp32 --lower-dtype bf16 --seed 42
python book/sae-plan-de-investigacion.py --early-exit 1000 --dtype fp32 --lower-dtype fp16 --seed 42
```

**Intention:** Implement autocast for mixed precision training, using PyTorch's automatic mixed precision for (fp32,bf16/fp16) combinations while using nullcontext for same-precision scenarios.

**What to Verify:**
- Same precision modes work identically to before
- Autocast context is set up correctly (check no crashes)
- Parameter creation and conversion work correctly
- **EXPECTED**: Mixed precision may fail due to lack of normalization

**Risks:**
- Autocast context setup errors
- Parameter dtype conversion issues
- **EXPECTED**: Numerical instabilities with mixed precision (overflow/underflow)

---

### `ac0ab4d` - Add .float() conversion for reconstruction error computation
**Risk Level: 🟡 MEDIUM**  
**Test Steps: 64k**  
**Testing Command:**
```bash
# Test with fp32 first
python book/sae-plan-de-investigacion.py --early-exit 64000 --dtype fp32 --lower-dtype fp32 --seed 42

# Mixed precision still expected to be problematic (no scaling yet):
python book/sae-plan-de-investigacion.py --early-exit 1000 --dtype fp32 --lower-dtype fp16 --seed 42
```

**Intention:** Prevent numerical instability in reconstruction loss computation by ensuring error calculation happens in fp32, critical for mixed precision training.

**What to Verify:**
- No NaN/inf values in reconstruction loss with fp32
- .float() conversions work correctly
- **EXPECTED**: Mixed precision may still overflow (no input scaling yet)

**Risks:**
- Performance overhead from conversions
- **EXPECTED**: Still insufficient for mixed precision without scaling

---

### `a630099` - Add input scaling with proper interpretable loss handling
**Risk Level: 🔴 HIGH**  
**Test Steps: 64k**  
**Testing Command:**
```bash
# Test fp32 first (should work)
python book/sae-plan-de-investigacion.py --early-exit 64000 --dtype fp32 --lower-dtype fp32 --seed 42 --sparsity-coeff 0.001

# Mixed precision may work better now but still incomplete (no weight scaling):
python book/sae-plan-de-investigacion.py --early-exit 10000 --dtype fp32 --lower-dtype fp16 --seed 42
```

**Intention:** Implement input scaling to normalize input variance to 1.0 for optimal precision utilization, with proper threshold/bandwidth scaling and interpretable loss calculation that undoes scaling effects.

**What to Verify:**
- Training curves look similar to unscaled version
- Interpretable loss values match expected ranges (0.1-0.4 typical)
- Threshold and bandwidth scaling preserves training behavior
- Step function gradients remain reasonable
- No sudden sparsity changes due to threshold scaling

**Critical Risks:**
- Incorrect threshold/bandwidth scaling breaking sparse activation
- Loss interpretation errors masking real problems
- Input scaling affecting convergence properties
- Step function numerical behavior changes

**Success Criteria:**
- L0 loss (sparsity) should be in reasonable range (1000-5000 for d_sae=49152)
- Reconstruction loss should be 0.1-0.4 range
- Training should converge normally

---

### `7521c21` - Add comprehensive scaling factors with simplified Adam trick
**Risk Level: 🔴 HIGH**  
**Test Steps: 64k**  
**Testing Command:**
```bash
python book/sae-plan-de-investigacion.py --early-exit 64000 --dtype fp32 --lower-dtype fp16 --adam-trick --seed 42
```

**Intention:** Complete mixed precision implementation with weight scaling, activation scaling (c), and corrected gradient scaling. Uses simplified Adam trick with fixed gradient scaler that accounts for batch size and original error variance.

**What to Verify:**
- Training converges with mixed precision
- Gradient magnitudes remain reasonable (check grad norms in logs)
- No overflow/underflow in any precision format
- Interpretable loss matches expected values
- Adam trick prevents gradient de-scaling issues
- All scaling factors work together correctly

**Critical Risks:**
- Gradient scaling formula errors causing divergence
- Interaction between all scaling factors
- Numerical instability in lower precision
- Weight scaling affecting convergence
- Adam trick implementation bugs
- Bandwidth scaling with multiple factors

**Success Criteria:**
- Stable training with mixed precision
- Similar convergence to fp32 baseline
- Reasonable gradient norms (1e-6 to 1e-2 range)
- No NaN/inf values anywhere
- Interpretable losses in expected ranges

**Debugging Guidelines if Issues Found:**
1. **Gradient explosion**: Check grad_scaler calculation, verify batch size is correct
2. **Gradient vanishing**: Verify total_error_linear_scaling computation
3. **Training divergence**: Test with --adam-trick flag disabled (should error)
4. **Sparsity issues**: Check bandwidth scaling includes both x and w factors
5. **Loss scale issues**: Verify interpretable loss division includes all factors

---

## Testing Protocol

### For Each Commit:
1. **Checkout commit**: `git checkout <commit_hash>`
2. **Run test command** with specified early exit steps
3. **Monitor for**:
   - Script completion without errors
   - Reasonable loss values
   - No NaN/inf in outputs
   - Expected sparsity levels
   - Gradient norms in logs

### Success Criteria by Risk Level:
- **🟢 LOW**: Script runs without errors, features work as intended
- **🟡 MEDIUM**: Training proceeds normally, numerical values reasonable
- **🔴 HIGH**: Training converges, matches baseline performance, numerical stability

### Expected Training Dynamics:

**Interpretable Reconstruction Loss Pattern:**
- **Start**: ~0.3 ± 0.05 (initial reconstruction quality)
- **Early training**: Decreases to ~0.02 (learning phase)
- **Mid/late training**: Increases to ~0.2 and stabilizes (sparsity-reconstruction tradeoff)
- **🚨 RED FLAG**: Reconstruction > 0.27 at ANY point indicates serious problems

**L0 Loss (Sparsity) Pattern:**
- **Start**: ~d_sae/2 (≈24,576 for default d_sae=49,152)
- **First tens of steps**: Rapid increase to nearly d_sae (fast sparsity changes expected)
- **Rest of training**: Steady decrease throughout training

**Sparsity Change Characteristics:**
- **Fast changes expected**: First steps show dramatic sparsity increases
- **Then steady decline**: Gradual reduction in active features over training
- **This is normal behavior**: Don't treat rapid early sparsity changes as red flags

### Red Flags Requiring Investigation:
- NaN or inf values in any logged metric
- **Reconstruction loss > 0.27 at any point** (critical threshold)
- Reconstruction loss < 0.01 (suspicious underflow)
- L0 loss failing to follow expected pattern (start ~d_sae/2 → peak ~d_sae → decline)
- Gradient norms outside 1e-8 to 1e2 range
- Training divergence or failure to decrease loss after initial phase
- Reconstruction loss not stabilizing around 0.2 in later training

### Fix Guidelines:
- Always understand the original intention before modifying
- Test fixes with the same early exit steps
- Verify fixes don't break subsequent commits
- Maintain mathematical correctness over performance
- Document any changes made during testing
