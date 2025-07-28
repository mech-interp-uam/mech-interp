#!/usr/bin/env python3
"""
Script to run multiple SAE training experiments with different configurations.
"""

import subprocess
import sys
import time

def run_experiment(exp_factor, sparsity_coeff, use_d_model_std=False, total_steps=100000, sparsity_warmup_full=True, max_lr=7e-5):
    """Run a single experiment with given parameters."""
    
    # Generate run name
    norm_type = "d_model_std" if use_d_model_std else "d_sae_std"
    warmup_type = "fullwarmup" if sparsity_warmup_full else "100kwarmup"
    lr_suffix = f"_lr{max_lr:.0e}" if max_lr != 7e-5 else ""
    run_name = f"exp{exp_factor}_sparse{sparsity_coeff:.4f}_{norm_type}_{warmup_type}_steps{total_steps}{lr_suffix}"
    
    print(f"\n{'='*60}")
    print(f"Starting experiment: {run_name}")
    print(f"Expansion factor: {exp_factor}")
    print(f"Sparsity coeff: {sparsity_coeff}")
    print(f"Use d_model std: {use_d_model_std}")
    print(f"Sparsity warmup full: {sparsity_warmup_full}")
    print(f"Total steps: {total_steps}")
    print(f"Max learning rate: {max_lr}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        "python", "sae-plan-de-investigacion.py",
        "--run-name", run_name,
        "--exp-factor", str(exp_factor),
        "--sparsity-coeff", str(sparsity_coeff),
        "--total-steps", str(total_steps),
        "--max-lr", str(max_lr)
    ]
    
    if use_d_model_std:
        cmd.append("--use-d-model-std")
    
    if sparsity_warmup_full:
        cmd.append("--sparsity-warmup-full")
    
    # Run the experiment
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"✓ Experiment {run_name} completed successfully in {end_time - start_time:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Experiment {run_name} failed!")
        print(f"Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def main():
    """Run all experiments."""
    
    total_steps = 100000  # Fixed for all runs
    experiments = []
    
    # First run: exp_factor=16 with d_model normalization (keep only 1)
    experiments.append((16, 0.001, True, total_steps, True, 7e-5))
    
    # Main experiments: exp_factor 16, 24 with different sparsity coeffs and double lr variants
    exp_factors = [16, 24]
    sparsity_coeffs = [0.0005, 0.001, 0.0015, 0.002]
    lr_multipliers = [1.0, 2.0]  # normal lr and double lr
    
    for exp_factor in exp_factors:
        for sparsity_coeff in sparsity_coeffs:
            for lr_mult in lr_multipliers:
                # Skip the d_model_std experiment for exp_factor=16, sparsity=0.001, lr=7e-5
                # since we already added it above
                if exp_factor == 16 and sparsity_coeff == 0.001 and lr_mult == 1.0:
                    continue
                base_lr = 7e-5
                current_lr = base_lr * lr_mult
                experiments.append((exp_factor, sparsity_coeff, False, total_steps, True, current_lr))
    
    print(f"Planning to run {len(experiments)} experiments:")
    for i, (exp_factor, sparsity_coeff, use_d_model_std, steps, warmup_full, max_lr) in enumerate(experiments, 1):
        norm_type = "d_model_std" if use_d_model_std else "d_sae_std"
        warmup_type = "fullwarmup" if warmup_full else "100kwarmup"
        lr_suffix = f"_lr{max_lr:.0e}" if max_lr != 7e-5 else ""
        print(f"{i:2d}. exp{exp_factor}_sparse{sparsity_coeff:.4f}_{norm_type}_{warmup_type}_steps{steps}{lr_suffix}")
    
    print(f"\nStarting experiments...")
    
    successful = 0
    failed = 0
    
    for i, (exp_factor, sparsity_coeff, use_d_model_std, steps, warmup_full, max_lr) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]", end=" ")
        
        if run_experiment(exp_factor, sparsity_coeff, use_d_model_std, steps, warmup_full, max_lr):
            successful += 1
        else:
            failed += 1
            
        # Small delay between experiments
        time.sleep(2)
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()