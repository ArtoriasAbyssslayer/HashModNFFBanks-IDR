#!/usr/bin/env python3
"""
Enhanced Real-Time IDR Training Script - FIXED VERSION

This script provides IDR training with real-time rendering capabilities.
All indentation and syntax issues have been resolved.
"""

import sys
import os
import argparse
import time
import threading

# Add code directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to reach project root
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'code'))

def main():
    parser = argparse.ArgumentParser(description='Enhanced IDR Training with Real-Time Rendering')
    parser.add_argument('--conf', type=str, required=True, help='Configuration file path')
    parser.add_argument('--scan_id', type=int, help='Scan ID override')
    parser.add_argument('--nepochs', type=int, help='Number of epochs override')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU device ID')
    parser.add_argument('--realtime', action='store_true', default=True, help='Enable real-time rendering (default: True)')
    parser.add_argument('--headless', action='store_true', help='Run without GUI')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Enhanced Real-Time IDR Training")
    print("="*70)
    
    # Check system status
    try:
        from training.idr_train import IDRTrainRunner
        idr_training = True
        print("[OK] IDR training module loaded")
    except ImportError as e:
        print(f"[ERROR] IDR training not available: {e}")
        idr_training = False
        return
    
    # Check real-time components
    real_time_available = False
    patch_idr_trainer = None
    try:
        import realtime
        from realtime.patches import patch_idr_trainer
        real_time_available = True
        print("[OK] Real-time rendering package loaded")
    except ImportError as e:
        print(f"[WARNING] Real-time package not available: {e}")
    
    # GPU Setup
    gpu = args.gpu
    if gpu == 'auto':
        try:
            import GPUtil
            deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5)
            gpu = deviceIDs[0] if len(deviceIDs) > 0 else 0
        except:
            gpu = 0
    
    print(f"\n[INFO] Using GPU: {gpu}")
    
    # Create trainer with all required parameters
    try:
        trainer = IDRTrainRunner(conf=args.conf, 
                               batch_size=1,
                               nepochs=args.nepochs,
                               expname='',  # Optional suffix
                               gpu_index=gpu,
                               exps_folder_name='exps',
                               is_continue=False,
                               timestamp='latest',
                               checkpoint='latest',
                               scan_id=args.scan_id,
                               train_cameras=False,
                               validation_slope_print=False,
                               calc_image_similarity=False)
        print("[OK] Trainer initialized")
    except Exception as e:
        print(f"[ERROR] Failed to create trainer: {e}")
        print("\n[TIPS] Common issues:")
        print("- Check that your conf file has 'batch_size' in train section")
        print("- Verify all required paths exist in configuration")
        print("- Ensure dataset directory is accessible")
        
        # Try to provide more specific help
        if 'batch_size' in str(e):
            print("\n[Fix] Add 'batch_size = 1' to the train section of your conf file")
        
        # Try a different approach - use original working method
        print("\n[INFO] Trying original training method...")
        try:
            # Fall back to original exp_runner approach
            import subprocess
            import sys
            
            # Change to code directory first
            original_cwd = os.getcwd()
            code_dir = os.path.join(project_root, 'code')
            os.chdir(code_dir)
            
            # Convert conf path to be relative to code directory
            conf_path = args.conf
            if not os.path.isabs(conf_path):
                conf_path = os.path.relpath(os.path.join(project_root, conf_path), code_dir)
            
            cmd = [
                sys.executable, 'training/exp_runner.py',
                '--conf', conf_path,
                '--scan_id', str(args.scan_id),
                '--nepoch', str(args.nepochs),
                '--gpu', str(gpu),
                '--batch_size', '1'
            ]
            
            print(f"[INFO] Running: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            
            # Restore original directory
            os.chdir(original_cwd)
            return
            
        except Exception as e2:
            print(f"[ERROR] Both methods failed. Original error: {e}")
            print(f"[ERROR] Fallback error: {e2}")
            return
    
    # Apply real-time patch if requested (or by default)
    if (args.realtime or not args.headless) and real_time_available and patch_idr_trainer is not None:
        try:
            trainer = patch_idr_trainer(trainer, enable_realtime=True)
            print("[OK] Real-time patch applied")
        except Exception as e:
            print(f"[WARNING] Real-time patch failed: {e}")
            real_time_available = False
    
    # Run training
    if args.headless or not real_time_available:
        print("\n[INFO] Starting headless training...")
        trainer.run()
    
    elif args.realtime and real_time_available:
        print("\n[INFO] Starting GUI mode with real-time rendering...")
        
        try:
            # Simple GUI implementation
            print("Real-time GUI running...")
            print("Press Ctrl+C to stop training")
            
            # Start training in background
            training_thread = threading.Thread(target=trainer.run, daemon=True)
            training_thread.start()
            
            # Wait for user interaction
            try:
                while training_thread.is_alive():
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n[INFO] Training interrupted by user")
                training_thread.join(timeout=5.0)
            
        except Exception as e:
            print(f"[WARNING] GUI mode failed: {e}")
            print("[INFO] Falling back to headless mode")
            trainer.run()
    
    else:
        print("[INFO] Running standard training (no real-time)")
        trainer.run()

if __name__ == "__main__":
    main()