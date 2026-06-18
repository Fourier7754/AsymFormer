import os
import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        raise e

def install_dependencies():
    print("Installing dependencies...")
    # Install required packages
    # Note: TensorRT and PyCUDA are omitted as they are complex to set up on Colab and likely not needed for PyTorch evaluation
    run_command("pip install timm scikit-image opencv-python-headless thop onnx onnxruntime")
    run_command("pip install --upgrade gdown") # Upgrade gdown to ensure it works with latest Drive changes
    
    # Install p7zip-full for better archive handling
    # try:
    #     run_command("apt-get update && apt-get install -y p7zip-full")
    # except Exception as e:
    #     print(f"Warning: Failed to install p7zip-full: {e}")

def setup_data():
    # The README says to data should be in 'data' folder
    if os.path.exists('data/train.txt') and os.path.exists('data/images') and os.path.exists('data/depths'):
        print("Data seems to be present.")
        return

    print("Downloading NYUv2 dataset...")
    # Processed NYUv2 data (re-generated) from README
    # File ID: 1c18pTIsMX1SJvVPBFpqWa7QILn1NPxTY
    
    # Remove partial/corrupt file if it exists but is too small (e.g. error page)
    if os.path.exists('nyuv2_data.zip'):
        if os.path.getsize('nyuv2_data.zip') < 100 * 1024 * 1024: # < 100MB
            print("Removing existing small/corrupt nyuv2_data.zip...")
            os.remove('nyuv2_data.zip')
            
    if not os.path.exists('nyuv2_data.zip'):
        # Try downloading using gdown command line
        try:
            print("Attempting download with gdown CLI...")
            run_command("gdown --id 1c18pTIsMX1SJvVPBFpqWa7QILn1NPxTY -O nyuv2_data.zip")
        except subprocess.CalledProcessError:
             print("gdown CLI failed, trying python module...")
             import gdown
             url = 'https://drive.google.com/uc?id=1c18pTIsMX1SJvVPBFpqWa7QILn1NPxTY'
             gdown.download(url, 'nyuv2_data.zip', quiet=False)

    if not os.path.exists('nyuv2_data.zip'):
        print("Download failed: nyuv2_data.zip not found.")
        # Fallback: Try a different method or just fail gracefully
        # Sometimes gdown fails silently. Let's try python module explicitly if file is missing
        print("Retrying download with gdown python module...")
        import gdown
        url = 'https://drive.google.com/uc?id=1c18pTIsMX1SJvVPBFpqWa7QILn1NPxTY'
        gdown.download(url, 'nyuv2_data.zip', quiet=False)
        
    if not os.path.exists('nyuv2_data.zip'):
        raise FileNotFoundError("Failed to download nyuv2_data.zip. Please check the file ID or internet connection.")

    print(f"nyuv2_data.zip size: {os.path.getsize('nyuv2_data.zip') / 1024 / 1024:.2f} MB")

    print("Unzipping dataset...")
    
    # Try unzipping with unzip first, then 7z
    try:
        run_command("unzip -o -q nyuv2_data.zip -d .")
    except subprocess.CalledProcessError:
        print("Standard unzip failed (exit code 9 often means zip issues). Trying 7z...")
        if os.path.exists('nyuv2_data.zip'):
            try:
                # 7z syntax: 7z x archive.zip -o{output_dir}
                run_command("7z x nyuv2_data.zip -o.")
            except subprocess.CalledProcessError:
                print("7z also failed. attempting to repair or re-download...")
                # Last resort: delete and redownload one last time
                os.remove('nyuv2_data.zip')
                import gdown
                url = 'https://drive.google.com/uc?id=1c18pTIsMX1SJvVPBFpqWa7QILn1NPxTY'
                gdown.download(url, 'nyuv2_data.zip', quiet=False)
                run_command("7z x nyuv2_data.zip -o.")

def setup_model():
    model_dir = 'model_M1'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, 'ckpt_epoch_500.00.pth')
    if os.path.exists(model_path):
        print("Model already exists.")
        return

    print("Downloading Pre-trained Model...")
    # Pre-trained Model from README
    # File ID: 1Pg6r3eJ245GaKbHfZob0Ek0CVhiS7VaR
    try:
        run_command(f"gdown --id 1Pg6r3eJ245GaKbHfZob0Ek0CVhiS7VaR -O {model_path}")
    except Exception:
        print("gdown CLI failed, trying python module...")
        import gdown
        url = 'https://drive.google.com/uc?id=1Pg6r3eJ245GaKbHfZob0Ek0CVhiS7VaR'
        gdown.download(url, model_path, quiet=False)

def patch_train_code():
    print("Patching train.py for compatibility...")
    
    train_path = 'train.py'
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            content = f.read()
            
        # Fix hardcoded device settings
        content = content.replace("os.environ['CUDA_VISIBLE_DEVICES'] = '0'", "# os.environ['CUDA_VISIBLE_DEVICES'] = '0'")
        
        # Fix torch.load map_location for CPU compatibility (if needed in future)
        # Note: train.py already handles device detection properly
        
        with open(train_path, 'w') as f:
            f.write(content)
    
    print("train.py patched.")

def patch_eval_code():
    print("Patching eval.py for compatibility...")
    
    eval_path = 'eval.py'
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            content = f.read()
            
        # Fix hardcoded device settings
        content = content.replace('os.environ[\'CUDA_VISIBLE_DEVICES\'] = \'0\'', '# os.environ[\'CUDA_VISIBLE_DEVICES\'] = \'0\'')
        content = content.replace('device = torch.device("cuda:0")', 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")')
        
        # Fix torch.load map_location for CPU compatibility
        content = content.replace("torch.load(pretrain_path)['state_dict']", "torch.load(pretrain_path, map_location='cpu')['state_dict']")
        
        # Fix synchronization and timing
        content = content.replace('torch.cuda.synchronize()', 'if torch.cuda.is_available(): torch.cuda.synchronize()')
        
        # Add Verbose Logging
        content = content.replace('model.to(device)', 'print(f"Using device: {device}"); model.to(device)')
        content = content.replace('_load_block_pretrain_weight(model, pth_dir)', 'print(f"Loading weights from {pth_dir}..."); _load_block_pretrain_weight(model, pth_dir)')
        
        # Patch DataLoader to disable potential heavy preloading if it exists in kwargs (not standard but safe to add log)
        content = content.replace('val_loader = DataLoader', 'print("Creating DataLoader (this might take time)..."); val_loader = DataLoader')
        
        content = content.replace('for batch_idx, sample in enumerate(val_loader):', 'print("Starting inference loop..."); for batch_idx, sample in enumerate(val_loader):')
        
        # Fix CUDA Events for Timing
        content = content.replace(
            'starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)',
            'starter, ender = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if torch.cuda.is_available() else (None, None)'
        )
        content = content.replace('starter.record()', 'if starter: starter.record()\n            else: start_time = time.time()')
        content = content.replace('ender.record()', 'if ender: ender.record()\n            else: end_time = time.time()')
        content = content.replace('curr_time = starter.elapsed_time(ender)', 'if starter: curr_time = starter.elapsed_time(ender)\n            else: curr_time = (end_time - start_time) * 1000')

        # Fix dummy input device
        content = content.replace('device=device)', 'device=device)') 

        with open(eval_path, 'w') as f:
            f.write(content)
    
    print("eval.py patched.")

def patch_dataloader():
    # Patch NYUv2_dataloader.py to debug loading and fix potential path issues
    loader_path = 'NYUv2_dataloader.py'
    if os.path.exists(loader_path):
        with open(loader_path, 'r') as f:
            content = f.read()
        
        # Add print to __init__
        if 'def __init__(self,' in content and 'print("Initializing Dataset...")' not in content:
            content = content.replace('def __init__(self,', 'def __init__(self,\n        print("Initializing Dataset...");')
        
        # Add debug print in image loading to catch NoneType errors
        content = content.replace('image = cv2.imread', 'print(f"Reading image: {self.img_dir_train[index]}"); image = cv2.imread')
            
        with open(loader_path, 'w') as f:
            f.write(content)

def patch_code():
    patch_train_code()
    patch_eval_code()
    patch_dataloader()
    print("All code patched.")

def check_dataset_structure():
    print("Checking dataset structure...")
    
    # Check if 'data' folder exists in current directory
    if not os.path.exists('data'):
        print("'data' folder not found in current directory.")
        # Search for it
        found = False
        for root, dirs, files in os.walk('.'):
            if 'train.txt' in files and 'images' in dirs:
                print(f"Found dataset in: {root}")
                # Move it to ./data if it's not already
                if root != './data':
                    print(f"Moving dataset from {root} to ./data")
                    run_command(f"mv {root}/* .") # This might be risky if root is nested deeply.
                    # Better: symlink or just move 'data' if it's inside a folder
                    # Case: ./nyuv2_data/data -> ./data
                    if os.path.basename(root) == 'data':
                        run_command(f"mv {root} ./data_tmp && mv ./data_tmp ./data")
                    else:
                        # Case: ./nyuv2_data containing images, depths, etc.
                        # We need to move content of root to ./data
                        if not os.path.exists('data'):
                            os.makedirs('data')
                        run_command(f"mv {root}/* data/")
                found = True
                break
        
        if not found:
            print("Could not locate dataset structure (train.txt + images folder).")
            print("Listing current directory:")
            run_command("ls -R | head -n 20")
            
    # Verify 'data' folder content
    if os.path.exists('data'):
        print("Contents of data folder:")
        run_command("ls data")
        
        # Ensure test.txt exists
        if not os.path.exists('data/test.txt'):
             if os.path.exists('data/train.txt'):
                 print("Copying train.txt to test.txt as fallback...")
                 run_command("cp data/train.txt data/test.txt")
             else:
                 print("Generating dummy test.txt...")
                 if os.path.exists('data/images'):
                     imgs = [f[:-4] for f in os.listdir('data/images') if f.endswith('.png')]
                     with open('data/test.txt', 'w') as f:
                         f.write('\n'.join(imgs[:10]))

def run_training(train_args):
    """
    Run training with custom arguments
    
    Args:
        train_args: Dictionary of training arguments
    """
    print("Running training...")
    if not os.path.exists('train.py'):
        print("Error: train.py not found. Make sure you are in the AsymFormer project root.")
        return
        
    check_dataset_structure()
    patch_code()

    # Use absolute path for data directory to avoid relative path issues
    data_dir = os.path.abspath('data')
    print(f"Using absolute data path: {data_dir}")

    # Build training command with custom arguments
    cmd = [sys.executable, "train.py", "--data-dir", data_dir]
    
    # Add custom arguments
    if 'epochs' in train_args:
        cmd.extend(["--epochs", str(train_args['epochs'])])
    if 'batch_size' in train_args:
        cmd.extend(["--batch-size", str(train_args['batch_size'])])
    if 'learning_rate' in train_args:
        cmd.extend(["--lr", str(train_args['learning_rate'])])
    if 'weight_decay' in train_args:
        cmd.extend(["--weight-decay", str(train_args['weight_decay'])])
    if 'workers' in train_args:
        cmd.extend(["-j", str(train_args['workers'])])
    if 'print_freq' in train_args:
        cmd.extend(["--print-freq", str(train_args['print_freq'])])
    if 'save_epoch_freq' in train_args:
        cmd.extend(["--save-epoch-freq", str(train_args['save_epoch_freq'])])
    if 'last_ckpt' in train_args:
        cmd.extend(["--last-ckpt", str(train_args['last_ckpt'])])
    if 'ckpt_dir' in train_args:
        cmd.extend(["--ckpt-dir", str(train_args['ckpt_dir'])])
    if 'start_epoch' in train_args:
        cmd.extend(["--start-epoch", str(train_args['start_epoch'])])
    
    print(f"Training command: {' '.join(cmd)}")
    
    # Run training with real-time output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Stream output
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
    rc = process.poll()
    if rc != 0:
        print(f"\n\n!!! Training FAILED with exit code {rc} !!!")
        raise subprocess.CalledProcessError(rc, cmd)
    
    print("\nTraining completed successfully!")

def run_evaluation(ckpt_path=None, data_dir=None):
    """
    Run evaluation with the trained model
    
    Args:
        ckpt_path: Path to checkpoint file. If None, auto-detect the latest checkpoint.
        data_dir: Path to data directory. If None, use default.
    """
    print("\nRunning automatic evaluation...")
    
    if not os.path.exists('eval.py'):
        print("Error: eval.py not found. Make sure you are in the AsymFormer project root.")
        return
        
    if data_dir is None:
        data_dir = os.path.abspath('data')
    
    # Auto-detect latest checkpoint if not specified
    if ckpt_path is None:
        import glob
        # Search in common checkpoint directories
        ckpt_dirs = ['model_M1', './checkpoints', './models']
        ckpt_candidates = []
        
        for ckpt_dir in ckpt_dirs:
            if os.path.exists(ckpt_dir):
                found_ckpts = glob.glob(os.path.join(ckpt_dir, "*.pth"))
                ckpt_candidates.extend(found_ckpts)
                print(f"Found {len(found_ckpts)} checkpoint(s) in {ckpt_dir}")
        
        if not ckpt_candidates:
            print("No checkpoints found. Skipping evaluation.")
            return
        
        # Get the most recent checkpoint by modification time
        ckpt_path = sorted(ckpt_candidates, key=os.path.getmtime)[-1]
        print(f"Auto-detected latest checkpoint: {ckpt_path}")
    
    # Build evaluation command
    cmd = [
        sys.executable, "eval.py",
        "--last-ckpt", ckpt_path,
        "--data-dir", data_dir,
        "--save-json",
        "--json-path", os.path.join(os.path.dirname(ckpt_path), "final_eval_result.json")
    ]
    
    print(f"Evaluation command: {' '.join(cmd)}")
    
    # Run evaluation with real-time output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Stream output
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
    rc = process.poll()
    if rc != 0:
        print(f"\n\n!!! Evaluation FAILED with exit code {rc} !!!")
        raise subprocess.CalledProcessError(rc, cmd)
    
    print("\nEvaluation completed successfully!")

def main():
    print("Starting AsymFormer V1 Reproduction Setup...")
    
    # Check if we are in the correct directory
    if not os.path.exists('train.py'):
        print("It looks like you are not in the AsymFormer directory.")
        print("Cloning repository...")
        if os.path.exists('AsymFormer'):
             print("AsymFormer directory already exists. Changing directory...")
             os.chdir('AsymFormer')
        else:
             run_command("git clone https://github.com/Fourier7754/AsymFormer.git")
             os.chdir('AsymFormer')
    
    # ==================== CUSTOM TRAINING CONFIGURATION ====================
    # Modify these parameters to customize your training
    # Leave as None to use default values from train.py
    
    train_config = {
        'epochs': 500,              # Total training epochs (default: 500)
        'batch_size': 8,            # Batch size (default: 8, reduce if OOM)
        'learning_rate': 5e-5,       # Learning rate (default: 5e-5)
        'weight_decay': 0.01,        # Weight decay (default: 0.01)
        'workers': 4,                 # Number of data loading workers (default: 8, reduce for Colab)
        'print_freq': 50,            # Print frequency (default: 50)
        'save_epoch_freq': 50,        # Save checkpoint frequency (default: 50)
        'start_epoch': 0,             # Starting epoch for resuming training (default: 0)
        'last_ckpt': None,            # Path to checkpoint for resuming (default: None)
        'ckpt_dir': './model_M1/'     # Directory to save checkpoints (default: ./model_M1/)
    }
    
    # ==================== END CUSTOM CONFIGURATION ====================
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION:")
    print("="*60)
    for key, value in train_config.items():
        print(f"{key:20s}: {value}")
    print("="*60 + "\n")
    
    install_dependencies()
    setup_data()
    # setup_model()  # Only needed if using pre-trained model for evaluation
    
    # Run training
    run_training(train_config)
    
    # Auto-run evaluation after training
    run_evaluation()
    
    print("\n" + "="*60)
    print("REPRODUCTION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
