#!/usr/bin/env python


import os
import sys
import chardet
from pathlib import Path

def scan_and_fix_encodings(directory, fix_mode=False):
    """
    Comprehensive scan and repair of file encodings.

    Args:
        directory: Root directory to scan
        fix_mode: If True, automatically converts problematic files to UTF-8
    """
    print(f"{'FIXING' if fix_mode else 'SCANNING'} ENCODINGS IN: {directory}\n")
    print("=" * 80)

    problematic_files = []

    for root, _, files in os.walk(directory):
        # Skip version control and cache directories
        if any(skip in root for skip in ['.git', '__pycache__', '.tox', '.venv', 'venv']):
            continue

        for filename in files:
            # Focus on text-based files that might be processed during installation
            if filename.endswith(('.py', '.md', '.toml', '.txt', '.ini', '.cfg', '.in')):
                file_path = os.path.join(root, filename)
                try:
                    # Use binary mode to detect actual encoding
                    with open(file_path, 'rb') as f:
                        raw_content = f.read()

                    # Skip empty files
                    if not raw_content:
                        continue

                    # Detect encoding with confidence
                    result = chardet.detect(raw_content)
                    encoding = result['encoding']
                    confidence = result['confidence']

                    # Check for potentially problematic encodings or low confidence
                    if encoding != 'utf-8' and encoding != 'ascii':
                        problematic_files.append((file_path, encoding, confidence))
                        print(f"ISSUE: {file_path}")
                        print(f"    Detected encoding: {encoding} (confidence: {confidence:.2f})")

                        if fix_mode:
                            try:
                                # Try to decode with detected encoding then re-encode as UTF-8
                                content = raw_content.decode(encoding, errors='replace')
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(content)
                                print(f"    ✓ FIXED: Converted to UTF-8")
                            except Exception as e:
                                print(f"    ✗ ERROR: Failed to convert: {str(e)}")
                        else:
                            print(f"    Run with --fix to automatically convert to UTF-8")

                except Exception as e:
                    print(f"ERROR processing {file_path}: {str(e)}")

    print("\n" + "=" * 80)
    if problematic_files:
        print(f"\nFound {len(problematic_files)} files with encoding issues.")
        if not fix_mode:
            print("\nRun with --fix parameter to automatically convert files to UTF-8.")
    else:
        print("\nNo encoding issues detected. All files are properly encoded.")

    return problematic_files

if __name__ == "__main__":
    try:
        # Check for dependencies
        try:
            import chardet
        except ImportError:
            print("Installing required dependency (chardet)...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"])
            import chardet

        # Process command line arguments
        fix_mode = "--fix" in sys.argv
        directory = "."

        # Allow specifying directory
        for arg in sys.argv[1:]:
            if not arg.startswith("--") and os.path.isdir(arg):
                directory = arg

        problematic_files = scan_and_fix_encodings(directory, fix_mode)

        if problematic_files and not fix_mode:
            sys.exit(1)  # Indicate issues were found but not fixed

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)