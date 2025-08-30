#!/usr/bin/env python3
"""
Script to fix the absolute imports in generated gRPC files to relative imports.
Run this after regenerating proto files with protoc.
"""

import os
import re

def fix_grpc_imports(grpc_file_path):
    """Fix absolute imports to relative imports in gRPC generated files."""
    
    if not os.path.exists(grpc_file_path):
        print(f"File not found: {grpc_file_path}")
        return False
    
    # Read the file
    with open(grpc_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to find the problematic import line
    # Looks for: import model_service_pb2 as model__service__pb2
    pattern = r'^import\s+model_service_pb2\s+as\s+model__service__pb2$'
    replacement = 'from . import model_service_pb2 as model__service__pb2'
    
    # Replace the import
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Check if any changes were made
    if new_content != content:
        # Write the fixed content back
        with open(grpc_file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed imports in {grpc_file_path}")
        return True
    else:
        print(f"No changes needed in {grpc_file_path}")
        return False

def main():
    """Main function to fix proto imports."""
    
    # Path to the gRPC file
    grpc_file = "proto/model_service_pb2_grpc.py"
    
    print("Fixing protobuf import issues...")
    
    if fix_grpc_imports(grpc_file):
        print("All proto imports fixed successfully!")
    else:
        print("No fixes were needed.")

if __name__ == "__main__":
    main()
