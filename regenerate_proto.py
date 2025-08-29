#!/usr/bin/env python3
"""
Complete script to regenerate protobuf files and automatically fix imports.
Use this instead of running protoc manually.
"""

import subprocess
import sys
import os
from fix_proto_imports import fix_grpc_imports

def run_protoc():
    """Run the protoc command to generate protobuf files."""
    
    print("🔄 Regenerating protobuf files...")
    
    # The protoc command
    cmd = [
        "python", "-m", "grpc_tools.protoc",
        "--proto_path=proto",
        "--python_out=proto", 
        "--grpc_python_out=proto",
        "proto/model_service.proto"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Protobuf files generated successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating protobuf files: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def main():
    """Main function to regenerate and fix proto files."""
    
    print("🚀 Starting protobuf regeneration and fix process...")
    
    # Step 1: Generate protobuf files
    if not run_protoc():
        print("❌ Failed to generate protobuf files. Exiting.")
        sys.exit(1)
    
    # Step 2: Fix imports
    print("🔧 Fixing import issues...")
    grpc_file = "proto/model_service_pb2_grpc.py"
    
    if fix_grpc_imports(grpc_file):
        print("✅ Import fixes applied successfully!")
    else:
        print("ℹ️  No import fixes needed.")
    
    print("🎉 All done! Your protobuf files are ready to use.")

if __name__ == "__main__":
    main()
