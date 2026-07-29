# Test that web frontend builds
import subprocess
import os
os.chdir('C:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/web')
result = subprocess.run(
    ['npx', 'tsc', '--noEmit'],
    capture_output=True,
    text=True,
    timeout=300,
)
print("STDOUT:", result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print("Exit code:", result.returncode)
