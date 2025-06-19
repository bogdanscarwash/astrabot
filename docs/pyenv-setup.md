# Pyenv Setup Guide for Astrabot

This guide explains how to set up pyenv for the Astrabot project on Debian 12.

## Installing pyenv on Debian 12

```bash
# Install dependencies
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
    liblzma-dev python3-openssl git

# Install pyenv
curl https://pyenv.run | bash

# Add to ~/.bashrc
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# Reload shell
source ~/.bashrc
```

## Setting up Python for Astrabot

```bash
# Install Python 3.11.9 (recommended for compatibility)
pyenv install 3.11.9

# Set local Python version for the project
cd /home/percy/git/astrabot
pyenv local 3.11.9

# Verify
python --version  # Should show Python 3.11.9
```

## Shebang Lines in Scripts

All Python scripts in this project use the pyenv-compatible shebang:

```python
#!/usr/bin/env python3
```

This ensures the script uses the Python version managed by pyenv.

## Virtual Environment (Optional but Recommended)

```bash
# Create a virtual environment specific to this project
python -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Making Scripts Executable

```bash
# Make Python scripts executable
chmod +x test_runner.py
chmod +x test_conversation_utilities.py
chmod +x test_structured_outputs.py
```

## Running Scripts

With pyenv properly configured, you can run scripts directly:

```bash
./test_runner.py all
./test_conversation_utilities.py
```

Or use Python explicitly:

```bash
python test_runner.py all
python test_conversation_utilities.py
```

## Troubleshooting

### Script uses system Python instead of pyenv Python

1. Check your PATH:
   ```bash
   echo $PATH
   which python3
   ```

2. Ensure pyenv is initialized:
   ```bash
   pyenv init
   ```

3. Verify the shebang line is `#!/usr/bin/env python3`

### Permission denied when running script

```bash
chmod +x script_name.py
```

### Module not found errors

Ensure you're in the correct directory and have installed requirements:

```bash
cd /home/percy/git/astrabot
pip install -r requirements.txt
```