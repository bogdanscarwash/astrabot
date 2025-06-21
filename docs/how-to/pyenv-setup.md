# Setup Guide for Astrabot

This guide explains how to set up pyenv for the Astrabot project on Debian 12 and other systems.

## Why Pyenv?

Pyenv allows you to easily switch between multiple versions of Python and ensures consistent development environments across different systems.

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
cd /path/to/astrabot
pyenv local 3.11.9

# Verify
python --version  # Should show Python 3.11.9
```

## Modern Setup with UV

Astrabot now uses `uv` for fast, reliable package management. The bootstrap script handles everything:

```bash
# Run the bootstrap script
bash scripts/setup/bootstrap.sh
```

This will:

1. Check Python version (3.9+ required)
2. Install `uv` package manager
3. Create a virtual environment in `.venv`
4. Install all dependencies from `pyproject.toml`
5. Set up pre-commit hooks

## Manual Virtual Environment (Alternative)

If you prefer traditional venv:

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate

# Install using uv
uv sync
```

## Shebang Lines in Scripts

All Python scripts in this project use the pyenv-compatible shebang:

```python
#!/usr/bin/env python3
```

This ensures the script uses the Python version managed by pyenv.

## Making Scripts Executable

```bash
# Example scripts
chmod +x scripts/train.py
chmod +x scripts/process_signal_data.py
chmod +x scripts/setup/setup-secrets.py
```

## Running Scripts

With pyenv properly configured, you can run scripts directly:

```bash
./scripts/train.py --help
./scripts/setup/setup-secrets.py
```

Or use Python explicitly:

```bash
python scripts/train.py --config configs/training_config.yaml
```

## Using Just Task Runner

For the best development experience, install `just`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin
```

Then use simple commands:

```bash
just test        # Run tests
just format      # Format code
just train       # Start training
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

Ensure you're in the correct directory and have run the setup:

```bash
cd /path/to/astrabot
bash scripts/setup/bootstrap.sh
source .venv/bin/activate
```

### UV not found

The bootstrap script installs `uv` automatically, but if you need to install it manually:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Benefits of UV over pip

- **Speed**: 10-100x faster than pip
- **Reliability**: Better dependency resolution
- **Lock files**: Automatic `uv.lock` for reproducible installs
- **Integration**: Works seamlessly with `pyproject.toml`
