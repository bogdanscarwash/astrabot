# Environment Variables Guide

This guide explains how to securely manage environment variables in the Astrabot project.

## Quick Start

1. **Copy the example file**:
   ```bash
   cp .env.example .env
   ```

2. **Run the setup script** (recommended):
   ```bash
   python scripts/setup-secrets.py
   ```

3. **Or manually edit `.env`**:
   ```bash
   nano .env  # or your preferred editor
   ```

## Security Best Practices

### 🔒 Never Commit Secrets

- `.env` is in `.gitignore` - NEVER remove it
- Never commit API keys, passwords, or personal data
- Use `.env.example` to document required variables (without values)

### 🔑 Secure File Permissions

On Linux/macOS, restrict access to your `.env` file:
```bash
chmod 600 .env  # Read/write for owner only
```

### 🔄 Rotate Keys Regularly

- Change API keys periodically
- Revoke old keys after rotation
- Monitor API usage for anomalies

## Available Environment Variables

### API Keys (Required for some features)

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o-mini vision | For image processing |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | For Claude vision |

### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `YOUR_RECIPIENT_ID` | Your Signal recipient ID | `2` |
| `DEBUG` | Enable debug mode | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Feature Flags

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_IMAGE_PROCESSING` | Enable image description features | `true` |
| `ENABLE_BATCH_PROCESSING` | Enable batch API calls | `true` |
| `MAX_BATCH_SIZE` | Maximum images per batch | `10` |

### Paths

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Data directory path | `./data` |
| `OUTPUT_DIR` | Output directory path | `./output` |

## Usage in Code

### Using the Config Module

```python
from config import config

# Check if API is configured
if config.has_openai():
    print("OpenAI is configured")

# Get a required value (raises if not set)
api_key = config.require('OPENAI_API_KEY')

# Get with default
debug = config.get('DEBUG', False)

# Direct access
your_id = config.YOUR_RECIPIENT_ID
```

### Validation

Check your configuration:
```bash
python config.py
```

This will show:
- Which APIs are configured
- Current settings
- Directory status

## Alternative Methods

### 1. System Environment Variables

Export variables in your shell:
```bash
export OPENAI_API_KEY="your-key-here"
export YOUR_RECIPIENT_ID="2"
```

Add to `~/.bashrc` or `~/.zshrc` to persist.

### 2. Virtual Environment Variables

Activate virtual environment and set:
```bash
source venv/bin/activate
export OPENAI_API_KEY="your-key-here"
```

### 3. IDE Configuration

**VS Code**: Create `.vscode/settings.json`:
```json
{
  "terminal.integrated.env.linux": {
    "OPENAI_API_KEY": "your-key-here"
  }
}
```

**PyCharm**: Run Configuration → Environment Variables

## Docker Secrets (Advanced)

For production deployments:

```yaml
# docker-compose.yml
version: '3.8'
services:
  astrabot:
    build: .
    secrets:
      - openai_key
      - anthropic_key
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_key
      - ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_key

secrets:
  openai_key:
    file: ./secrets/openai_key.txt
  anthropic_key:
    file: ./secrets/anthropic_key.txt
```

## Troubleshooting

### API Key Not Found

```python
# Check if key is loaded
from config import config
config.print_status()
```

### Permission Denied

```bash
# Fix permissions
chmod 600 .env
```

### Environment Not Loading

Ensure you're in the project directory:
```bash
cd /path/to/astrabot
python config.py
```

## Security Checklist

- [ ] `.env` is in `.gitignore`
- [ ] Never logged API keys
- [ ] File permissions set to 600
- [ ] Using strong, unique API keys
- [ ] Keys are rotated regularly
- [ ] No secrets in code comments
- [ ] No secrets in commit messages
- [ ] Production uses separate keys

## Emergency: Exposed Secrets

If you accidentally commit secrets:

1. **Immediately revoke the exposed keys**
2. **Generate new keys**
3. **Remove from Git history**:
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch .env" \
     --prune-empty --tag-name-filter cat -- --all
   ```
4. **Force push** (coordinate with team)
5. **Audit API usage** for unauthorized access