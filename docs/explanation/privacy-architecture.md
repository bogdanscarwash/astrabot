# Privacy Architecture and Best Practices

This document explains Astrabot's privacy architecture and provides comprehensive guidance for protecting your personal data throughout the AI training process.

## Privacy Philosophy

Astrabot is built on the principle that **your conversations are yours alone**. Creating a personalized AI shouldn't require sacrificing privacy. Our architecture ensures:

1. **Local-first processing** - Your data never leaves your machine unless you explicitly choose to share it
2. **Defense in depth** - Multiple layers of privacy protection
3. **Transparent data handling** - You know exactly what data is used and how
4. **User control** - You decide what to include, exclude, or anonymize

## Data Classification System

Understanding data sensitivity is crucial for privacy. Astrabot classifies data into four privacy levels:

### 🔴 Critical Privacy (Highest Protection)

**What it includes:**
- Message content (actual text of conversations)
- Phone numbers (E.164 format)
- Contact names (both profile and system names)
- Cryptographic keys and identifiers

**Protection measures:**
- Never logged in any form
- Automatically masked in all outputs
- Encrypted in memory when possible
- Requires explicit user action to export

**Example handling:**
```python
# Bad - Never do this
logger.info(f"Processing message: {message_text}")

# Good - Log only metadata
logger.info(f"Processing message ID: {message_id}, length: {len(message_text)}")
```

### 🟠 High Privacy (Strong Protection)

**What it includes:**
- User/group identifiers (ACI, PNI)
- Social connections (who talks to whom)
- Group memberships
- Message metadata (sender/receiver IDs)

**Protection measures:**
- Masked in logs with partial reveal
- Anonymized in exports by default
- Access controlled in processed data

**Example handling:**
```python
# Mask user identifiers
masked_id = mask_sensitive_data(user_aci)  # Shows: "xxx...xxx"
logger.info(f"Processing user: {masked_id}")
```

### 🟡 Medium Privacy (Moderate Protection)

**What it includes:**
- Timestamps (communication patterns)
- Read receipts and delivery status
- Blocked contact status
- Message counts and statistics

**Protection measures:**
- Aggregated in public outputs
- Timezone information stripped
- Patterns obfuscated in shared models

### 🟢 Low Privacy (Basic Protection)

**What it includes:**
- System metadata (thread IDs, message types)
- Application version information
- Non-identifying statistics

**Protection measures:**
- Can be logged for debugging
- Safe to include in error reports

## Built-in Privacy Features

### 1. Automatic Sensitive Data Masking

The logging system automatically detects and masks sensitive patterns:

```python
from src.utils import mask_sensitive_data

# Automatic masking examples
text = "My API key is sk-abc123xyz and phone is 555-123-4567"
masked = mask_sensitive_data(text)
# Result: "My API key is sk-...masked and phone is XXX-XXX-4567"
```

**Patterns automatically masked:**
- API keys (OpenAI, Anthropic, Google, etc.)
- Bearer tokens and auth headers
- Password fields
- Generic "token" or "key" patterns

### 2. Secure Configuration Management

```python
from src.utils import config

# Safe - API keys never exposed
if config.has_openai():
    # Use API without exposing key
    api_key = config.require('OPENAI_API_KEY')

# Status display masks sensitive values
config.print_status()
# Output:
# ✓ OPENAI_API_KEY: sk-...ged (configured)
# ✗ ANTHROPIC_API_KEY: Not configured
```

### 3. Privacy-Aware Logging

```python
from src.utils import get_logger, log_performance

logger = get_logger(__name__)

# Safe logging with context
with logger.add_context(operation="processing", record_count=1000):
    # Sensitive data automatically filtered
    logger.info("Processing messages")
    
@log_performance("data_processing")
def process_data(messages):
    # Performance logged without exposing data
    return transform_messages(messages)
```

### 4. Data Filtering Pipeline

```python
# Filter blocked contacts
filtered_messages = messages_df[
    ~messages_df['from_recipient_id'].isin(blocked_recipients)
]

# Remove sensitive conversations
sensitive_keywords = ['password', 'ssn', 'credit card']
for keyword in sensitive_keywords:
    filtered_messages = filtered_messages[
        ~filtered_messages['body'].str.contains(keyword, case=False, na=False)
    ]
```

## Configuration for Maximum Privacy

### Environment Setup

Create a `.env` file with privacy-focused settings:

```bash
# Privacy-first configuration
YOUR_RECIPIENT_ID=2
ENABLE_IMAGE_PROCESSING=false  # Disable external API calls
ENABLE_BATCH_PROCESSING=false  # Process one at a time
DEBUG=false  # Minimize logging
LOG_LEVEL=WARNING  # Only important messages

# Local-only processing
OFFLINE_MODE=true
DISABLE_TELEMETRY=true
```

### Secure File Permissions

```bash
# Restrict access to sensitive files
chmod 600 .env
chmod 700 data/raw/signal-flatfiles
chmod 600 data/processed/*

# Create secure directories
mkdir -p -m 700 data/secure
mkdir -p -m 700 models/private
```

### Memory Security

```python
import gc
import ctypes

# Clear sensitive data from memory
def secure_delete(data):
    """Overwrite memory before garbage collection"""
    if isinstance(data, str):
        # Overwrite string memory
        ctypes.memset(id(data), 0, len(data))
    gc.collect()

# Use in processing
sensitive_data = load_messages()
process_messages(sensitive_data)
secure_delete(sensitive_data)
```

## Data Handling Guidelines

### Do's ✅

1. **Always anonymize before sharing**
   ```python
   def anonymize_messages(messages_df):
       # Replace names with roles
       messages_df['body'] = messages_df['body'].str.replace(
           r'\b[A-Z][a-z]+\b', '[NAME]', regex=True
       )
       
       # Hash phone numbers
       messages_df['e164_hash'] = messages_df['e164'].apply(
           lambda x: hashlib.sha256(x.encode()).hexdigest()[:8] if pd.notna(x) else None
       )
       messages_df.drop('e164', axis=1, inplace=True)
       
       return messages_df
   ```

2. **Use encryption for exports**
   ```python
   from cryptography.fernet import Fernet
   
   # Generate encryption key
   key = Fernet.generate_key()
   cipher = Fernet(key)
   
   # Encrypt training data
   with open('training_data.json', 'rb') as f:
       encrypted = cipher.encrypt(f.read())
   
   with open('training_data.enc', 'wb') as f:
       f.write(encrypted)
   ```

3. **Implement access controls**
   ```python
   import os
   
   def load_secure_data(path, required_permission=0o600):
       # Check file permissions
       stat_info = os.stat(path)
       if stat_info.st_mode & 0o777 > required_permission:
           raise PermissionError(f"File {path} has too permissive permissions")
       
       return pd.read_csv(path)
   ```

### Don'ts ❌

1. **Never log message content**
   ```python
   # WRONG
   logger.debug(f"Message: {message_body}")
   
   # RIGHT
   logger.debug(f"Processing message {msg_id} ({len(message_body)} chars)")
   ```

2. **Never commit sensitive data**
   ```bash
   # Add to .gitignore
   echo "data/" >> .gitignore
   echo ".env" >> .gitignore
   echo "*.backup" >> .gitignore
   echo "models/" >> .gitignore
   ```

3. **Never transmit unencrypted data**
   ```python
   # WRONG
   requests.post('http://api.example.com', json=training_data)
   
   # RIGHT
   encrypted_data = encrypt(training_data)
   requests.post('https://api.example.com', 
                data=encrypted_data,
                headers={'Content-Type': 'application/octet-stream'})
   ```

## Anonymization Techniques

### Basic Anonymization

```python
def basic_anonymize(text):
    """Remove common PII patterns"""
    import re
    
    # Phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                  '[EMAIL]', text)
    
    # Social Security Numbers
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    # Credit card numbers
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CC]', text)
    
    return text
```

### Advanced Anonymization

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def advanced_anonymize(text):
    """Use NLP to detect and replace entities"""
    doc = nlp(text)
    
    # Replace named entities
    replacements = []
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']:
            replacements.append((ent.start_char, ent.end_char, f'[{ent.label_}]'))
    
    # Apply replacements in reverse order
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    
    return text
```

### Differential Privacy

```python
import numpy as np

def add_differential_privacy(messages_df, epsilon=1.0):
    """Add noise to protect individual messages"""
    # Add Laplace noise to message lengths
    sensitivity = 1.0  # Max change from single message
    scale = sensitivity / epsilon
    
    noise = np.random.laplace(0, scale, len(messages_df))
    messages_df['noisy_length'] = messages_df['body'].str.len() + noise
    
    # Randomly drop some messages
    drop_probability = 1 / (1 + np.exp(epsilon))
    mask = np.random.random(len(messages_df)) > drop_probability
    
    return messages_df[mask]
```

## Sharing Models Safely

### Pre-sharing Checklist

Before sharing a trained model, verify:

1. **No training data embedded**
   ```python
   # Check model files
   import torch
   
   model = torch.load('model.pt')
   
   # Ensure no training data in model
   assert 'training_data' not in model
   assert 'examples' not in model
   
   # Remove optimizer state (may contain gradients)
   if 'optimizer_state_dict' in model:
       del model['optimizer_state_dict']
   ```

2. **No personal patterns identifiable**
   ```python
   # Test for personal information leakage
   test_prompts = [
       "What's my phone number?",
       "What's my address?",
       "Who do I talk to most?",
   ]
   
   for prompt in test_prompts:
       response = model.generate(prompt)
       assert not contains_pii(response)
   ```

3. **Model card disclosure**
   ```markdown
   # Model Card
   
   ## Privacy Considerations
   - Trained on anonymized conversation data
   - No real names or identifiers included
   - Communication patterns only, no specific content
   - Filtered for sensitive information
   ```

### Safe Sharing Methods

```python
def prepare_model_for_sharing(model_path, output_path):
    """Prepare model for safe sharing"""
    import torch
    import json
    
    # Load model
    model = torch.load(model_path)
    
    # Remove sensitive components
    safe_model = {
        'model_state_dict': model['model_state_dict'],
        'config': model['config'],
        'training_args': {
            'num_epochs': model['training_args']['num_epochs'],
            'model_type': model['training_args']['model_type']
            # Exclude paths, data references
        }
    }
    
    # Add privacy notice
    safe_model['privacy_notice'] = {
        'anonymized': True,
        'data_filtered': True,
        'personal_info_removed': True,
        'created_date': datetime.now().isoformat()
    }
    
    # Save safely
    torch.save(safe_model, output_path)
    
    # Create accompanying documentation
    with open(output_path + '.privacy', 'w') as f:
        json.dump({
            'privacy_measures': [
                'Data anonymization applied',
                'Sensitive content filtered',
                'No personal identifiers included'
            ],
            'safe_for_sharing': True
        }, f, indent=2)
```

## Compliance Considerations

### GDPR Compliance

For EU users or when handling EU citizen data:

1. **Right to Access**
   ```python
   def export_user_data(user_id):
       """Export all data for a specific user"""
       user_messages = messages_df[
           (messages_df['from_recipient_id'] == user_id) |
           (messages_df['to_recipient_id'] == user_id)
       ]
       
       return {
           'messages': user_messages.to_dict('records'),
           'metadata': {
               'export_date': datetime.now().isoformat(),
               'message_count': len(user_messages)
           }
       }
   ```

2. **Right to Erasure**
   ```python
   def delete_user_data(user_id):
       """Completely remove user data"""
       # Remove from messages
       global messages_df
       messages_df = messages_df[
           (messages_df['from_recipient_id'] != user_id) &
           (messages_df['to_recipient_id'] != user_id)
       ]
       
       # Remove from recipients
       global recipients_df
       recipients_df = recipients_df[recipients_df['_id'] != user_id]
       
       # Log deletion (without user details)
       logger.info(f"User data deleted for ID: {hash(user_id)}")
   ```

3. **Data Minimization**
   ```python
   def minimize_data_collection(messages_df):
       """Keep only necessary fields"""
       essential_columns = [
           '_id', 'thread_id', 'from_recipient_id', 
           'to_recipient_id', 'body', 'date_sent'
       ]
       
       return messages_df[essential_columns]
   ```

### Data Retention

Implement automatic data cleanup:

```python
def setup_data_retention(retention_days=365):
    """Delete data older than retention period"""
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    cutoff_timestamp = int(cutoff_date.timestamp() * 1000)
    
    # Archive old data
    old_messages = messages_df[messages_df['date_sent'] < cutoff_timestamp]
    old_messages.to_csv('archive/old_messages.csv.gz', compression='gzip')
    
    # Remove from active dataset
    return messages_df[messages_df['date_sent'] >= cutoff_timestamp]
```

## Security Audit Checklist

Regular privacy audits ensure continued protection:

### Daily Checks
- [ ] No sensitive data in logs
- [ ] Proper file permissions maintained
- [ ] No unencrypted exports created

### Weekly Checks
- [ ] Review access logs
- [ ] Verify data minimization
- [ ] Check for unauthorized access attempts

### Monthly Checks
- [ ] Full data audit
- [ ] Update anonymization patterns
- [ ] Review and update privacy policies
- [ ] Test data deletion procedures

### Code Review Checklist
```python
# Before committing code, verify:
def privacy_code_review():
    checks = {
        'no_api_keys': not contains_api_keys(code),
        'no_phone_numbers': not contains_phone_numbers(code),
        'no_names': not contains_personal_names(code),
        'logging_safe': verify_safe_logging(code),
        'data_masked': verify_data_masking(code)
    }
    
    return all(checks.values()), checks
```

## Privacy-Preserving Architecture

### Data Flow with Privacy Boundaries

```
┌─────────────────────────────────────────────────────────┐
│                   User's Device                          │
│  ┌─────────────┐     ┌──────────────┐                  │
│  │Signal Backup│ --> │Docker Extract│ (Isolated)        │
│  └─────────────┘     └──────────────┘                  │
│         │                    │                           │
│         v                    v                           │
│  ┌─────────────┐     ┌──────────────┐                  │
│  │  Raw CSVs   │ --> │ Anonymizer   │ (PII Removal)    │
│  └─────────────┘     └──────────────┘                  │
│         │                    │                           │
│         └────────┬───────────┘                          │
│                  v                                       │
│           ┌──────────────┐                              │
│           │  Processing  │ (Local Only)                 │
│           └──────────────┘                              │
│                  │                                       │
│                  v                                       │
│           ┌──────────────┐                              │
│           │Training Data │ (Encrypted)                  │
│           └──────────────┘                              │
└─────────────────────────────────────────────────────────┘
                   │
                   v
            ┌──────────────┐
            │Trained Model │ (Sanitized)
            └──────────────┘
```

### Privacy Enforcement Points

1. **Extraction boundary** - Data never leaves Docker container unprocessed
2. **Anonymization gate** - All PII removed before processing
3. **Processing isolation** - No network access during processing
4. **Export sanitization** - Models cleaned before sharing

## Emergency Response

### Data Breach Protocol

If personal data might be exposed:

1. **Immediate containment**
   ```bash
   # Revoke all API keys
   rm .env
   
   # Lock down files
   chmod 000 data/
   
   # Stop all processes
   pkill -f astrabot
   ```

2. **Assessment**
   ```python
   # Check what was accessed
   def audit_access_logs():
       logs = parse_logs('data/logs/access.log')
       suspicious = [
           log for log in logs 
           if log['status'] != 200 or log['ip'] not in trusted_ips
       ]
       return suspicious
   ```

3. **Notification**
   - Document what data was potentially exposed
   - Identify affected conversations
   - Prepare notification for affected contacts

## Best Practices Summary

### For Developers

1. **Privacy by Design** - Consider privacy at every step
2. **Data Minimization** - Only collect what's needed
3. **Secure by Default** - Require opt-in for sharing
4. **Transparent Operations** - Log actions, not data
5. **Regular Audits** - Continuously verify privacy

### For Users

1. **Review Before Processing** - Check what data you're including
2. **Use Anonymization** - Always anonymize before sharing
3. **Local First** - Keep processing on your machine
4. **Backup Securely** - Encrypt all backups
5. **Share Cautiously** - Understand model privacy implications

## Conclusion

Privacy in Astrabot isn't an afterthought—it's fundamental to the architecture. By following these practices, you can create personalized AI models while maintaining complete control over your personal data.

Remember: **Your conversations shape your AI, but they remain yours alone.**

## See Also

- [Security Configuration](../reference/utilities.md#configuration-configpy)
- [Logging Security Features](../reference/utilities.md#logging-loggingpy)
- [Data Schema Privacy Levels](../reference/signal-data-schema.md#privacy-and-security-considerations)