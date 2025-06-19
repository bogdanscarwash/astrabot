# How to Process Signal Backup Files

This guide explains how to extract data from Signal backup files for use with Astrabot.

## Prerequisites

- Docker installed and running
- Your Signal backup file (.backup)
- Your 30-digit backup password

## Finding Your Signal Backup

### On Android
1. Open Signal
2. Go to Settings → Chats → Chat backups
3. Note the 30-digit password (digits are grouped)
4. Find backup in: `/storage/emulated/0/Signal/Backups/`

### On iOS
Signal for iOS doesn't support the same backup format. You'll need to:
1. Transfer your Signal account to an Android device temporarily
2. Create a backup on Android
3. Transfer back to iOS if desired

## Processing the Backup

### Step 1: Build Docker Image

```bash
cd docker/signalbackup-tools
docker build -t signalbackup-tools .
cd ../..
```

### Step 2: Run Extraction

Basic extraction:
```bash
docker run -v /path/to/backup:/backup -v $(pwd)/data/raw/signal-flatfiles:/output \
  signalbackup-tools /backup/signal-2024-01-01-00-00-00.backup \
  --password "12345 67890 12345 67890 12345 67890" \
  --output /output --csv
```

### Step 3: Verify Output

Check that CSV files were created:
```bash
ls -la data/raw/signal-flatfiles/
```

You should see:
- `signal.csv` - Main messages table
- `recipient.csv` - Contact information  
- `thread.csv` - Conversation threads
- `reaction.csv` - Message reactions
- And several other tables

## Understanding the Data

### Key Tables

1. **signal.csv**: Contains all messages
   - `_id`: Message ID
   - `thread_id`: Conversation ID
   - `from_recipient_id`: Sender ID
   - `body`: Message text
   - `date_sent`: Timestamp (milliseconds)

2. **recipient.csv**: Contact information
   - `_id`: Recipient ID
   - `profile_given_name`: Display name
   - `blocked`: Whether contact is blocked

3. **thread.csv**: Conversation metadata
   - `_id`: Thread ID
   - `recipient_id`: Main recipient

## Privacy Considerations

1. **Backup File**: Keep your backup file secure
2. **Password**: Never share your backup password
3. **Extracted Data**: The CSV files contain all your messages in plain text
4. **Cleanup**: Delete files you don't need

## Troubleshooting

### "Incorrect password" Error
- Ensure you're using the exact 30-digit password
- Include spaces if copying from Signal
- Try without spaces if that fails

### Docker Permission Errors
```bash
# On Linux, you might need sudo
sudo docker run ...

# Or add your user to docker group
sudo usermod -aG docker $USER
```

### Large Backup Files
For backups over 1GB, extraction may take 10-30 minutes. Be patient.

### Corrupted Backup
If extraction fails:
1. Try creating a fresh backup
2. Ensure backup completed successfully in Signal
3. Check disk space

## Next Steps

After successful extraction:
1. Review the data quality
2. Check your recipient ID (usually 2)
3. Proceed to [create training data](create-training-data.md)