# Signal CSV Database Schema Analysis

## Overview
The Signal backup has been exported to CSV files representing different tables from the Signal SQLite database. The main tables contain messages, contacts, conversations, reactions, and group information.

## Primary Tables and Schema

### 1. **signal.csv** (Main Messages Table)
- **Lines**: 109,189 (largest table - contains all messages)
- **Purpose**: Stores all individual messages sent and received
- **Key Fields**:
  - `_id`: Primary key for messages
  - `date_sent`, `date_received`, `date_server`: Timestamps
  - `thread_id`: Foreign key to thread.csv (conversation ID)
  - `from_recipient_id`: Foreign key to recipient.csv (sender)
  - `to_recipient_id`: Foreign key to recipient.csv (receiver)
  - `type`: Message type code (10485780 appears common)
  - `body`: Message text content (often empty for media messages)
  - `read`: Boolean read status
  - `quote_id`, `quote_author`, `quote_body`: For quoted replies
  - `server_guid`: Unique server identifier
  - `mentions_self`: Boolean for mentions
  - `remote_deleted`: Boolean for deleted messages

### 2. **recipient.csv** (Contacts/Recipients)
- **Lines**: 973
- **Purpose**: Stores all contacts, groups, and system recipients
- **Key Fields**:
  - `_id`: Primary key (referenced by message tables)
  - `type`: Recipient type (0=individual, 2=group, 4=?)
  - `e164`: Phone number in E.164 format
  - `aci`: Account identifier (UUID)
  - `pni`: Phone number identifier
  - `username`: Signal username
  - `blocked`: Boolean blocking status
  - `profile_given_name`, `profile_family_name`: Profile names
  - `system_given_name`, `system_family_name`: Phone contact names
  - `registered`: Registration status (0=not on Signal, 1=registered, 2=?)
  - **Sensitive**: Contains phone numbers, names, profile information

### 3. **thread.csv** (Conversations)
- **Lines**: 418
- **Purpose**: Represents conversation threads (1-to-1 and groups)
- **Key Fields**:
  - `_id`: Primary key (referenced by messages)
  - `recipient_id`: Foreign key to recipient.csv
  - `date`: Last activity timestamp
  - `meaningful_messages`: Count of non-system messages
  - `read`: Read status
  - `snippet`: Preview of last message
  - `snippet_type`: Type code for snippet
  - `unread_count`: Number of unread messages
  - `archived`: Archive status
  - `pinned_order`: Pin position (if pinned)

### 4. **message_fts.csv** (Full Text Search)
- **Lines**: 106,532
- **Purpose**: Indexed message content for search
- **Key Fields**:
  - `body`: Message text content
  - `thread_id`: Foreign key to thread.csv
- **Note**: Contains message text stripped of formatting

### 5. **reaction.csv** (Message Reactions)
- **Lines**: 17,121
- **Purpose**: Stores emoji reactions to messages
- **Key Fields**:
  - `_id`: Primary key
  - `message_id`: Foreign key to signal.csv
  - `author_id`: Foreign key to recipient.csv (who reacted)
  - `emoji`: The reaction emoji
  - `date_sent`, `date_received`: Timestamps

### 6. **groups.csv** (Group Information)
- **Lines**: 795
- **Purpose**: Stores group chat metadata
- **Key Fields**:
  - `_id`: Primary key
  - `group_id`: Unique group identifier
  - `recipient_id`: Foreign key to recipient.csv
  - `title`: Group name
  - `active`: Active status
  - `master_key`: Encrypted group key
  - `decrypted_group`: Contains encrypted group metadata

### 7. **group_membership.csv** (Group Members)
- **Lines**: 613
- **Purpose**: Maps users to groups
- **Key Fields**:
  - `_id`: Primary key
  - `group_id`: Foreign key to groups.csv
  - `recipient_id`: Foreign key to recipient.csv
  - `endorsement`: Member endorsement data

### 8. **identities.csv** (Cryptographic Identities)
- **Lines**: 455
- **Purpose**: Stores Signal identity keys for contacts
- **Key Fields**:
  - `address`: Account identifier (ACI or PNI)
  - `identity_key`: Public identity key
  - `verified`: Verification status
  - **Sensitive**: Contains cryptographic keys

## Data Relationships

```
thread.csv (conversations)
    ↓ recipient_id
recipient.csv (contacts/groups) ← from_recipient_id/to_recipient_id ← signal.csv (messages)
    ↑ recipient_id                                                        ↓ _id
groups.csv (group info)                                             reaction.csv
    ↓ group_id                                                          ↑ message_id
group_membership.csv ← recipient_id

message_fts.csv ← thread_id → thread.csv
```

## Privacy Considerations

1. **Phone Numbers**: Stored in E.164 format in recipient.csv
2. **Contact Names**: Both Signal profile names and system contact names
3. **Message Content**: Full text in signal.csv and message_fts.csv
4. **Group Membership**: Reveals social connections
5. **Timestamps**: Precise timing of all communications
6. **Cryptographic Keys**: Identity keys that could identify accounts

## Notes for Processing

1. The main message content is in `signal.csv`, not a separate `message.csv`
2. Messages with empty `body` field likely contain media attachments
3. Recipient ID 2 appears to be the backup owner (self)
4. Recipient ID 3 is Signal system messages
5. Message type codes indicate different content types (text, media, etc.)
6. Group messages can be identified by checking if thread.recipient_id points to a group recipient (type=2)