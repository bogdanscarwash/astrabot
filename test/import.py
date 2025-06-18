import pandas as pd
import json
from datetime import datetime

def transform_to_conversations():
    # Load the data
    messages = pd.read_csv('signal-flatfiles/signal.csv')
    recipients = pd.read_csv('signal-flatfiles/recipient.csv')
    threads = pd.read_csv('signal-flatfiles/thread.csv')
    
    # Create recipient lookup
    recipient_lookup = recipients.set_index('_id')['profile_given_name'].to_dict()
    
    conversations = []
    
    # Group messages by thread
    for thread_id in messages['thread_id'].unique():
        thread_messages = messages[messages['thread_id'] == thread_id].sort_values('date_sent')
        
        if len(thread_messages) < 2:  # Skip single message threads
            continue
            
        conversation = []
        for _, msg in thread_messages.iterrows():
            if pd.notna(msg['body']) and msg['body'].strip():
                sender_name = recipient_lookup.get(msg['from_recipient_id'], 'Unknown')
                conversation.append({
                    "role": "user" if sender_name != "You" else "assistant",
                    "content": msg['body'],
                    "timestamp": msg['date_sent']
                })
        
        if len(conversation) >= 2:
            conversations.append({
                "conversation_id": thread_id,
                "messages": conversation
            })
    
    return conversations


def extract_qa_pairs(conversations):
    qa_pairs = []
    
    for conv in conversations:
        messages = conv['messages']
        for i in range(len(messages) - 1):
            current = messages[i]
            next_msg = messages[i + 1]
            
            # Look for question patterns
            if ('?' in current['content'] or 
                current['content'].lower().startswith(('what', 'how', 'why', 'when', 'where', 'who'))):
                qa_pairs.append({
                    "instruction": current['content'],
                    "response": next_msg['content'],
                    "context": conv['conversation_id']
                })
    
    return qa_pairs

def create_persona_dataset():
    # Filter for your messages only
    your_messages = messages[messages['from_recipient_id'] == 2]  # Assuming 2 is your ID
    
    persona_data = []
    for _, msg in your_messages.iterrows():
        if pd.notna(msg['body']) and len(msg['body']) > 10:
            persona_data.append({
                "input": "Respond in the style of the user:",
                "output": msg['body'],
                "instruction": "Generate a response that matches this communication style"
            })
    
    return persona_data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def cluster_by_topics(conversations, n_clusters=10):
    # Extract all message content
    all_text = []
    conv_mapping = []
    
    for conv in conversations:
        conv_text = " ".join([msg['content'] for msg in conv['messages']])
        all_text.append(conv_text)
        conv_mapping.append(conv)
    
    # Vectorize and cluster
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(all_text)
    
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(X)
    
    # Group by clusters
    clustered_convs = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in clustered_convs:
            clustered_convs[cluster_id] = []
        clustered_convs[cluster_id].append(conv_mapping[i])
    
    return clustered_convs