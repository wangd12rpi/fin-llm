def change_target(x):
    """
    Cleans and normalizes the target value from the input data.
    Handles various formats including JSONL context for XBRL test data.
    """
    # Remove unnecessary characters and whitespace
    x = x.replace('"', '').replace('\n', '').strip()

    # Handle specific cases based on known formats or tags
    # Example: If the file contains specific tag mappings, normalize them here
    if 'positive' in x.lower():
        return 'positive'
    elif 'negative' in x.lower():
        return 'negative'
    elif 'neutral' in x.lower():
        return 'neutral'
    elif 'organization' in x.lower():
        return 'organization'
    elif 'person' in x.lower():
        return 'person'
    elif 'location' in x.lower():
        return 'location'
    
    # Default case: If no specific match, return cleaned string
    return x
