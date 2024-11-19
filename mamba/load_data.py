import json

def load_train_texts():
    train_texts = []
    
    with open("../data/train/fingpt_sentiment_train.jsonl", "r") as f:
        
        for line in f:
            value = json.loads(line.strip())  # Parse each JSON line
            question = value.get("context", "")
            contexts = value.get("target", [])
            
            train_texts.append({"prompt": question, "completion": contexts})
            
    return train_texts


def load_test_texts():
    train_texts = []
    
    with open("data/train/fingpt_sentiment_train.jsonl", "r") as f:
        for line in f:
            value = json.loads(line.strip())  # Parse each JSON line
            question = value.get("context", "")
            #contexts = " ".join(value.get("CONTEXTS", []))
            train_texts.append(f"Question: {question}\nContexts: {contexts}\nAnswer: ")
            
    return train_texts


def load_test_answer_texts():
    test_answers = []
    
    with open("data/train/fingpt_sentiment_train.jsonl", "r") as f:
        for line in f:
            value = json.loads(line.strip())  # Parse each JSON line
            test_answers.append(value.get("final_decision", ""))
            
    return test_answers
