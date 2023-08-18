import pickle

def save_response_to_pkl(chat):
    with open("chat_logs/chat_log2.pkl", 'wb') as file:
        pickle.dump(chat, file)


def save_response_to_txt(chat):        
    with open("chat_logs/chat_log2.txt", "w", encoding="utf-8") as file:
        for chat_entry in chat:
            role = chat_entry["role"]
            content = chat_entry["content"]
            file.write(f"{role}: {content}\n")

