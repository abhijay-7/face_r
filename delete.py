import pickle

def delete_person(name_to_delete, embedding_path='trained_model/face_embeddings.pkl'):
  
    with open(embedding_path, 'rb') as f:
        embeddings, names = pickle.load(f)

   
    filtered = [(e, n) for e, n in zip(embeddings, names) if n != name_to_delete]

    if not filtered:
        print("No remaining data after deletion!")
        return

  
    new_embeddings, new_names = zip(*filtered)

 
    with open(embedding_path, 'wb') as f:
        pickle.dump((list(new_embeddings), list(new_names)), f)

    print(f"Deleted  data for '{name_to_delete}' and updated embedding file.")

if __name__ == "__main__":
    delete_person("001udca_abhijay")  
