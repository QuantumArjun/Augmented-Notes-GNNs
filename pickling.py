import pickle

# Read the contents of the text file into a string
with open('new_node.txt', 'r') as file:
    text = file.read()

# Serialize the text string using pickle
with open('fulldata.pickle', 'wb') as file:
    pickle.dump(text, file)
