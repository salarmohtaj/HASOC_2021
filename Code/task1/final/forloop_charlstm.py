embedding_size1 = [20,50,100,300]
hidden_size1 = [64, 128, 256]
dropout1 = [float(0.25), float(0.5), float(0.75)]
learning_rate1 = [float(0.01), float(0.003), float(0.001)]
for embedding_size in embedding_size1:
    for hidden_size in hidden_size1:
        for dropout in dropout1:
            for learning_rate in learning_rate1:
                print(embedding_size, hidden_size, dropout, learning_rate)
                exec(open("character_LSTM.py").read())
                print(embedding_size, hidden_size, dropout, learning_rate)
                exec(open("character_LSTM.py").read())
