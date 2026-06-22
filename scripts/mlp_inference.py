import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)


class SentDataset(Dataset):
    """
    Define a pytorch dataset class that accepts a text path, and optionally label path and
    a vocabulary (depends on your implementation). This class holds all the data and implement
    a __getitem__ method to be used by a Python generator object or other classes that need it.

    DO NOT shuffle the dataset here, and DO NOT pad the tensor here.
    """
    def __init__(self, train_path, label_path=None, vocab=None):
        """
        Read the content of vocab and text_file
        Args:
            vocab (string): Path to the vocabulary file.
            text_file (string): Path to the text file.
        """
        with open(train_path, mode="r", encoding="utf-8") as training_text:
            self.training_text = [sequence.strip() for sequence in training_text if sequence.strip()]
        if label_path:
            with open(label_path, mode="r", encoding="utf-8") as labels:
                self.labels = labels.read().split("\n")
        else:
            self.labels = [0] * len(self.training_text)
        self.vocab = vocab if vocab else get_bigrams(self.training_text)

    def vocab_size(self):
        """
        A function to inform the vocab size. The function returns two numbers:
            num_vocab: size of the vocabulary
        """
        num_vocab = len(self.vocab) + 1
        return num_vocab
    
    def __len__(self):
        """
        Return the number of instances in the data
        """
        return len(self.training_text)

    def __getitem__(self, i):
        """
        Return the i-th instance in the format of:
            (text, label)
        Text and label should be encoded according to the vocab (word_id).

        DO NOT pad the tensor here, do it at the collator function.
        """
        tokens = self.training_text[i].split()
        text = torch.tensor(data=[self.vocab.get((tokens[i], tokens[i + 1]), 0) for i in range(len(tokens)-1)], dtype=torch.long)
        label = torch.tensor(data=[int(self.labels[i])], dtype=torch.int8)
        
        return text, label


class Model(nn.Module):
    """
    Define your model here
    """
    def __init__(self, num_vocab):
        super().__init__()
        # define your model attributes here
        self.embed_layer = nn.Embedding(num_embeddings=num_vocab, embedding_dim=14, padding_idx=0)
        self.input_projection = nn.Linear(in_features=14, out_features=32, bias=True)
        self.activation_layer = nn.ReLU()
        self.dropout_layer = nn.Dropout()
        self.output_projection = nn.Linear(in_features=32, out_features=1, bias=True)
        self.classification = nn.Sigmoid()

    def forward(self, x):
        # define the forward function here
        embeddings = self.embed_layer(x)
        e_mean = embeddings.mean(dim=1)
        
        h1 = self.activation_layer(self.input_projection(e_mean))
        h1 = self.dropout_layer(h1)
        
        logits = self.output_projection(h1)
        classes = self.classification(logits)
        
        return classes



def collator(batch):
    """
    Define a function that receives a list of (text, label) pair
    and return a pair of tensors:
        texts: a tensor that combines all the text in the mini-batch, pad with 0
        labels: a tensor that combines all the labels in the mini-batch
    """
    sequences, labels = zip(*batch)
    lengths = [len(sequence) for sequence in sequences]
    longest_sequence_length = max(lengths)
    padded_texts =  torch.stack([
        F.pad(sequence, pad=(0, longest_sequence_length - len(sequence)), mode='constant', value=0)
        for sequence in sequences
    ])
    labels = torch.stack(labels).to(dtype=torch.float)
    
    return padded_texts, labels

def get_bigrams(text):
    vocab = []
    for sequence in text:
        tokens = sequence.strip().split()
        for i in range(len(tokens) - 1):
            vocab.append((tokens[i], tokens[i + 1]))
    
    vocab = {bigram: idx + 1 for idx, bigram in enumerate(set(vocab))}
    return vocab

def train(model, dataset, vocab, batch_size, learning_rate, num_epoch, device='cpu', model_path=None):
    """
    Complete the training procedure below by specifying the loss function
    and optimizers with the specified learning rate and specified number of epoch.
    
    Do not calculate the loss from padding.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    # assign these variables
    criterion = nn.BCELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            texts = data[0].to(device)
            labels = data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # do forward propagation
            predictions = model(texts)
            # calculate the loss
            loss = criterion(predictions, labels)
            # do backward propagation
            loss.backward()
            # do the parameter optimization
            optimizer.step()
            # calculate running loss value for non padding
            running_loss += loss.item()
            # print loss value every 100 iterations and reset running loss
            if step % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0

    end = datetime.datetime.now()
    
    # define the checkpoint and save it to the model path
    # tip: the checkpoint can contain more than just the model
    checkpoint = {"model": model.state_dict(), "vocab": vocab};
    torch.save(checkpoint, model_path)

    print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))


def test(model, dataset, thres=0.5, device='cpu'):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=20, collate_fn=collator, shuffle=False)
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            texts = data[0].to(device)
            classifications = model(texts)
            predictions.extend((classifications >= thres).long().squeeze(-1).cpu().tolist())
    predictions = [str(p) for p in predictions]
    return predictions

def main(args):
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)
    else:
        device_str = 'cpu'
    device = torch.device(device_str)
    
    assert args.train or args.test, "Please specify --train or --test"
    if args.train:
        assert args.label_path is not None, "Please provide the labels for training using --label_path argument"
        dataset = SentDataset(args.text_path, args.label_path)
        num_vocab = dataset.vocab_size()
        model = Model(num_vocab).to(device)
        
        # specify these hyper-parameters
        batch_size = 64
        learning_rate = 1e-3
        num_epochs = 10

        train(model, dataset, dataset.vocab, batch_size, learning_rate, num_epochs, device, args.model_path)
    if args.test:
        assert args.model_path is not None, "Please provide the model to test using --model_path argument"
        
        # load the checkpoint
        checkpoint = torch.load(args.model_path)
        # create the test dataset object using SentDataset class
        dataset = SentDataset(train_path=args.text_path, vocab=checkpoint["vocab"])
        # initialize and load the model
        model = Model(dataset.vocab_size()).to(device)
        model.load_state_dict(checkpoint["model"])
        # run the prediction
        preds = test(model, dataset, 0.5, device)

        # write the output
        with open(args.output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(preds))
    print('\n==== All done ====')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', help='path to the text file')
    parser.add_argument('--label_path', default=None, help='path to the label file')
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    parser.add_argument('--model_path', required=True, help='path to the model file during testing')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)
