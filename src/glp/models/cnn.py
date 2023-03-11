import torch
from torch import nn
import torch.nn.functional as F

# simple one-layer CNN inspired by https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
class CNN_NLP(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        """
        The constructor for CNN_NLP class.

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. Default: False
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            n_classes (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """

        super(CNN_NLP, self).__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits
    
    def train(self, optimizer, train_dataloader, val_dataloader=None, epochs=10):
        """Train the CNN model."""
        
        # Tracking best validation accuracy
        best_accuracy = 0

        # Start training loop
        print("Start training...\n")
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {\
        'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*60)

        for epoch_i in range(epochs):
            # =======================================
            #               Training
            # =======================================

            # Tracking time and loss
            t0_epoch = time.time()
            total_loss = 0

            # Put the model into the training mode
            self.train()

            for step, batch in enumerate(train_dataloader):
                # Load batch to GPU
                b_input_ids, b_labels = tuple(t.to(device) for t in batch)

                # Zero out any previously calculated gradients
                model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = model(b_input_ids)

                # Compute loss and accumulate the loss values
                loss = loss_fn(logits, b_labels)
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Update parameters
                optimizer.step()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            # =======================================
            #               Evaluation
            # =======================================
            if val_dataloader is not None:
                # After the completion of each training epoch, measure the model's
                # performance on our validation set.
                val_loss, val_accuracy = self.evaluate(model, val_dataloader)

                # Track the best accuracy
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch
                print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {\
                val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                
        print("\n")
        print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")

    def evaluate(self, val_dataloader):
        """After the completion of each training epoch, measure the model's
        performance on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled
        # during the test time.
        self.eval()

        # Tracking variables
        val_accuracy = []
        val_loss = []

        # For each batch in our validation set...
        for batch in val_dataloader:
            # Load batch to GPU
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)

            # Compute logits
            with torch.no_grad():
                logits = self(b_input_ids)

            # Compute loss
            loss = loss_fn(logits, b_labels)
            val_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()

            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy

# A simple CNN model inspired by https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/blob/main/src/genomic_benchmarks/models/torch.py
class CNN(nn.Module):
    def __init__(self, number_of_classes, vocab_size, embedding_dim, input_len):
        super(CNN, self).__init__()
        if number_of_classes == 2:
            number_of_output_neurons = 1
            loss = torch.nn.functional.binary_cross_entropy_with_logits
            output_activation = nn.Sigmoid()
        else:
            raise Exception("Not implemented for number_of_classes!=2")
            # number_of_output_neurons = number_of_classes
            # loss = torch.nn.CrossEntropyLoss()
            # output_activation = nn.Softmax(dim=)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=8, bias=True)
        self.norm1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=8, bias=True)
        self.norm2 = nn.BatchNorm1d(16)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=4, bias=True)
        self.norm3 = nn.BatchNorm1d(8)
        self.pool3 = nn.MaxPool1d(2)

        # compute output shape of conv layers
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(self.count_flatten_size(input_len), 512)
        self.lin2 = nn.Linear(512, number_of_output_neurons)
        self.output_activation = output_activation
        self.loss = loss

    def count_flatten_size(self, input_len):
        zeros = torch.zeros([1, input_len], dtype=torch.long)
        x = self.embeddings(zeros)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.flatten(x)
        return x.size()[1]

    def forward(self, x):
        x = self.embeddings(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.output_activation(x)
        return x

    def train_loop(self, dataloader, optimizer, val_dataloader):
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = self(x)
            if y.shape != pred.shape:
                y = y.unsqueeze(1)
                y = y.float()
            loss = self.loss(pred, y)
            loss.backward()
            optimizer.step()


        train_loss, train_correct = self._eval(dataloader=dataloader)
        if val_dataloader != None:
            val_loss, val_correct = self._eval(dataloader=val_dataloader)
            print(f"Train metrics: \n Accuracy: {(100*train_correct):>0.1f}%, Avg loss: {train_loss:>8f} Val accuracy: {(100*val_correct):>0.1f}%, Val avg loss: {val_loss:>8f} \n")
        else:
            print(f"Train metrics: \n Accuracy: {(100*train_correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")


    def train(self, dataloader, epochs, val_datdaloader = None):
        optimizer = torch.optim.Adam(self.parameters())
        for t in range(epochs):
            print(f"Epoch {t}")
            self.train_loop(dataloader, optimizer, val_datdaloader)

    def _eval(self, dataloader):
        size = dataloader.dataset.__len__()
        num_batches = len(dataloader)
        loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X)
                if y.shape != pred.shape:
                    y = y.unsqueeze(1)
                    y = y.float()
                loss += self.loss(pred, y).item()
                correct += (torch.round(pred) == y).sum().item()

        loss /= num_batches
        correct /= size

        return loss, correct

# TODO: update for multiclass classification datasets
    def test(self, dataloader, positive_label = 1):
        size = dataloader.dataset.__len__()
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        tp, p, fp = 0, 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X)
                if y.shape != pred.shape:
                    y = y.unsqueeze(1)
                    y = y.float()
                test_loss += self.loss(pred, y).item()
                correct += (torch.round(pred) == y).sum().item()
                p += (y == positive_label).sum().item() 
                if(positive_label == 1):
                    tp += (y * pred).sum(dim=0).item()
                    fp += ((1 - y) * pred).sum(dim=0).item()
                else:
                    tp += ((1 - y) * (1 - pred)).sum(dim=0).item()
                    fp += (y * (1 - pred)).sum(dim=0).item()

        print("p ", p, "; tp ", tp, "; fp ", fp)
        recall = tp / p
        precision = tp / (tp + fp)
        print("recall ", recall, "; precision ", precision)
        f1_score = 2 * precision * recall / (precision + recall)
        
        print("num_batches", num_batches)
        print("correct", correct)
        print("size", size)

        test_loss /= num_batches
        accuracy = correct / size
        print(f"Test metrics: \n Accuracy: {accuracy:>6f}, F1 score: {f1_score:>6f}, Avg loss: {test_loss:>6f} \n")
        
        return accuracy, f1_score