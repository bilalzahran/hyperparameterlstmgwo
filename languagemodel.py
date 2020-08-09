import time 
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

torch.manual_seed(1111)

class LanguageModel():
    def __init__(self, dropout,embdropout,outdropout,hiddenunit,emsize, corpus, lr, layers, bptt, batch_size, clip = 0.1):
        self.dropout = dropout
        self.embdropout = embdropout
        self.outdropout = outdropout
        self.hiddenunit = hiddenunit
        self.layers = layers
        self.emsize = emsize
        self.clip = clip
        self.lr = lr
        self.batch_size = batch_size
        self.corpus = corpus
        self.bptt = bptt
        self.device = torch.device("cuda")
        self.ntokens = len(corpus.dictionary)
        self.model = model.Model(self.ntokens,self.emsize,self.hiddenunit,self.layers,self.dropout,self.outdropout,self.embdropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
    # ###############################################################################
    # # Training code
    # ###############################################################################

    def repackage_hidden(self,h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)
    
    def get_batch(self,source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target
    
    def evaluate(self,data_source,eval_batch_size):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0.
        ntokens = len(self.corpus.dictionary)
        hidden = self.model.init_hidden(eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, self.bptt):
                data, targets = self.get_batch(data_source, i)
                output, hidden = self.model(data, hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * self.criterion(output_flat, targets).item()
                hidden = self.repackage_hidden(hidden)
        return total_loss / len(data_source)
    
    def train(self,train_data,log_interval,epoch,lr):
        # Turn on training mode which enables dropout.
        self.model.train()
        total_loss = 0.
        start_time = time.time()
        ntokens = len(self.corpus.dictionary)
        hidden = self.model.init_hidden(self.batch_size)
        for batch, i in enumerate(range(0, train_data.size(0) - 1,self.bptt)):
            data, targets = self.get_batch(train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = self.repackage_hidden(hidden)
            self.model.zero_grad()
            output, hidden = self.model(data, hidden)
            loss = self.criterion(output.view(-1, ntokens), targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            for p in self.model.parameters():
                p.data.add_(-lr, p.grad.data)

            total_loss += loss.item()

            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.8f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // self.bptt, lr,
                    elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
                
    def training(self,train_data,val_data,epochs,lr,batch_size):
        best_val_loss = None;
        log_interval = 200;
        try:
            for epoch in range(1, epochs+1):
                epoch_start_time = time.time()
                self.train(train_data,log_interval,epoch,lr)
                val_loss = self.evaluate(val_data,batch_size)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                val_loss, math.exp(val_loss)))
                print('-' * 89)               
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open('model.pt', 'wb') as f:
                        torch.save(self.model, f)
                    best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    lr /= 4.0
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        
        print('Evaluate Data')
        test_loss = self.evaluate(val_data,batch_size)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
                test_loss, math.exp(test_loss)))
        print('=' * 89)
        torch.cuda.empty_cache()
        return math.exp(test_loss)