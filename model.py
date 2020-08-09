import torch.nn as nn

class Model(nn.Module):
    
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout, outdropout, embdropout):
        super(Model, self).__init__()
        self.outdrop = nn.Dropout(outdropout)
        self.embdropout = nn.Dropout(embdropout)
        self.embedding = nn.Embedding(ntoken, ninp)
        self.lstm = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.full = nn.Linear(nhid, ntoken)
        #Use Tie-Weight
        self.embedding.weight = self.full.weight
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange,initrange)
        self.full.bias.data.zero_()
        self.full.weight.data.uniform_(-initrange,initrange)
        
    def forward(self, input, hidden):
        emb = self.embdropout(self.embedding(input))
        output, hidden = self.lstm(emb, hidden)
        output = self.outdrop(output)
        decoded = self.full(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1),decoded.size(1)), hidden
    
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))