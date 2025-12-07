from torch.utils.data import DataLoader

from model.parser import Parser
from model.dataset import ParsingDataset, parsing_collater
from model.utils import tensors_to_conllu

# Dataloading
train_set = ParsingDataset(
    data_dir = '.\\data\\treebanks\\UD_English-GUM\\en_gum-ud-train.conllu'
)
train_loader = DataLoader(
    train_set, 
    batch_size=64, 
    shuffle=True,
    collate_fn=parsing_collater
)
batch = next(iter(train_loader))
words, trees, lengths = batch

# Model training
embedding_model = 'goldfish-models/eng_latn_1000mb'
model = Parser(
    embedding_model_name=embedding_model,
    reg=1e-2,
    learning_rate=1e-3,
    potential_clamp=20,
    dropout=0
)
loss = model.training_step(batch, 0)
loss.backward()
print(f'Loss = {loss}')

# Tree prediction
y_pred = model.predict(words, lengths)
y_pred_conllu = tensors_to_conllu(words, y_pred, 'debug.conllu')

