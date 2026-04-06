import re
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# 1. Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 2. Tiny parallel dataset
# ---------------------------
pairs = [
    ("i am a student", "저는 학생이에요"),
    ("i am hungry", "저는 배고파요"),
    ("i am tired", "저는 피곤해요"),
    ("i am happy", "저는 행복해요"),
    ("i am sorry", "미안해요"),
    ("i like coffee", "저는 커피를 좋아해요"),
    ("i like tea", "저는 차를 좋아해요"),
    ("i like music", "저는 음악을 좋아해요"),
    ("i like korean food", "저는 한식을 좋아해요"),
    ("i need help", "도움이 필요해요"),
    ("i go to school", "저는 학교에 가요"),
    ("i go home", "저는 집에 가요"),
    ("he is my friend", "그는 제 친구예요"),
    ("she is my sister", "그녀는 제 여동생이에요"),
    ("this is my book", "이것은 제 책이에요"),
    ("that is a cat", "저것은 고양이예요"),
    ("where are you", "당신은 어디에 있나요"),
    ("how are you", "어떻게 지내세요"),
    ("what is this", "이것은 무엇인가요"),
    ("what is your name", "이름이 무엇인가요"),
    ("my name is minsu", "제 이름은 민수예요"),
    ("thank you", "감사합니다"),
    ("you are welcome", "천만에요"),
    ("good morning", "좋은 아침이에요"),
    ("good afternoon", "안녕하세요"),
    ("good night", "안녕히 주무세요"),
    ("see you tomorrow", "내일 봐요"),
    ("open the door", "문을 여세요"),
    ("close the window", "창문을 닫으세요"),
    ("please sit down", "앉으세요"),
    ("please come in", "들어오세요"),
    ("i am studying", "저는 공부하고 있어요"),
    ("i am eating", "저는 먹고 있어요"),
    ("i am reading a book", "저는 책을 읽고 있어요"),
    ("i am watching tv", "저는 텔레비전을 보고 있어요"),
    ("do you like coffee", "커피를 좋아하나요"),
    ("do you speak english", "영어를 할 수 있나요"),
    ("i do not know", "저는 몰라요"),
    ("i understand", "이해했어요"),
    ("i do not understand", "이해하지 못했어요"),
    ("where is the station", "역이 어디에 있나요"),
    ("where is the bathroom", "화장실이 어디에 있나요"),
    ("how much is this", "이것은 얼마인가요"),
    ("it is delicious", "맛있어요"),
    ("it is cold today", "오늘은 추워요"),
    ("it is hot today", "오늘은 더워요"),
    ("i am at home", "저는 집에 있어요"),
    ("i am at school", "저는 학교에 있어요"),
    ("the food is good", "음식이 맛있어요"),
    ("the weather is nice", "날씨가 좋아요"),
]

random.shuffle(pairs)

# ---------------------------
# 3. Tokenization
# ---------------------------
def tokenize_en(text): # 
    text = text.lower().strip()
    text = re.sub(r"([?.!,])", r" \1 ", text)
    text = re.sub(r"[^a-z?.!,']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def tokenize_ko(text):
    return text.strip().split()

# ---------------------------
# 4. Vocabulary
# ---------------------------
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

class Vocab: # 
    def __init__(self, tokenized_texts, min_freq=1):
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        self.itos = SPECIAL_TOKENS[:] #
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.itos:
                self.itos.append(token)

        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def encode(self, tokens):
        return [self.stoi.get(tok, self.stoi["<unk>"]) for tok in tokens]

    def decode(self, ids):
        tokens = []
        for idx in ids:
            token = self.itos[idx]
            if token == "<eos>":
                break
            if token not in ["<pad>", "<sos>"]:
                tokens.append(token)
        return tokens

src_tokenized = [tokenize_en(src) for src, _ in pairs]
trg_tokenized = [tokenize_ko(trg) for _, trg in pairs]

src_vocab = Vocab(src_tokenized)
trg_vocab = Vocab(trg_tokenized)

SRC_PAD_IDX = src_vocab.stoi["<pad>"]
TRG_PAD_IDX = trg_vocab.stoi["<pad>"]
TRG_SOS_IDX = trg_vocab.stoi["<sos>"]
TRG_EOS_IDX = trg_vocab.stoi["<eos>"]

# ---------------------------
# 5. Dataset
# ---------------------------
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, trg_vocab):
        self.data = pairs
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, trg_text = self.data[idx]

        src_tokens = tokenize_en(src_text)
        trg_tokens = tokenize_ko(trg_text)

        src_ids = [self.src_vocab.stoi["<sos>"]] + self.src_vocab.encode(src_tokens) + [self.src_vocab.stoi["<eos>"]]
        trg_ids = [self.trg_vocab.stoi["<sos>"]] + self.trg_vocab.encode(trg_tokens) + [self.trg_vocab.stoi["<eos>"]]

        return torch.tensor(src_ids), torch.tensor(trg_ids)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)

    src_lengths = torch.tensor([len(x) for x in src_batch], dtype=torch.long)
    trg_lengths = torch.tensor([len(x) for x in trg_batch], dtype=torch.long)

    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=SRC_PAD_IDX)
    trg_padded = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=TRG_PAD_IDX)

    return src_padded, src_lengths, trg_padded, trg_lengths

dataset = TranslationDataset(pairs, src_vocab, trg_vocab)
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# ---------------------------
# 6. Encoder / Decoder / Seq2Seq
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=src.device)

        hidden = self.encoder(src, src_lengths)
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            prediction, hidden = self.decoder(input_token, hidden)
            outputs[:, t, :] = prediction

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = trg[:, t] if teacher_force else top1

        return outputs

INPUT_DIM = len(src_vocab.itos)
OUTPUT_DIM = len(trg_vocab.itos)
EMB_DIM = 64
HID_DIM = 128
DROPOUT = 0.2

encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, DROPOUT, SRC_PAD_IDX)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DROPOUT, TRG_PAD_IDX)
model = Seq2Seq(encoder, decoder).to(device)

# ---------------------------
# 7. Train setup
# ---------------------------
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

def train_one_epoch(model, loader, optimizer, criterion, clip=1.0):
    model.train()
    epoch_loss = 0

    for src, src_lengths, trg, _ in loader:
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output = model(src, src_lengths, trg, teacher_forcing_ratio=0.5)

        output_dim = output.shape[-1]
        output = output[:, 1:, :].reshape(-1, output_dim)
        trg_gold = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg_gold)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)

# ---------------------------
# 8. Inference
# ---------------------------
def translate_sentence(model, sentence, src_vocab, trg_vocab, max_len=20):
    model.eval()

    tokens = tokenize_en(sentence)
    src_ids = [src_vocab.stoi["<sos>"]] + src_vocab.encode(tokens) + [src_vocab.stoi["<eos>"]]

    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_lengths = torch.tensor([len(src_ids)], dtype=torch.long).to(device)

    with torch.no_grad():
        hidden = model.encoder(src_tensor, src_lengths)

    input_token = torch.tensor([trg_vocab.stoi["<sos>"]], dtype=torch.long).to(device)
    generated_ids = []

    for _ in range(max_len):
        with torch.no_grad():
            prediction, hidden = model.decoder(input_token, hidden)

        top1 = prediction.argmax(1).item()

        if top1 == trg_vocab.stoi["<eos>"]:
            break

        generated_ids.append(top1)
        input_token = torch.tensor([top1], dtype=torch.long).to(device)

    generated_tokens = trg_vocab.decode(generated_ids)
    return " ".join(generated_tokens)

# ---------------------------
# 9. Training loop
# ---------------------------
N_EPOCHS = 200

for epoch in range(1, N_EPOCHS + 1):
    loss = train_one_epoch(model, loader, optimizer, criterion)

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")

        test_sentences = [
            "i am hungry",
            "thank you",
            "where is the bathroom",
            "i like coffee",
            "i am at home",
        ]

        for s in test_sentences:
            pred = translate_sentence(model, s, src_vocab, trg_vocab)
            print(f"EN: {s}")
            print(f"KO: {pred}")
        print("-" * 50)

# ---------------------------
# 10. Final test
# ---------------------------
print("\nFinal test:")
examples = [
    "i am hungry",
    "i am tired",
    "thank you",
    "how are you",
    "where is the station",
    "i like music",
    "i am at school",
]

for s in examples:
    print(f"EN: {s}")
    print(f"KO: {translate_sentence(model, s, src_vocab, trg_vocab)}")
    print()