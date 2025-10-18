from collections import Counter
import json

class BytePairEncodingTokenizer:
  def __init__(self,word_split = 'Ġ'):
    self.vocab={}
    self.inverse_vocab={}
    self.tokens=[]
    self.bpe_pairs={}
    self.word_split = word_split

  def set_vocabulary(self,word_split,special_tokens):
    # Step 1: Setting the unique characters
    unique_characters=[chr(i) for i in range(256)]

    # Step 2: Word split characters
    if word_split not in unique_characters:
      unique_characters.append(self.word_split)

    # Step 3: Special tokens
    if special_tokens:
      unique_characters.extend(self.special_tokens)

    return unique_characters

  def get_max_freq_pair(self,tokens):
    pairs=[]
    # Step 1: Gettkng all the token pairs-> (token[i],token[i+1])
    for index in range(len(tokens)-1):
      pairs.append((tokens[index],tokens[index+1]))

    # Step 2: Getting the count of occurences of each of token pairs
    pairs_counts=Counter(pairs)

    # Step 3: Get the token pair whose count is the highest
    max_pair=max(pairs_counts.items(),key=lambda x: x[1])[0]
    return max_pair

  def merge_tokens(self,tokens,max_pair,new_pair_id):
    # In the tokens, check the presence of occurence of max_pair and if exists then replace max_pair with new_pair_id
    # Eg: tokens=[87,76,44,25,38,44,25,19], max_pair=[44,25],  new_pair_id=123
    # Output: [87,76,123,38,123,19]
    new_tokens=[]
    i=0
    while i<=len(tokens)-1:
      if i==len(tokens)-1:
        new_tokens.append(tokens[i])
        break

      elif (tokens[i],tokens[i+1])==max_pair:
        new_tokens.append(new_pair_id)
        i+=2

      else:
        new_tokens.append(tokens[i])
        i+=1
    return new_tokens

  def train(self,text,vocab_size,special_tokens):
    if vocab_size<=258:
      raise ValueError('Please enter a vocab size greater than 258 since this defines the basic set of characters')
    self.special_tokens = special_tokens

    # Setting the vocabulary
    vocab = self.set_vocabulary(self.word_split,self.special_tokens)
    for index,character in enumerate(vocab):
      self.vocab[index]=character
      self.inverse_vocab[character]=index

    # Transforming the text
    ## Step 1: Replacing all thw white-space character
    processed_text=[]
    for index,char in enumerate(text):
      if index!=0 and char==' ':
        processed_text.append(self.word_split)
      if char!=' ':
        processed_text.append(char)
    processed_text="".join(processed_text)

    ## Step 2: Getting the numerical form of token
    self.tokens=[]
    for char in processed_text:
      self.tokens.append(self.inverse_vocab[char])

    ## Step 3: BPE-algorithm
    vocab_length=len(self.vocab)
    for i in range(vocab_length,vocab_size):
      max_pair=self.get_max_freq_pair(self.tokens)
      if max_pair is None:
        break
      self.bpe_pairs[max_pair]=i
      self.tokens=self.merge_tokens(self.tokens,max_pair,i)

    ## Step 4: Update vocab with BPE
    for pair,new_index in self.bpe_pairs.items():
      merged_token=self.vocab[pair[0]]+self.vocab[pair[1]]
      self.vocab[new_index]=merged_token
      self.inverse_vocab[merged_token]=new_index

  def encode(self,text):
    # Step 1: Basically tokens are split into words. Replace all the occurences of "\n" to " <NEWLINE> ". This is to avoid splitting issues.
    tokens_split=text.replace('\n',' <NEWLINE> ').split()
    tokens=[]
    for i in tokens_split:
      if i=='<NEWLINE>':
        tokens.append('\n')
      else:
        tokens.append(i)

    # Step 2: Cleaning of tokens
    ## Eg: 'This is a ball' will be tokenized as ['The','Ġis','Ġa', 'Ġball']
    # Ensures that all the tokens in a line other than the first one will be prefixed with "Ġ" to show the word boundaries
    tokens_cleaned=[]
    for index,token in enumerate(tokens):
      if index>0 and not token.startswith('\n'):
        tokens_cleaned.append(self.word_split+token)
      else:
        tokens_cleaned.append(token)

    # Step 3: Getting the corresponding token IDs from the cleaned tokens
    ## Checks whether tokens exist in the vocabulary. If not, then perform BPE tokenization of the token
    token_ids=[]
    for token in tokens_cleaned:
      if token in self.inverse_vocab.keys():
        token_ids.append(self.inverse_vocab[token])
      else:
        token_ids.extend(self.tokenize_using_bpe(token))
    return token_ids

  def tokenize_using_bpe(self,token):
    # Step 1: Mapping the tokens to their IDs from the vocabulary
    token_ids=[]
    for char in token:
      if char in self.inverse_vocab.keys():
        token_ids.append(self.inverse_vocab[char])
      else:
        token_ids.append(None)

    # Step 2: Check whether token does not exist in Vocabulary- In that case stop
    if None in token_ids:
      token_dict=dict(zip(token_ids,token))
      missing_characters=[]
      for id,ch in token_dict.items():
        if id is None:
          missing_characters.append(ch)
      raise ValueError(f"No token IDs found for the characters:{missing_characters}")

    # Step 3: Now merging
    can_merge=True
    while can_merge and len(token_ids)>1:
      can_merge=False
      i=0
      new_tokens=[]
      """
      Check whether the token pair is part of bpe_pairs occured during training,
      If yes, index = index + 2, else index = index + 1.
      This iteration occurs until there exists no merging exists for all the tokens in token_ids.
      No merging exists means that there are no more possible keys to merge in bpe_pairs.
      """
      while i<len(token_ids)-1:
        pair=(token_ids[i],token_ids[i+1])
        if pair in self.bpe_pairs.keys():
          pair_id=self.bpe_pairs[pair]
          new_tokens.append(pair_id)
          i+=2
          can_merge=True
        else:
          new_tokens.append(token_ids[i])
          i+=1
      if i<len(token_ids):
        new_tokens.append(token_ids[i])
      token_ids=new_tokens

    return token_ids

  def decode(self,token_ids):
    # Step 1: Check whether there are non-existing token IDs
    non_existing_ids=[]
    for id in token_ids:
      if id not in self.vocab.keys():
        non_existing_ids.append(id)
    if len(non_existing_ids)>0:
      raise ValueError(f"No token found for the token IDs:{non_existing_ids}")

    # Step 2: Decoding- Check whether text corresponding to token ID starts with word_split-symbol('Ġ'). If yes replace word_split-symbol with " " else just append the text to string
    final=""
    for id in token_ids:
      text=self.vocab[id]
      if text.startswith(self.word_split):
        final+=" "+text[1:]
      else:
        final+=""+text

    return final

  def save_bpe_vocab_and_merges(self,vocab_path,bpe_path):
    with open(vocab_path,'w',encoding='utf-8') as f:
      json.dump(self.vocab,f,ensure_ascii=False, indent=2)
    with open(bpe_path,'w',encoding='utf-8') as f:
      json.dump([{'pair':list(pair),'id':id } for pair,id in self.bpe_pairs.items()],f,
                ensure_ascii=False, indent=2)

  def load_bpe_vocab_and_merges(self,vocab_path,bpe_path):
    with open(vocab_path,'r',encoding='utf-8') as f:
      loaded_vocab=json.load(f)
      self.vocab = {int(id):token for id,token in loaded_vocab.items()}
      self.inverse_vocab={token:int(id) for id,token in self.vocab.items()}
    with open(bpe_path,'r',encoding='utf-8') as f:
      bpe=json.load(f)
      for merge in bpe:
        self.bpe_pairs[tuple(merge['pair'])]=merge['id']

if __name__ == "__main__":
  pass