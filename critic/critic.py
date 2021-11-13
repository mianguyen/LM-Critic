import sys
import torch
import random
import hashlib
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, '.')
from critic.perturbations import get_local_neighbors_char_level, get_local_neighbors_word_level
from utils.spacy_tokenizer import spacy_tokenize_gec


model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
model.cuda()
print (f'Loaded {model_name}')


def get_loss(input_ids, attention_mask, labels):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lm_logits = outputs["logits"]
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(bsize, seqlen-1)
            loss = (loss * shift_mask).sum(dim=1) #[bsize, ]
        return loss

def get_inputs(sents):
    if model_name == "gpt2":
      tokenizer.pad_token = tokenizer.eos_token
      _sents = [tokenizer.bos_token + s for s in sents]
      inputs = tokenizer(_sents, return_tensors="pt", padding=True)
    elif model_name == "xlnet-large-cased":
      _sents = [s + tokenizer.sep_token + tokenizer.cls_token for s in sents]
      inputs = tokenizer(_sents, return_tensors="pt", padding=True)
    elif model_name == "bert-large-cased" or model_name == "roberta-base":
      inputs = tokenizer(sents, return_tensors="pt", padding=True)
    return inputs

def run_model(sents, cuda=True, model_name=None):
    assert isinstance(sents, list)
    inputs = get_inputs(sents)
    
    if cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    loss = get_loss(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])
    logps = - loss.detach()
    assert logps is not None
    return logps


def critic_char_level_only(sent, verbose=1, cuda=True, fp16=True, seed='auto', n_samples=100):
    if seed == 'auto':
        seed = int(hashlib.md5(sent.encode()).hexdigest(), 16) % (2**32) #Seed must be between 0 and 2**32 - 1
    if verbose > 1:
        print ('seed', seed)
    np.random.seed(seed); random.seed(seed)
    is_good = True
    for _ in range(1):
        sent_perturbations = get_local_neighbors_char_level(sent, max_n_samples=n_samples)
        if verbose > 1:
            print ("#sent_perturbations (char-level)", len(sent_perturbations))
        sents = [sent] + list(sent_perturbations)
        if fp16:
            with torch.cuda.amp.autocast():
                logps = run_model(sents, cuda)
        else:
            logps = run_model(sents, cuda)
        if logps is None:
            if verbose:
                print ('Invalid input. Maybe the sentence is too long.')
            return None
        best_idx = int(logps.argmax())
        if best_idx != 0:
            is_good = False
            break
    if verbose:
        if is_good:
            print ('Good! Your sentence log(p) = {:.3f}'.format(float(logps[0])))
        else:
            print ('Bad! Your sentence log(p) = {:.3f}'.format(float(logps[0])))
            print ('Neighbor sentence with highest log(p): {} (= {:.3f})'.format(sents[best_idx], float(logps[best_idx])))
    counter_example = None
    if not is_good:
        counter_example = [sents[best_idx], float(logps[best_idx])]
    return is_good, float(logps[0]), counter_example


def critic(sent, verbose=1, cuda=True, fp16=True, seed='auto', n_samples=100, word_level_mode='refine'):
    if seed == 'auto':
        seed = int(hashlib.md5(sent.encode()).hexdigest(), 16) % (2**32) #Seed must be between 0 and 2**32 - 1
    if verbose > 1:
        print ('seed', seed)
    np.random.seed(seed); random.seed(seed)
    sent_toked = spacy_tokenize_gec(sent)
    is_good = True
    for _ in range(1):
        sent_perturbations_w, orig_sent = get_local_neighbors_word_level(sent_toked, max_n_samples=n_samples//2, mode=word_level_mode)
        sent_perturbations_c = get_local_neighbors_char_level(orig_sent, max_n_samples=n_samples//2)
        if verbose > 1:
            print ("#sent_perturbations (char-level)", len(sent_perturbations_c))
            print ("#sent_perturbations (word-level)", len(sent_perturbations_w))
        sents = [orig_sent] + list(sent_perturbations_c.union(sent_perturbations_w))
        if fp16:
            with torch.cuda.amp.autocast():
                logps = run_model(sents, cuda)
        else:
            logps = run_model(sents, cuda)
        if logps is None:
            if verbose:
                print ('Invalid input. Maybe the sentence is too long.')
            return None
        best_idx = int(logps.argmax())
        if best_idx != 0:
            is_good = False
            break
    if verbose:
        if is_good:
            print ('Good! Your sentence log(p) = {:.3f}'.format(float(logps[0])))
        else:
            print ('Bad! Your sentence log(p) = {:.3f}'.format(float(logps[0])))
            print ('Neighbor sentence with highest log(p): {} (= {:.3f})'.format(sents[best_idx], float(logps[best_idx])))
    counter_example = None
    if not is_good:
        counter_example = [sents[best_idx], float(logps[best_idx])]
    return is_good, float(logps[0]), counter_example




if __name__ == '__main__':
    while True:
        sent = input("Enter a sentence: ")
        _ = critic(sent)
