import torch
import re


def print_model_parameters(model: torch.nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


pat_lt_gt = re.compile(r'&lt;[\s\S]*?&gt;')
pat_big_bracket = re.compile(r'{{[\s\S]*?}}')
pat_square_bracket = re.compile(r'(\[\[[\s\S]*?\]\])')


def square_bracket(string):
    return string.split('|')[-1][:-2] if '|' in string else string.strip('[').strip(']')


def preprocessing_infobox(infobox):
    data = []
    for p in infobox:
        if p[1] == '':
            continue
        c = p[1]
        c = re.sub(pat_lt_gt, '', c)
        c = re.sub(pat_big_bracket, '', c)
        for string in re.findall(pat_square_bracket, c):
            c = c.replace(string, square_bracket(string))
        c = c.replace('\n', '').replace('\t', '')
        if re.search(r'[a-z]', p[0]) is None or re.search(r'[a-z]', c) is None or len(c.split(' ')) > 25:
            continue
        data.extend([p[0], c])
    return data
