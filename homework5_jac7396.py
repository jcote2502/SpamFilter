############################################################
# CMPSC442: Classification
############################################################

student_name = "Justin Cote"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import email
import email.iterators
from collections import Counter
import math
import os

############################################################
# Section 1: Spam Filter
############################################################

def load_tokens(email_path):
    with open(email_path, 'r', encoding='utf-8') as curr_file:
        message = email.message_from_file(curr_file)
    tokens = []
    for line in email.iterators.body_line_iterator(message):
        line_tokens = line.split()
        tokens.extend(line_tokens)
    return tokens

def log_probs(email_paths, smoothing):
    word_counts = Counter()
    vocab_dict = set()
    for path in email_paths:
        tokens = load_tokens(path)
        word_counts.update(tokens)
        vocab_dict.update(tokens)
    vocab_dict.add('<UNK>')
    total_words = sum(word_counts.values())
    log_probs_dict = {}
    for word in vocab_dict:
        count = word_counts[word]
        smoothed_probability = (count+smoothing) / (total_words + smoothing * (len(vocab_dict)+1))
        log_probs_dict[word] = math.log(smoothed_probability)
    return log_probs_dict


class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir, smoothing):
        self.spam_log_probs = log_probs([os.path.join(spam_dir, filename) for filename in os.listdir(spam_dir)], smoothing)
        self.ham_log_probs = log_probs([os.path.join(ham_dir, filename) for filename in os.listdir(ham_dir)], smoothing)
        num_spam = len(os.listdir(spam_dir))
        num_ham = len(os.listdir(ham_dir))
        total_emails = num_spam + num_ham
        self.p_spam = num_spam / total_emails
        self.p_ham = num_ham / total_emails
        self.intersection = set(self.spam_log_probs.keys()).intersection(self.ham_log_probs.keys())
    
    def is_spam(self, email_path):
        email_tokens = load_tokens(email_path)
        spam_score = sum(self.spam_log_probs.get(token, self.spam_log_probs['<UNK>']) for token in email_tokens)
        ham_score = sum(self.ham_log_probs.get(token, self.ham_log_probs['<UNK>']) for token in email_tokens)
        return spam_score > ham_score

    def most_indicative_spam(self, n):
        indicative_words = [(word, self.spam_log_probs.get(word, self.spam_log_probs['<UNK>']) - self.ham_log_probs.get(word, self.ham_log_probs['<UNK>'])) for word in self.intersection]
        indicative_words.sort(key=lambda x: x[1], reverse=True)
        return [word[0] for word in indicative_words[:n]]

    def most_indicative_ham(self, n):
        indicative_words = [(word, self.ham_log_probs.get(word, self.ham_log_probs['<UNK>']) - self.spam_log_probs.get(word, self.spam_log_probs['<UNK>'])) for word in self.intersection]
        indicative_words.sort(key=lambda x: x[1], reverse=True)
        return [word[0] for word in indicative_words[:n]]

