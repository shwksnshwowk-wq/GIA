import torch
from torch.utils.data import Dataset
from typing import List
import random
import pandas as pd
import time
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import psutil
import csv


def can_be_expressed_two(vectors, target, device):
    A = vectors.t().float().to(device)  # Data format: vectors are basis vectors (n, m), and target is the (n1, n2, m) target vector.
    target_temp = target.float().to(device)
    n1, n2, m = target_temp.shape
    target_flattened = target_temp.view(n1 * n2, m)
    q, r = torch.linalg.qr(A, mode='reduced')
    q_t_target = q.t() @ target_flattened.t()
    solution = torch.linalg.solve_triangular(r, q_t_target, upper=True)
    reconstructed_vector_in = A @ solution
    isclose_tensor = torch.isclose(reconstructed_vector_in.t(), target_flattened, atol=1e-3, rtol=1e-3)
    is_close_in = torch.all(isclose_tensor, dim=1)
    is_close_in_first = is_close_in.view(n1, n2)

    A = vectors.float().to(device)
    target_temp = target.float().to(device)
    n1, n2, m = target_temp.shape
    target_flattened = target_temp.view(n1 * n2, m)
    coefficients = torch.linalg.pinv(A.t()) @ target_flattened.t()
    reconstructed_vector_flattened = A.t() @ coefficients
    isclose_tensor = torch.isclose(reconstructed_vector_flattened.t(), target_flattened, atol=1e-3, rtol=1e-3)
    is_close_in = torch.all(isclose_tensor, dim=1)
    is_close_in_second = is_close_in.view(n1, n2)

    is_close_in = torch.logical_or(is_close_in_first, is_close_in_second)
    is_close_in_rows, is_close_in_cols = torch.where(is_close_in)  # "is_close_in_rows" represents the index of the word vector, while "is_close_in_cols" indicates the position of the word vector.
    coefficients_in = solution.t().view(n1, n2, -1)
    reconstructed_vector_in = reconstructed_vector_in.t().view(n1, n2, -1)
    coefficients_in = coefficients_in[is_close_in_rows, is_close_in_cols, :]
    reconstructed_vector_in = reconstructed_vector_in[is_close_in_rows, is_close_in_cols, :]
    return is_close_in_rows, is_close_in_cols, coefficients_in, reconstructed_vector_in


def can_be_expressed(vectors, target, device):
    A = vectors.t().float().to(device)  # Data format: vectors are basis vectors (n, m), and target is the (n1, n2, m) target vector.
    target = target.float().to(device)
    n1, n2, m = target.shape
    target_flattened = target.view(n1 * n2, m)
    q, r = torch.linalg.qr(A, mode='reduced')
    q_t_target = q.t() @ target_flattened.t()
    solution = torch.linalg.solve_triangular(r, q_t_target, upper=True)
    reconstructed_vector_in = A @ solution
    isclose_tensor = torch.isclose(reconstructed_vector_in.t(), target_flattened, atol=1e-3, rtol=1e-3)
    is_close_in = torch.all(isclose_tensor, dim=1)
    is_close_in = is_close_in.view(n1, n2)
    is_close_in_rows, is_close_in_cols = torch.where(is_close_in)    # "is_close_in_rows" represents the index of the word vector, while "is_close_in_cols" indicates the position of the word vector.
    coefficients_in = solution.t().view(n1, n2, -1)
    reconstructed_vector_in = reconstructed_vector_in.t().view(n1, n2, -1)
    coefficients_in = coefficients_in[is_close_in_rows, is_close_in_cols, :]
    reconstructed_vector_in = reconstructed_vector_in[is_close_in_rows, is_close_in_cols, :]
    return is_close_in_rows, is_close_in_cols, coefficients_in, reconstructed_vector_in


def can_be_expressed_one(vectors, target, device):
    vectors = vectors.float().to(device)
    target = target.float().to(device)
    n1, n2, m = target.shape
    target_flattened = target.view(n1 * n2, m)
    coefficients = torch.linalg.pinv(vectors.t()) @ target_flattened.t()
    reconstructed_vector_flattened = vectors.t() @ coefficients
    isclose_tensor = torch.isclose(reconstructed_vector_flattened.t(), target_flattened, atol=1e-3, rtol=1e-3)
    is_close_in = torch.all(isclose_tensor, dim=1)
    is_close_in = is_close_in.view(n1, n2)
    is_close_rows, is_close_cols = torch.where(is_close_in)
    reconstructed_vector_in = reconstructed_vector_flattened.t().view(n1, n2, -1)
    coefficients = coefficients.t().view(n1, n2, -1)
    coefficients = coefficients[is_close_rows, is_close_cols, :]
    reconstructed_vector_in = reconstructed_vector_in[is_close_rows, is_close_cols, :]

    return is_close_rows, is_close_cols, coefficients, reconstructed_vector_in


# Load the datasets
class CustomDataset(Dataset):
    def __init__(self, encodings, labels, sentences):
        self.encodings = encodings
        self.labels = labels
        self.sentences = sentences

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        item['sentence'] = self.sentences[idx]
        return item

    def __len__(self):
        return len(self.labels)


# load data
def preprocess_tsv_cola(file_path, tokenizer):
    df = pd.read_csv(file_path, delimiter='\t', header=None, names=['label', 'sentence_id', 'sentence'])
    df = df[df['label'] == 1]
    sentences = df['sentence'].tolist()
    labels = df['label'].tolist()
    encodings = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = CustomDataset(encodings, labels_tensor, sentences)
    return dataset


def preprocess_tsv_sst(file_path, tokenizer):
    #
    df = pd.read_csv(file_path)
    # choose the sentences and labels
    sentences = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].tolist()
    # preprocess the tokenizer of BERT
    encodings = tokenizer(sentences, truncation=True, padding=True, max_length=512, return_tensors='pt')
    # transfer labels to tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    # Create CustomDataset
    dataset = CustomDataset(encodings, labels_tensor, sentences)
    return dataset


# Detect consecutive identical indices
def has_consecutive_indices(tensors: List[torch.Tensor]) -> torch.Tensor:
    consecutive_indices = []
    for tensor in tensors:
        diff = tensor[1:] - tensor[:-1]
        consecutive_indices.append((diff == 0).any().item())
    return torch.tensor(consecutive_indices, dtype=torch.bool)


# Batch testing of large-scale models
def word_test_by_model(model, tokenizer, data: List[str], batch_size=32, threshold=0.2):
    device = next(model.parameters()).device
    new_data = []
    # Process the data in batches
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        with torch.no_grad():
            # Encode the batch data
            encoded_input = tokenizer(batch_data, return_tensors='pt', padding=True, truncation=True).to(device)
            output = model(**encoded_input)
            logits = output.logits
            # Calculate the probability
            probas_output = torch.nn.functional.softmax(logits, dim=-1)
            # Obtain the index of the "acceptable" category
            acceptable_index = list(model.config.id2label.keys())[
                list(model.config.id2label.values()).index('acceptable')]
            # Extract the probability of the "acceptable" category
            acceptable_scores = probas_output[:, acceptable_index].cpu().tolist()
            # Check whether the probability of the "acceptable" category is greater than the threshold
            for orig_temp_data, score in zip(batch_data, acceptable_scores):
                if score > threshold:
                    new_data.append(orig_temp_data)
    return new_data


def set_random_seed(seed: int = 1):
    # init Pytorch seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # init python random seed
    random.seed(seed)
    # init numpy random seed
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def get_random_batch(train_dataloader):
    # Obtain the batch size of the data loader
    num_batch = len(train_dataloader)
    # Select a random batch index
    batch_num_target = random.randrange(0, num_batch)
    # Initialize a dictionary to store the data of random batches
    random_batch_target = {}
    # Traverse the data loader and find the target batch
    for i, batch in enumerate(train_dataloader):
        if i == batch_num_target:
            random_batch_target = batch
            break
    return random_batch_target



def com_mat(combination: List[torch.Tensor], new_indices: List[int]) -> List[torch.Tensor]:

    if not combination:
        return [torch.tensor([index]) for index in new_indices]
    new_combinations = []
    for tensor in combination:
        for index in new_indices:
            new_tensor = torch.cat((tensor, torch.tensor([index])))
            new_combinations.append(new_tensor)
    return new_combinations



def com_mat_memory_test(combination: List[torch.Tensor], new_indices: List[int]) -> List[torch.Tensor]:
    # Check the current memory usage
    memory_info = psutil.virtual_memory()
    # print(f"Current CPU memory usage: {memory_info.percent}%")
    # Define a CPU memory threshold. If the memory usage exceeds this threshold, then no operation will be executed.
    cpu_memory_threshold = 98  #
    if memory_info.percent > cpu_memory_threshold:
        # print("Insufficient CPU memory to perform the operation. Returning an empty list.")
        return []

    if not combination:
        return [torch.tensor([index]) for index in new_indices]
    new_combinations = []
    for tensor in combination:
        for index in new_indices:
            try:
                # guarantee Tensor
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.tensor(tensor)

                index_tensor = torch.tensor([index])

                new_tensor = torch.cat((tensor, index_tensor))
                new_combinations.append(new_tensor)
            except RuntimeError as e:
                print(f"Caught an error: {e}")
                print("Insufficient memory to perform the operation. Returning an empty list.")
                return []
    return new_combinations



def com_mat_memory_test_save(combination: List[torch.Tensor], new_indices: List[int], batch_size: int = 1000) -> List[
    torch.Tensor]:
    # Check the current memory usage
    memory_info = psutil.virtual_memory()
    # print(f"Current CPU memory usage: {memory_info.percent}%")
    # Define a CPU memory threshold. If the memory usage exceeds this threshold, then no operation will be executed.
    cpu_memory_threshold = 98  #
    if memory_info.percent > cpu_memory_threshold:
        # print("Insufficient CPU memory to perform the operation. Returning an empty list.")
        return []

    if not combination:
        return [torch.tensor([index]) for index in new_indices]
    new_combinations = []
    for tensor in combination:
        for index in new_indices:
            try:
                new_tensor = torch.cat((tensor, torch.tensor([index])))
                new_combinations.append(new_tensor)
                if len(new_combinations) >= batch_size:
                    yield new_combinations
                    new_combinations = []
            except RuntimeError as e:
                print(f"Caught an error: {e}")
                print("Insufficient memory to perform the operation. Returning an empty list.")
                return []
    if new_combinations:
        yield new_combinations


# Count the frequency of occurrence of different lengths in the statistical combination
def count_rows_by_length(csv_file):
    # read csv
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        row_len = []
        for row in reader:
            row_len.append(len(row))
    series = pd.Series(row_len)
    count = series.value_counts().sort_index()
    frequency_list = count.tolist()

    return frequency_list


# readlines the data in csv
def get_row_by_index_cached(csv_file, row_index):
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = []
        for index, row in enumerate(reader):
            row = [int(item) for item in row]
            if index in row_index:
                rows.append(row)
    return rows


def get_ngrams(text, n):
    """obtain the n-grams in text"""
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    tokens = text.split()
    if len(tokens) < n:
        return []  # return [] if tokens < n
    ngrams = [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return ngrams


def count_match(candidate_ngrams, reference_ngrams):
    """Calculate the number of co-occurring n-grams"""
    candidate_count = Counter(candidate_ngrams)
    reference_count = Counter(reference_ngrams)
    match_count = sum(min(candidate_count[ngram], reference_count[ngram]) for ngram in
                      set(candidate_ngrams).intersection(set(reference_ngrams)))
    return match_count


def rouge_n(candidate_summary, reference_summaries, n):
    """Calculate ROUGE-N scores"""
    if not isinstance(candidate_summary, str):
        raise ValueError("Candidate summary must be a string.")
    if not all(isinstance(summary, str) for summary in reference_summaries):
        raise ValueError("All reference summaries must be strings.")

    candidate_ngrams = get_ngrams(candidate_summary, n)
    max_match_count = 0
    for summary in reference_summaries:
        ref_ngrams = get_ngrams(summary, n)
        match_count = count_match(candidate_ngrams, ref_ngrams)
        max_match_count = max(max_match_count, match_count)

    total_ngrams = len(candidate_ngrams)  # Use the number of n-grams in the candidate abstract as the denominator
    rouge_score = max_match_count / total_ngrams if total_ngrams > 0 else 0
    return rouge_score


def functionTokenize(text):
    """
    Tokenize the input text into a list of words.
    :param text: Input text as a string.
    :return: List of words.
    """
    return text.split()


def rouge_1(candidate, references):
    """
    Calculate ROUGE-1 score.
    :param candidate: Candidate summary as a string.
    :param references: List of reference summaries as strings.
    :return: ROUGE-1 score.
    """
    candidate_tokens = functionTokenize(candidate)
    candidate_counter = Counter(candidate_tokens)
    max_recall = 0
    for reference in references:
        reference_tokens = functionTokenize(reference)
        reference_counter = Counter(reference_tokens)
        common_words = candidate_counter & reference_counter
        recall = sum(common_words.values()) / max(1, sum(reference_counter.values()))
        max_recall = max(max_recall, recall)
    return max_recall


def rouge_2(candidate, references):
    """
    Calculate ROUGE-2 score.
    :param candidate: Candidate summary as a string.
    :param references: List of reference summaries as strings.
    :return: ROUGE-2 score.
    """
    candidate_tokens = functionTokenize(candidate)
    candidate_bigrams = Counter(zip(candidate_tokens, candidate_tokens[1:]))
    max_recall = 0
    for reference in references:
        reference_tokens = functionTokenize(reference)
        reference_bigrams = Counter(zip(reference_tokens, reference_tokens[1:]))
        common_bigrams = candidate_bigrams & reference_bigrams
        recall = sum(common_bigrams.values()) / max(1, sum(reference_bigrams.values()))
        max_recall = max(max_recall, recall)
    return max_recall


def nested_list_to_string(lst):
    """
    Convert a nested list to a string.
    :param lst: The nested list to convert.
    :return: A string representation of the nested list.
    """
    if isinstance(lst, list):
        return ' '.join(nested_list_to_string(item) for item in lst)
    else:
        return str(lst)


def process_combinations_without_modeltest_csv_opt(combination_result_temp, bert_tokenizer, model, model_word_test,
                                                   tokenizer_word_test, weight_divided_bias_0, Bert_model,
                                                   bert_batch_size_test, device):
    possible_sentence = []
    model.eval()

    # combination_result_temp = [[101] + sub_list + [102] for sub_list in combination_result_temp]
    combination_result_temp = [sub_list for sub_list in combination_result_temp]
    num_batches = (len(combination_result_temp) + bert_batch_size_test - 1) // bert_batch_size_test
    batches = [combination_result_temp[i * bert_batch_size_test:(i + 1) * bert_batch_size_test] for i in range(num_batches)]

    # Construct the required variables
    token_type_ids = torch.zeros((bert_batch_size_test, len(combination_result_temp[0])), dtype=torch.int32).to(device)
    position_ids = None
    extended_attention_mask = torch.zeros((bert_batch_size_test, 1, 1, len(combination_result_temp[0])), dtype=torch.float32).to(device)
    head_mask = [None] * Bert_model.config.num_hidden_layers

    for batch in batches:
        input_ids = torch.tensor(batch, dtype=torch.int32).to(device)
        # When the shape is incorrect, re-construct it.
        if input_ids.shape != token_type_ids.shape:
            batch_temp, seq_length = input_ids.shape
            token_type_ids = torch.zeros(input_ids.shape, dtype=torch.int32).to(device)
            extended_attention_mask = torch.zeros((batch_temp, 1, 1, seq_length), dtype=torch.float32).to(device)

        layer_output_1 = model.forward_partial_1(input_ids=input_ids, position_ids=position_ids,
                                                 token_type_ids=token_type_ids,
                                                 extended_attention_mask=extended_attention_mask,
                                                 head_mask=head_mask)
        # print("f() layer_output_1:", layer_output_1)
        is_close_in_rows, is_close_in_cols, coefficients, reconstructed_vector = can_be_expressed_two(
            weight_divided_bias_0, layer_output_1, device)
        if is_close_in_rows.numel() > 0:
            is_close_in_rows_list = is_close_in_rows.cpu().tolist()
            valid_sentences = [batch[i] for i in is_close_in_rows_list]
            possible_sentence.extend(valid_sentences)

    possible_sentence = [tuple(sentence) for sentence in possible_sentence]
    possible_sentence = list(set(possible_sentence))
    possible_sentence = [bert_tokenizer.decode(ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                         for ids in possible_sentence]

    possible_sentence = word_test_by_model(model_word_test, tokenizer_word_test, possible_sentence, num_batches)

    return possible_sentence


# code for GPT2
def GPT_process_combinations_without_modeltest_csv_opt(combination_result_temp, GPT2_tokenizer, model, model_word_test,
                                                       tokenizer_word_test, weight_divided_bias_0, bert_batch_size_test,
                                                       device):
    possible_sentence = []
    model.eval()

    input_ids = torch.stack(combination_result_temp)
    input_ids = input_ids.view(1, -1)
    # print("f()input_ids:", input_ids)
    batch_size, max_seq_length = input_ids.size()
    attention_mask = torch.tensor([[[[1.0] * max_seq_length]]]).to(device)
    position_ids = torch.arange(max_seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    layer_output, layer_output_embeddings, layer_output_1 = model(input_ids=input_ids,
                                                                  position_ids=position_ids,
                                                                  attention_mask=attention_mask)
    # print("f()layer_output_1:", layer_output_1)
    is_close_in_rows, is_close_in_cols, coefficients, reconstructed_vector = can_be_expressed(weight_divided_bias_0,
                                                                                              layer_output_1, device)
    if is_close_in_rows.numel() > 0:
        text_combination_result = [GPT2_tokenizer.decode(ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                                   for ids in input_ids]
        possible_sentence = nested_list_to_string(text_combination_result)

    return possible_sentence







