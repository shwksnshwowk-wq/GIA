from .auxiliary_code import attack_auxiliary_function
from .auxiliary_code import model_with_adapter
import torch
from transformers import AutoTokenizer, Qwen2Model, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import csv
import pandas as pd
import time
import os


# The actual working code begins.
from Qwen2.auxiliary_code import attack_auxiliary_function

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

# Setting the parameters
randSeed = 1
model_name = "E:/Model/Qwen2-7B"
data_name = 'E:/Dataset/cola_public_1.1/cola_public/raw/in_domain_train.tsv'
num_labels = 20                  # number of classes
batch_size = [1, 2, 4, 8, 16, 32, 64, 128, 256]        # the batch_size settings
reduction_factor = 2             # reduction_factor setting
cycle_number = range(78, 100, 1)    # the number of experiments

# the pos of results
current_dir = os.path.dirname(os.path.abspath(__file__))
result_file_save_path = os.path.join(current_dir, 'results/Qwen_cola_1.csv')
final_result_file_save_path = os.path.join(current_dir, 'results/Qwen_cola_1_final.csv')
temp_file = os.path.join(current_dir, 'results/temp_token.csv')  # This folder should be fixed.
result_file_header = ['randSeed', 'batch size', 'the number of reconstruction text', 'minimum computer', 'R-1', 'R-2', 'token numbers', 'Shortest sentence length', 'Resume text sentences', 'Drawn text', 'token', 'token length', 'ture token', 'ture token length']
final_result_header = ['batch size', 'the number of reconstruction text', 'ave minimum computer', 'ave R-1', 'ave R-2', 'ave token numbers', 'Shortest sentence length']

# seed init
attack_auxiliary_function.set_random_seed(randSeed)

# init
qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
model = model_with_adapter.Qwen2Manual(num_labels=num_labels, qwen_path=model_name, reduction_factor=reduction_factor)
model.set_trainable_adapters()
model.to(device)

# debug
"""
temp = [93594, 94479, 97000, 98225, 99072, 99074]
vocab_batch = torch.tensor(temp).to(device)
embedding_out = model.forward_partial_0(input_ids=vocab_batch)
print(embedding_out)
print(qwen_tokenizer.decode(temp, clean_up_tokenization_spaces=True, skip_special_tokens=True))
"""

# filter tokens
model.eval()
epsilon = 5e-4
vocab_size = len(qwen_tokenizer)
token_indices = torch.arange(vocab_size, device=device)
vocab_size_batch = 1024
vocab_batches = [token_indices[i:i + vocab_size_batch] for i in range(0, len(token_indices), vocab_size_batch)]
Token_index_effective = []
if not os.path.exists(temp_file):
    with open(temp_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for index_vocab_batches, vocab_batch in enumerate(vocab_batches):
            vocab_size_batch_temp = vocab_batch.clone().detach()
            vocab_size_batch_temp = vocab_size_batch_temp.unsqueeze(1)
            attention_mask = torch.ones_like(vocab_size_batch_temp)
            with torch.no_grad():  # Disable gradient calculation to save memory
                layer_output, layer_output_embeddings, layer_output_1 = model(input_ids=vocab_size_batch_temp,
                                                                              attention_mask=attention_mask)
                abs_embedding_output = torch.abs(layer_output_embeddings)
                elements_close = torch.all(abs_embedding_output < epsilon, dim=-1)
                indices_not_close = torch.where(~elements_close)[0]
                Token_index_effective.extend(vocab_batch[indices_not_close].tolist())
        Token_index_effective = [[x] for x in Token_index_effective]
        writer.writerows(Token_index_effective)


data = pd.read_csv(temp_file)
Token_index_effective = data.values.tolist()                                    # the format of data is [[0],[1],[2],[3]]

# set train parameters
loss_fn = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

# the vars with results
rec_num_all = [[] for _ in range(len(batch_size))]
com_over_all = [[] for _ in range(len(batch_size))]
ave_rouge_1_all = [[] for _ in range(len(batch_size))]
ave_rouge_2_all = [[] for _ in range(len(batch_size))]
token_num_all = [[] for _ in range(len(batch_size))]             # num of tokens
min_len_all = [[] for _ in range(len(batch_size))]               # the min length of text

with open(result_file_save_path, 'a', newline='', encoding='utf-8') as result_file:
    writer = csv.writer(result_file)
    writer.writerow(result_file_header)

    for randSeed in cycle_number:
        print("########################", randSeed, "########################")
        attack_auxiliary_function.set_random_seed(randSeed)

        # load the datasets
        train_dataset = attack_auxiliary_function.preprocess_tsv_cola(data_name, tokenizer=qwen_tokenizer)

        for num_batch_size, batch_size_temp in enumerate(batch_size):
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size_temp, shuffle=True)
            random_batch_target = attack_auxiliary_function.get_random_batch(train_dataloader)

            # set train
            model.train()
            model.qwen.eval()

            # single train
            input_encodings = qwen_tokenizer(random_batch_target["sentence"], truncation=True, padding=True, return_tensors='pt').to(device)
            layer_output, layer_output_embeddings, _ = model(**input_encodings)
            loss = loss_fn(layer_output, random_batch_target["labels"].to(device))
            optimizer.zero_grad()
            loss.backward()

            # extract the gradients
            adapter_embedding_weight_grade = model.adapter_embedding.adapter[0].weight.grad
            adapter_embedding_bias_grade = model.adapter_embedding.adapter[0].bias.grad
            weight_divided_bias_embedding = []
            for weight_grad_row_num in range(len(adapter_embedding_bias_grade)):
                if adapter_embedding_bias_grade[weight_grad_row_num] != 0:
                    weight_divided_bias_embedding.append(adapter_embedding_weight_grade[weight_grad_row_num]/adapter_embedding_bias_grade[weight_grad_row_num])
            weight_divided_bias_embedding = torch.stack(weight_divided_bias_embedding, dim=0)
            weight_divided_bias_embedding = weight_divided_bias_embedding.to(device)

            adapter_0_weight_grade = model.adapter_0.adapter[0].weight.grad
            adapter_0_bias_grade = model.adapter_0.adapter[0].bias.grad
            weight_divided_bias_0 = []
            for weight_grad_row_num in range(len(adapter_0_bias_grade)):
                if adapter_0_bias_grade[weight_grad_row_num] != 0:
                    weight_divided_bias_0.append(adapter_0_weight_grade[weight_grad_row_num]/adapter_0_bias_grade[weight_grad_row_num])
            weight_divided_bias_0 = torch.stack(weight_divided_bias_0, dim=0)


            model.eval()

            # word bag inference
            index_effect = []
            vocab_size = len(Token_index_effective)
            vocab_size_batch = 64
            vocab_batches = [Token_index_effective[i:i + vocab_size_batch] for i in range(0, vocab_size, vocab_size_batch)]
            for index_vocab_batches, vocab_size_batch_temp in enumerate(vocab_batches):
                vocab_size_batch_temp = torch.tensor(vocab_size_batch_temp).to(device)
                embedding_out = model.forward_partial_0(input_ids=vocab_size_batch_temp)
                # check with embedding_adapter
                is_close_in_rows, is_close_in_cols, coefficients, reconstructed_vector = attack_auxiliary_function.\
                    can_be_expressed_two(weight_divided_bias_embedding, embedding_out, device)
                is_close_in_rows = [t.item() for t in is_close_in_rows]
                is_close_in_rows = list(set(is_close_in_rows))
                for index_is_close_in_rows in range(len(is_close_in_rows)):
                    index_effect.append(vocab_size_batch_temp[is_close_in_rows[index_is_close_in_rows]])   # [tensor([429], device='cuda:1'), tensor([1059], device='cuda:1')]
            index_effect = [t.item() for t in index_effect]                                                # [429, 1059]
            index_effect = torch.tensor(index_effect).to(device)                                           # totensor([429, 1059])

            # data inference
            true_token = []
            possible_sentence = []
            min_len = []
            for index_input, temp_input in enumerate(input_encodings['input_ids']):
                temp_input_mask = (temp_input != 151643)                       # Format: tensor([1, 2]), Search for fill-in element 151643
                temp_input = temp_input[temp_input_mask]                       # drop151643
                if not min_len:
                    min_len = len(temp_input)
                elif min_len > len(temp_input):
                    min_len = len(temp_input)
                is_result = torch.isin(temp_input, index_effect)
                is_result = is_result.all()
                true_token.extend(temp_input.tolist())
                if is_result:
                    temp_input = temp_input.unsqueeze(0)
                    temp_attention_mask = torch.ones_like(temp_input)
                    with torch.no_grad():
                        layer_output, _, layer_output_1 = model(input_ids=temp_input, attention_mask=temp_attention_mask)
                    # leverage adapter0's gradients
                    is_close_in_rows, is_close_in_cols, coefficients, reconstructed_vector = attack_auxiliary_function.can_be_expressed_two(weight_divided_bias_0, layer_output_1, device)
                    is_close_in_rows = [t.item() for t in is_close_in_rows]
                    if len(is_close_in_rows) > 0:
                        possible_sentence.append(temp_input.tolist()[0])          # temp_inpur.tolist()：[[1, 2, 3]]

            # Calculate the relevant data for writing to the result file
            rec_num = len(possible_sentence)         # text len
            token_num = len(index_effect)            # filter token len
            com_over = token_num ** min_len          # num of calculation
            rouge_1_score = []                       # r-1
            rouge_2_score = []                       # r-2
            if len(possible_sentence) > 0:
                possible_sentence = [qwen_tokenizer.decode(temp, skip_special_tokens=True, clean_up_tokenization_spaces=True) for temp in possible_sentence]
                for temp in possible_sentence:
                    rouge_1_score.append(attack_auxiliary_function.rouge_1(temp, random_batch_target["sentence"]))
                    rouge_2_score.append(attack_auxiliary_function.rouge_2(temp, random_batch_target["sentence"]))
                ave_rouge_1 = sum(rouge_1_score) / len(rouge_1_score)
                ave_rouge_2 = sum(rouge_2_score) / len(rouge_2_score)
            else:
                ave_rouge_1 = 0
                ave_rouge_2 = 0

            # writer
            rec_num_all[num_batch_size].append(rec_num)
            com_over_all[num_batch_size].append(com_over)
            ave_rouge_1_all[num_batch_size].append(ave_rouge_1)
            ave_rouge_2_all[num_batch_size].append(ave_rouge_2)
            token_num_all[num_batch_size].append(token_num)
            min_len_all[num_batch_size].append(min_len)
            true_token = list(set(true_token))
            writer.writerow([randSeed, batch_size_temp, rec_num, com_over, ave_rouge_1, ave_rouge_2, token_num, min_len, possible_sentence, random_batch_target["sentence"], index_effect.tolist(), len(index_effect.tolist()), true_token, len(true_token)])
            result_file.flush()

with open(final_result_file_save_path, 'w', newline='', encoding='utf-8') as final_file:
    writer = csv.writer(final_file)
    writer.writerow(final_result_header)

    for num_batch_size, batch_size_temp in enumerate(batch_size):
        ave_rec_num_final = sum(rec_num_all[num_batch_size]) / len(rec_num_all[num_batch_size])
        ave_com_over_final = sum(com_over_all[num_batch_size]) / len(com_over_all[num_batch_size])
        ave_ave_rouge_1_final = sum(ave_rouge_1_all[num_batch_size]) / len(ave_rouge_1_all[num_batch_size])
        ave_ave_rouge_2_final = sum(ave_rouge_2_all[num_batch_size]) / len(ave_rouge_2_all[num_batch_size])
        token_num_final = sum(token_num_all[num_batch_size]) / len(token_num_all[num_batch_size])
        min_len_final = sum(min_len_all[num_batch_size]) / len(min_len_all[num_batch_size])
        writer.writerow([batch_size_temp, ave_rec_num_final, ave_com_over_final, ave_ave_rouge_1_final, ave_ave_rouge_2_final, token_num_final, min_len_final])

































