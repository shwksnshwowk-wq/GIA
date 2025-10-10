from auxiliary_code import attack_auxiliary_function
from auxiliary_code import model_with_adapter
import torch
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import csv
import pandas as pd
import time
import os


noise_multiplier = 0.0001
sensitivity = 1.0
clip_norm = 10.0

def clip_gradient(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = clip_norm / (total_norm + 1e-6)  # default the clip_norm == 1.0
    if clip_coef < 1:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)


def add_noise_to_gradients(model, batch_size, dataset_size):
    noise_scale = noise_multiplier * sensitivity

    for param in model.parameters():
        if param.grad is not None:
            noise = torch.normal(
                mean=0,
                std=noise_scale,
                size=param.grad.data.size(),
                device=param.grad.data.device
            )
            param.grad.data.add_(noise)



# The actual working code begins.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setting the parameters
randSeed = 1
model_name = "/home/csluo/FL-LLM/models/GPT2-large"
data_name = '/home/csluo/FL-LLM/datasets/cola_public/raw/in_domain_train.tsv'
word_test_model = '/home/csluo/FL-LLM/Abiratebert_fine_tuned_cola'
num_labels = 20                  # number of classes
batch_size = [1, 2, 4, 8]        # the batch_size settings
bert_batch_size_test = 512       # the size of the inference batch for testing using a large model
guess_max_tok_len = 64           # Estimate the maximum length of the text in the training set
reduction_factor = 2             # reduction_factor setting
cycle_number = range(0, 10, 1)  # the number of experiments
min_lenth = 6                    # The minimum length of the cause-of-accident speculation
# the pos of results
current_dir = os.path.dirname(os.path.abspath(__file__))
result_file_save_path = os.path.join(current_dir, 'results/GPT_cola_1.csv')
final_result_file_save_path = os.path.join(current_dir, 'results/GPT_cola_1_final.csv')

# seed init
attack_auxiliary_function.set_random_seed(randSeed)

# init the tokenizer
GPT2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
GPT2_tokenizer.pad_token = GPT2_tokenizer.eos_token  # Use the end marker as the fill marker

# init the model settings
model = model_with_adapter.GPT2Manual(num_labels=num_labels, gpt_path=model_name, reduction_factor=reduction_factor,
                                      init_config=GPT2Config.from_pretrained(model_name, output_hidden_states=True))
model.set_trainable_adapters()
model.to(device)

# set the loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
model.train()
model.wte.eval()
model.wpe.eval()
model.drop.eval()
model.layer_0.eval()
model.layer_1.eval()

# load the test model
tokenizer_word_test = AutoTokenizer.from_pretrained(word_test_model)
model_word_test = AutoModelForSequenceClassification.from_pretrained(word_test_model)
model_word_test.to(device)
model_word_test.eval()

# check the save path
if os.path.exists(result_file_save_path):
    mode = 'a'
else:
    mode = 'w'

# the vars with results
rec_num_all = [[] for _ in range(len(batch_size))]
com_over_all = [[] for _ in range(len(batch_size))]
ave_rouge_1_all = [[] for _ in range(len(batch_size))]
ave_rouge_2_all = [[] for _ in range(len(batch_size))]
token_num_all = [[] for _ in range(len(batch_size))]             # num of tokens
min_len_all = [[] for _ in range(len(batch_size))]               # the min length of text

with open(result_file_save_path, 'w', newline='', encoding='utf-8') as result_file:
    writer = csv.writer(result_file)

    for randSeed in cycle_number:
        print("########################", randSeed, "########################")
        attack_auxiliary_function.set_random_seed(randSeed)

        # set the dataset
        train_dataset = attack_auxiliary_function.preprocess_tsv_cola(data_name, tokenizer=GPT2_tokenizer)
        for num_batch_size, batch_size_temp in enumerate(batch_size):
            # set the dataset
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size_temp, shuffle=True)
            random_batch_target = attack_auxiliary_function.get_random_batch(train_dataloader)
            # print('the sentences chosen：\n', random_batch_target["sentence"])

            # start training
            input_encodings = GPT2_tokenizer(random_batch_target["sentence"], truncation=True, padding=True,
                                             return_tensors='pt').to(device)
            True_input_ids = input_encodings["input_ids"]
            # print("True_input_ids:", True_input_ids)
            attention_mask = input_encodings["attention_mask"]
            this_batch_size, max_seq_length = True_input_ids.size()
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, sequence_length)
            attention_mask = attention_mask.to(torch.float32)  # 转换为浮点类型
            position_ids = torch.arange(max_seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(this_batch_size, -1)
            layer_output, layer_output_embeddings, layer_output_1 = model(input_ids=True_input_ids,
                                                                          position_ids=position_ids,
                                                                          attention_mask=attention_mask)
            loss = loss_fn(layer_output, random_batch_target["labels"].to(device))  # compute the loss
            optimizer.zero_grad()
            loss.backward()

            # DP-SGD
            clip_gradient(model)
            data_size = len(train_dataloader.dataset)
            add_noise_to_gradients(model, batch_size, data_size)

            # get gradients
            adapter_embedding_weight_grade = model.adapter_embedding.adapter[0].weight.grad
            adapter_embedding_bias_grade = model.adapter_embedding.adapter[0].bias.grad
            weight_divided_bias_embedding = []
            for weight_grad_row_num in range(len(adapter_embedding_bias_grade)):
                if adapter_embedding_bias_grade[weight_grad_row_num] != 0:
                    weight_divided_bias_embedding.append(adapter_embedding_weight_grade[weight_grad_row_num] /
                                                         adapter_embedding_bias_grade[weight_grad_row_num])
            weight_divided_bias_embedding = torch.stack(weight_divided_bias_embedding, dim=0)

            adapter_0_weight_grade = model.adapter_0.adapter[0].weight.grad
            adapter_0_bias_grade = model.adapter_0.adapter[0].bias.grad
            weight_divided_bias_0 = []
            for weight_grad_row_num in range(len(adapter_0_bias_grade)):
                if adapter_0_bias_grade[weight_grad_row_num] != 0:
                    weight_divided_bias_0.append(adapter_0_weight_grade[weight_grad_row_num] /
                                                 adapter_0_bias_grade[weight_grad_row_num])
            weight_divided_bias_0 = torch.stack(weight_divided_bias_0, dim=0)
            # print(weight_divided_bias_0)

            # Create a vocabulary list matrix
            resualt_is_close_in_rows = []
            resualt_is_close_in_cols = []
            result_embeddings = []
            vocab_size = len(GPT2_tokenizer)
            token_indices = torch.arange(vocab_size, device=device)
            vocab_size_batch = 64
            vocab_batches = [token_indices[i:i + vocab_size_batch] for i in
                             range(0, len(token_indices), vocab_size_batch)]
            Token_index_all = []
            for index_vocab_batches, vocab_size_batch_temp in enumerate(vocab_batches):
                vocab_size_batch_temp = vocab_size_batch_temp.clone().detach()
                input_ids_group = torch.ones((len(vocab_size_batch_temp), guess_max_tok_len), dtype=torch.int).to(
                    device)
                input_ids_group = input_ids_group * vocab_size_batch_temp.unsqueeze(-1)

                bert_batch_size_test_temp, _ = input_ids_group.size()
                position_ids_group = torch.arange(guess_max_tok_len, dtype=torch.long, device=device).unsqueeze(
                    0).expand(bert_batch_size_test_temp, -1)

                with torch.no_grad():  # Disable gradient calculation to save memory
                    result_embedding = model.forward_partial_0(input_ids=input_ids_group, position_ids=position_ids_group)
                    is_close_in_rows, is_close_in_cols, coefficients, reconstructed_vector = attack_auxiliary_function.can_be_expressed_two(
                        weight_divided_bias_embedding, result_embedding, device)
                    is_close_in_rows = [t.item() for t in is_close_in_rows]
                    is_close_in_rows = list(set(is_close_in_rows))
                    for index_is_close_in_rows in range(len(is_close_in_rows)):
                        Token_index_all.append(vocab_size_batch_temp[is_close_in_rows[index_is_close_in_rows]])
            Token_index_all = [t.item() for t in Token_index_all]

            # print("Token_inex_all:", Token_index_all)
            # print("Token_inex_all:", len(Token_index_all))

            # Inferential unordered tokens
            possible_sentence = []
            true_sentence = []
            for index, possible_token in enumerate(True_input_ids):
                possible_sentence_temp = []
                set_possible_token = [x for x in possible_token if x != 0]
                accumulated_list = [set_possible_token[:i + 1] for i in range(len(set_possible_token))]
                for temp_accumulated_list in accumulated_list:
                    temp_accumulated_set = set(temp_accumulated_list)
                    temp_accumulated_set = {t.item() for t in temp_accumulated_set}
                    is_subset = temp_accumulated_set.issubset(Token_index_all)
                    if is_subset:
                        possible_sentence_here = attack_auxiliary_function.GPT_process_combinations_without_modeltest_csv_opt(
                            temp_accumulated_list, GPT2_tokenizer, model, model_word_test,
                            tokenizer_word_test, weight_divided_bias_0, bert_batch_size_test,
                            device)
                        if possible_sentence_here:
                            possible_sentence_temp = possible_sentence_here
                        else:
                            if len(possible_sentence_temp) > min_lenth:
                                break
                if possible_sentence_temp != []:
                    possible_sentence.append(possible_sentence_temp)
                    true_sentence.append(random_batch_target["sentence"][index])

            # Count the number of reconstructions
            rec_num = len(possible_sentence)

            # Calculate the number of times that inference is needed
            True_input_ids_without_zero = []
            for temp in True_input_ids:
                temp = [x for x in temp if x != 0]
                True_input_ids_without_zero.append(temp)
            token_num = len(Token_index_all) - 2
            min_len = min(len(s) for s in True_input_ids_without_zero) - 2
            com_over = token_num ** min_len

            # Calculate the values of R-1 and R-2, only for the texts that have been successfully reconstructed.
            rouge_1_score = []
            rouge_2_score = []
            temp = true_sentence
            true_sentence = []
            for temp_sentence in temp:
                true_sentence.append(temp_sentence)
            if true_sentence != []:
                true_sentence = GPT2_tokenizer(true_sentence, truncation=True, padding=True,
                                               return_tensors='pt').to(device)
                true_sentence = [GPT2_tokenizer.decode(ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                                 for ids in true_sentence['input_ids']]
            if len(possible_sentence) != 0:
                for index in range(len(possible_sentence)):
                    # print(possible_sentence)
                    rouge_1_score.append(attack_auxiliary_function.rouge_1(possible_sentence[index], true_sentence))
                    rouge_2_score.append(attack_auxiliary_function.rouge_2(possible_sentence[index], true_sentence))
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
            writer.writerow([randSeed, batch_size_temp, rec_num, com_over, ave_rouge_1, ave_rouge_2, token_num, min_len,
                             possible_sentence, random_batch_target["sentence"]])

with open(final_result_file_save_path, 'w', newline='') as final_file:
    writer = csv.writer(final_file)

    for num_batch_size, batch_size_temp in enumerate(batch_size):
        ave_rec_num_final = sum(rec_num_all[num_batch_size]) / len(rec_num_all[num_batch_size])
        ave_com_over_final = sum(com_over_all[num_batch_size]) / len(com_over_all[num_batch_size])
        ave_ave_rouge_1_final = sum(ave_rouge_1_all[num_batch_size]) / len(ave_rouge_1_all[num_batch_size])
        ave_ave_rouge_2_final = sum(ave_rouge_2_all[num_batch_size]) / len(ave_rouge_2_all[num_batch_size])
        token_num_final = sum(token_num_all[num_batch_size]) / len(token_num_all[num_batch_size])
        min_len_final = sum(min_len_all[num_batch_size]) / len(min_len_all[num_batch_size])
        writer.writerow([batch_size_temp, ave_rec_num_final, ave_com_over_final, ave_ave_rouge_1_final,
                         ave_ave_rouge_2_final, token_num_final, min_len_final])































