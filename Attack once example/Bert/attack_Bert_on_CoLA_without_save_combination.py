from auxiliary_code import attack_auxiliary_function
from auxiliary_code import model_with_adapter
import torch
from transformers import BertModel, BertConfig, BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import csv
import pandas as pd
import time
import os


# The actual working code begins.
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Setting the parameters
randSeed = 1
model_name = 'E:/Model/bert-base-uncased'
data_name = 'E:/Dataset/cola_public_1.1/cola_public/raw/in_domain_train.tsv'
word_test_model = 'E:/Model/Abiratebert_fine_tuned_cola'
num_labels = 20                  # number of classes
batch_size = [2]        # the batch_size settings
bert_batch_size_test = 512       # the size of the inference batch for testing using a large model
guess_max_tok_len = 64           # Estimate the maximum length of the text in the training set
reduction_factor = 2             # reduction_factor setting
cycle_number = range(0, 1, 1)  # the number of experiments
# the pos of results
current_dir = os.path.dirname(os.path.abspath(__file__))
result_file_save_path = os.path.join(current_dir, 'results/Bert_cola_1.csv')
final_result_file_save_path = os.path.join(current_dir, 'results/Bert_cola_1_final.csv')

# seed init
attack_auxiliary_function.set_random_seed(randSeed)

# init the tokenizer
bert_tokenizer = BertTokenizer.from_pretrained(model_name)
Bert_model = BertModel.from_pretrained(model_name, config=BertConfig.from_pretrained(model_name, output_hidden_states=True))

# init the model settings
model = model_with_adapter.BertManual(num_labels=num_labels, bert_path=model_name,
                                      reduction_factor=reduction_factor,
                                      init_config=BertConfig.from_pretrained(model_name, output_hidden_states=True))
model.set_trainable_adapters()
model.to(device)

# set the loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

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
        train_dataset = attack_auxiliary_function.preprocess_tsv_cola(data_name, tokenizer=bert_tokenizer)
        for num_batch_size, batch_size_temp in enumerate(batch_size):
            # set the model
            model.train()
            model.embeddings.eval()
            model.transformer_layer_0.eval()
            model.transformer_layer_1.eval()
            model.pool_layer.eval()

            # set the dataset
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size_temp, shuffle=True)
            random_batch_target = attack_auxiliary_function.get_random_batch(train_dataloader)

            # start training
            input_encodings = bert_tokenizer(random_batch_target["sentence"], truncation=True, padding=True, return_tensors='pt').to(device)
            True_input_ids = input_encodings['input_ids']
            token_type_ids = input_encodings.get('token_type_ids', None)
            position_ids = input_encodings.get('position_ids', None)
            extended_attention_mask = Bert_model.get_extended_attention_mask(input_encodings['attention_mask'],
                                                                             True_input_ids.shape)
            head_mask = [None] * Bert_model.config.num_hidden_layers  # none without head_mask
            outputs, layer_output_embedding, layer_output_1 = model(input_ids=True_input_ids.to(device),
                                                                    position_ids=position_ids,
                                                                    token_type_ids=token_type_ids.to(device),
                                                                    extended_attention_mask=extended_attention_mask.to(device),
                                                                    head_mask=head_mask)
            loss = loss_fn(outputs, random_batch_target["labels"].to(device))  # compute the loss
            optimizer.zero_grad()
            loss.backward()

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

            # Create a vocabulary list matrix
            resualt_is_close_in_rows = []
            resualt_is_close_in_cols = []
            result_embeddings = []
            vocab_size = len(bert_tokenizer)
            token_indices = torch.arange(vocab_size, device=device)
            vocab_size_batch = 1024
            vocab_batches = [token_indices[i:i + vocab_size_batch] for i in
                             range(0, len(token_indices), vocab_size_batch)]
            Token_index_all = []
            for index_vocab_batches, vocab_size_batch_temp in enumerate(vocab_batches):
                vocab_size_batch_temp = vocab_size_batch_temp.clone().detach()
                input_ids_group = torch.ones((len(vocab_size_batch_temp), guess_max_tok_len), dtype=torch.int).to(device)
                input_ids_group = input_ids_group * vocab_size_batch_temp.unsqueeze(-1)

                with torch.no_grad():  # Disable gradient calculation to save memory
                    result_embedding = model.forward_partial_0(input_ids=input_ids_group)
                    is_close_in_rows, is_close_in_cols, coefficients, reconstructed_vector = attack_auxiliary_function.can_be_expressed_two(
                        weight_divided_bias_embedding, result_embedding, device)
                    is_close_in_rows = [t.item() for t in is_close_in_rows]
                    is_close_in_rows = list(set(is_close_in_rows))
                    for index_is_close_in_rows in range(len(is_close_in_rows)):
                        Token_index_all.append(vocab_size_batch_temp[is_close_in_rows[index_is_close_in_rows]])
            Token_index_all = [t.item() for t in Token_index_all]

            # recontruct tokens with Token_index_all
            decode_token = []
            for temp_token in Token_index_all:
                temp_token = bert_tokenizer.decode([temp_token], clean_up_tokenization_spaces=True)
                decode_token.append(temp_token)

            # Inferential unordered tokens
            possible_sentence = []
            true_sentence = []
            for index, possible_token in enumerate(True_input_ids):
                set_possible_token = [x for x in possible_token if x != 0]
                set_possible_token = set(set_possible_token)
                set_possible_token = {t.item() for t in set_possible_token}
                is_subset = set_possible_token.issubset(Token_index_all)
                if is_subset:
                    list_possible_token = [[x for x in possible_token if x != 0]]
                    possible_sentence_temp = attack_auxiliary_function.process_combinations_without_modeltest_csv_opt(
                        list_possible_token, bert_tokenizer, model, model_word_test, tokenizer_word_test,
                        weight_divided_bias_0, Bert_model, bert_batch_size_test, device)
                    if possible_sentence_temp is not []:
                        possible_sentence.extend(possible_sentence_temp)
                        true_sentence.append(random_batch_target["sentence"][index])

            writer.writerow([decode_token, possible_sentence])





























