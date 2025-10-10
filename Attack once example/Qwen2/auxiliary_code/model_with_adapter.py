from transformers import BertModel, GPT2Model, Qwen2Model, AutoModelForCausalLM
from torch import nn


# adapter setting
class Adapter(nn.Module):
    def __init__(self, input_feature, reduction_factor):
        super(Adapter, self).__init__()
        self.input_feature = input_feature
        self.output_feature = int(input_feature/reduction_factor)

        self.adapter = nn.Sequential(
            nn.Linear(self.input_feature, self.output_feature),
            nn.ReLU(),
            nn.Linear(self.output_feature, self.input_feature),
            nn.ReLU(),
        )

    def forward(self, input_ids=None):
        return self.adapter(input_ids)


# The class manually propagated by Bert only utilized the embedding layer and the first layer.
class BertManual(nn.Module):  # nn.Module
    def __init__(self, num_labels, bert_path, init_config, reduction_factor=8):
        super(BertManual, self).__init__()  # super(Net, self).__init__()
        self.bert_path = bert_path
        self.init_config = init_config
        Bert = BertModel.from_pretrained(bert_path, config=self.init_config)

        self.embeddings = Bert.embeddings
        self.adapter_embedding = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.transformer_layer_0 = Bert.encoder.layer[0]
        self.transformer_layer_1 = Bert.encoder.layer[1]
        self.adapter_0 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.adapter_1 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.pool_layer = Bert.pooler
        # write the classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(self.init_config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels)
        )

    # The forward propagation function, used for calculating the gradient
    def forward(self, input_ids=None, position_ids=None, token_type_ids=None, extended_attention_mask=None, head_mask=None):
        layer_output_embedding = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        layer_output_adapter_embedding = self.adapter_embedding(layer_output_embedding)
        layer_output_1 = self.transformer_layer_0(layer_output_adapter_embedding, attention_mask=extended_attention_mask, head_mask=head_mask[0])[0]
        layer_output = self.adapter_0(layer_output_1)
        layer_output = self.transformer_layer_1(layer_output, attention_mask=extended_attention_mask, head_mask=head_mask[1])[0]
        layer_output = self.adapter_1(layer_output)
        layer_output = self.pool_layer(layer_output)
        layer_output = self.classifier(layer_output)
        return layer_output, layer_output_embedding, layer_output_1

    # output embedding
    def forward_partial_0(self, input_ids=None, token_type_ids=None):
        layer_output_embedding = self.embeddings(input_ids, position_ids=None, token_type_ids=token_type_ids)
        return layer_output_embedding

    # calculate the first layer's output
    def forward_partial_1(self, input_ids=None, position_ids=None, token_type_ids=None, extended_attention_mask=None, head_mask=None):
        layer_output_embedding = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        layer_output_adapter_embedding = self.adapter_embedding(layer_output_embedding)
        layer_output_1 = self.transformer_layer_0(layer_output_adapter_embedding, attention_mask=extended_attention_mask, head_mask=head_mask[0])[0]
        return layer_output_1

    def set_trainable_adapters(self):
        """
        set the para of models
        """
        # freeze the models
        for param in self.parameters():
            param.requires_grad = False
        # set adapter trained
        for adapter in [self.adapter_embedding, self.adapter_0, self.adapter_1]:
            for param in adapter.parameters():
                param.requires_grad = True
        # set classifer trained
        for param in self.classifier.parameters():
            param.requires_grad = True


# Bert forward
class BertAuto(nn.Module):  # nn.Module
    def __init__(self, num_labels, bert_path, init_config, reduction_factor=8):
        super(BertAuto, self).__init__()  # super(Net, self).__init__()
        self.bert_path = bert_path
        self.init_config = init_config
        # load BERT with from_pretrained
        self.bert = BertModel.from_pretrained(bert_path, config=self.init_config)
        # write new layers
        self.adapter_embedding = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        # 12 layers
        self.transformer_layer_0 = self.bert.encoder.layer[0]
        self.transformer_layer_1 = self.bert.encoder.layer[1]
        self.transformer_layer_2 = self.bert.encoder.layer[2]
        self.transformer_layer_3 = self.bert.encoder.layer[3]
        self.transformer_layer_4 = self.bert.encoder.layer[4]
        self.transformer_layer_5 = self.bert.encoder.layer[5]
        self.transformer_layer_6 = self.bert.encoder.layer[6]
        self.transformer_layer_7 = self.bert.encoder.layer[7]
        self.transformer_layer_8 = self.bert.encoder.layer[8]
        self.transformer_layer_9 = self.bert.encoder.layer[9]
        self.transformer_layer_10 = self.bert.encoder.layer[10]
        self.transformer_layer_11 = self.bert.encoder.layer[11]
        # 12 adapters
        self.adapter_0 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.adapter_1 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.adapter_2 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.adapter_3 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.adapter_4 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.adapter_5 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.adapter_6 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.adapter_7 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.adapter_8 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.adapter_9 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.adapter_10 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.adapter_11 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        # pooler_layer
        self.pooler_layer = self.bert.pooler
        # write the classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(self.init_config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels)
        )
        self.dropout = nn.Dropout(self.init_config.hidden_dropout_prob)

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None,
                extended_attention_mask=None, head_mask=None):
        layer_output_embedding = self.bert.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        layer_output_adapter_embedding = self.adapter_embedding(layer_output_embedding)

        layer_output_1 = self.transformer_layer_0(layer_output_adapter_embedding, attention_mask=extended_attention_mask, head_mask=head_mask[0])[0]
        layer_output = self.adapter_0(layer_output_1)
        layer_output = self.transformer_layer_1(layer_output, attention_mask=extended_attention_mask, head_mask=head_mask[1])[0]
        layer_output = self.adapter_1(layer_output)
        layer_output = self.transformer_layer_2(layer_output, attention_mask=extended_attention_mask, head_mask=head_mask[2])[0]
        layer_output = self.adapter_2(layer_output)
        layer_output = self.transformer_layer_3(layer_output, attention_mask=extended_attention_mask, head_mask=head_mask[3])[0]
        layer_output = self.adapter_3(layer_output)
        layer_output = self.transformer_layer_4(layer_output, attention_mask=extended_attention_mask, head_mask=head_mask[4])[0]
        layer_output = self.adapter_4(layer_output)
        layer_output = self.transformer_layer_5(layer_output, attention_mask=extended_attention_mask, head_mask=head_mask[5])[0]
        layer_output = self.adapter_5(layer_output)
        layer_output = self.transformer_layer_6(layer_output, attention_mask=extended_attention_mask, head_mask=head_mask[6])[0]
        layer_output = self.adapter_6(layer_output)
        layer_output = self.transformer_layer_7(layer_output, attention_mask=extended_attention_mask, head_mask=head_mask[7])[0]
        layer_output = self.adapter_7(layer_output)
        layer_output = self.transformer_layer_8(layer_output, attention_mask=extended_attention_mask, head_mask=head_mask[8])[0]
        layer_output = self.adapter_8(layer_output)
        layer_output = self.transformer_layer_9(layer_output, attention_mask=extended_attention_mask, head_mask=head_mask[9])[0]
        layer_output = self.adapter_9(layer_output)
        layer_output = self.transformer_layer_10(layer_output, attention_mask=extended_attention_mask, head_mask=head_mask[10])[0]
        layer_output = self.adapter_10(layer_output)
        layer_output = self.transformer_layer_11(layer_output, attention_mask=extended_attention_mask, head_mask=head_mask[11])[0]
        layer_output = self.adapter_11(layer_output)
        layer_output = self.pooler_layer(layer_output)
        layer_output = self.classifier(layer_output)
        return layer_output, layer_output_embedding, layer_output_1

    def forward_partial_0(self, input_ids=None, token_type_ids=None):
        layer_output_embedding = self.bert.embeddings(input_ids, position_ids=None, token_type_ids=token_type_ids)
        return layer_output_embedding

    def forward_partial_1(self, input_ids=None, position_ids=None, token_type_ids=None,
                          extended_attention_mask=None, head_mask=None):
        layer_output_embedding = self.bert.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        layer_output_adapter_embedding = self.adapter_embedding(layer_output_embedding)
        layer_output_1 = self.transformer_layer_0(layer_output_adapter_embedding, attention_mask=extended_attention_mask, head_mask=head_mask[0])[0]
        return layer_output_1

    def set_trainable_adapters(self):
        """
        set the para of models
        """
        for param in self.parameters():
            param.requires_grad = False
        # defreeze index layers
        for adapter in [self.adapter_embedding, self.adapter_0, self.adapter_1, self.adapter_2, self.adapter_3,
                        self.adapter_4, self.adapter_5, self.adapter_6, self.adapter_7, self.adapter_8,
                        self.adapter_9, self.adapter_10, self.adapter_11]:
            for param in adapter.parameters():
                param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True


# GPT2
class GPT2Manual(nn.Module):
    def __init__(self, num_labels, gpt_path, init_config, reduction_factor=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpt_path = gpt_path
        self.init_config = init_config
        GPT2 = GPT2Model.from_pretrained(gpt_path, config=self.init_config)
        # create the related structure (embedding) as GPT2
        self.wte = GPT2.wte
        self.wpe = GPT2.wpe
        self.drop = GPT2.drop
        self.layer_0 = GPT2.h[0]
        self.layer_1 = GPT2.h[1]
        # two adapters
        self.adapter_embedding = Adapter(self.init_config.n_embd, reduction_factor=reduction_factor)
        self.adapter_0 = Adapter(self.init_config.n_embd, reduction_factor=reduction_factor)
        # set the classifers
        self.classifier = nn.Sequential(
            nn.Linear(self.init_config.n_embd, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids=None, position_ids=None, attention_mask=None):
        embeddings = self.wte(input_ids) + self.wpe(position_ids)
        layer_output_embeddings = self.drop(embeddings)
        layer_output_adapter_embedding = self.adapter_embedding(layer_output_embeddings)
        # layer_output_1 = self.layer_0(layer_output_adapter_embedding, attention_mask=attention_mask[0])[0]
        layer_output_1 = self.layer_0(layer_output_adapter_embedding)[0]
        layer_output_adapter_0 = self.adapter_0(layer_output_1)
        # layer_output_adapter_0 = self.layer_1(layer_output_adapter_0, attention_mask=attention_mask[0])[0]
        layer_output_adapter_0 = self.layer_1(layer_output_adapter_0)[0]
        layer_output_adapter_0 = layer_output_adapter_0[:, -1, :]
        layer_output = self.classifier(layer_output_adapter_0)
        return layer_output, layer_output_embeddings, layer_output_1

    def forward_partial_0(self, input_ids=None, position_ids=None):
        embeddings = self.wte(input_ids) + self.wpe(position_ids)
        layer_output_embeddings = self.drop(embeddings)
        return layer_output_embeddings

    def forward_partial_1(self, input_ids=None, position_ids=None, attention_mask=None):
        embeddings = self.wte(input_ids) + self.wpe(position_ids)
        layer_output_embeddings = self.drop(embeddings)
        layer_output_adapter_embedding = self.adapter_embedding(layer_output_embeddings)
        # layer_output_1 = self.layer_0(layer_output_adapter_embedding, attention_mask=attention_mask[0])[0]
        layer_output_1 = self.layer_0(layer_output_adapter_embedding)[0]
        return layer_output_1

    def set_trainable_adapters(self):
        """
        set the para of models
        """
        # freeze the models
        for param in self.parameters():
            param.requires_grad = False
        # set adapter trained
        for adapter in [self.adapter_embedding, self.adapter_0]:
            for param in adapter.parameters():
                param.requires_grad = True
        # set classifer trained
        for param in self.classifier.parameters():
            param.requires_grad = True



class Qwen2Manual(nn.Module):
    def __init__(self, num_labels, qwen_path, reduction_factor=8, *args, **kwargs):
        super(Qwen2Manual, self).__init__(*args, **kwargs)
        self.qwen = Qwen2Model.from_pretrained(qwen_path, output_hidden_states=True)
        self.qwen_path = qwen_path
        self.init_config = self.qwen.config

        # create the related structure (embedding) as GPT2
        self.qwen.layers = self.qwen.layers[0:2]

        # two adapters
        self.adapter_embedding = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.adapter_0 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)

        # set the classifers
        self.classifier = nn.Sequential(
            nn.Linear(self.init_config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None):
        [_, seq_length] = attention_mask.shape
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.expand(-1, -1, seq_length, -1)
        attention_mask = attention_mask.bool()

        embeds_output = self.qwen.embed_tokens(input_ids)
        inputs_embeds_after_adapter_embedding = self.adapter_embedding(embeds_output)

        layer_0_output = self.qwen.layers[0](inputs_embeds_after_adapter_embedding, attention_mask=attention_mask)[0]
        hidden_states = self.adapter_0(layer_0_output)

        for layer in self.qwen.layers[1:2]:
            layer_outputs = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]

        hidden_states = self.qwen.norm(hidden_states)
        logics = self.classifier(hidden_states[:, 0, :])
        return logics, embeds_output, layer_0_output

    def forward_partial_0(self, input_ids=None):
        embeds_output = self.qwen.embed_tokens(input_ids)
        return embeds_output

    def forward_partial_1(self, input_ids=None, attention_mask=None):
        [_, seq_length] = attention_mask.shape
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.expand(-1, -1, seq_length, -1)
        attention_mask = attention_mask.bool()
        embeds_output = self.qwen.embed_tokens(input_ids)
        inputs_embeds_after_adapter_embedding = self.adapter_embedding(embeds_output)
        layer_0_output = self.qwen.layers[0](inputs_embeds_after_adapter_embedding, attention_mask=attention_mask)[0]
        return layer_0_output

    def set_trainable_adapters(self):
        """
        set the para of models
        """
        # freeze the models
        for param in self.parameters():
            param.requires_grad = False
        # set adapter trained
        for adapter in [self.adapter_embedding, self.adapter_0]:
            for param in adapter.parameters():
                param.requires_grad = True
        # set classifer trained
        for param in self.classifier.parameters():
            param.requires_grad = True

"""
class Qwen2ManualV2(nn.Module):
    def __init__(self, num_labels, qwen_path, reduction_factor=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qwen = Qwen2Model.from_pretrained(qwen_path, output_hidden_states=True)
        self.qwen_path = qwen_path
        self.init_config = self.qwen.config

        # create the related structure (embedding) as GPT2
        self.qwen.layers = self.qwen.layers[0:2]
        # self.embed_tokens = Qwen.embed_tokens
        # self.layers = Qwen.layers[0:2]
        # self.norm = Qwen.norm

        # two adapters
        self.adapter_embedding = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)
        self.adapter_0 = Adapter(self.init_config.hidden_size, reduction_factor=reduction_factor)

        # set the classifers
        self.classifier = nn.Sequential(
            nn.Linear(self.init_config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None):
        [_, seq_length] = attention_mask.shape
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.expand(-1, -1, seq_length, -1)
        attention_mask = attention_mask.bool()
        embeds_output = self.embed_tokens(input_ids)
        inputs_embeds_after_adapter_embedding = self.adapter_embedding(embeds_output)
        layer_0_output = self.layers[0](inputs_embeds_after_adapter_embedding, attention_mask=attention_mask)[0]
        adapter_0_output = self.adapter_0(layer_0_output)
        layer_1_output = self.layers[1](adapter_0_output, attention_mask=attention_mask)[0]
        norm_output = self.norm(layer_1_output)
        layer_output = self.classifier(norm_output[:, 0, :])

        return layer_output, embeds_output, layer_0_output

    def forward_partial_0(self, input_ids=None):
        embeds_output = self.embed_tokens(input_ids)
        return embeds_output

    def forward_partial_1(self, input_ids=None, attention_mask=None):
        [_, seq_length] = attention_mask.shape
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.expand(-1, -1, seq_length, -1)
        attention_mask = attention_mask.bool()
        embeds_output = self.embed_tokens(input_ids)
        inputs_embeds_after_adapter_embedding = self.adapter_embedding(embeds_output)
        layer_0_output = self.layers[0](inputs_embeds_after_adapter_embedding, attention_mask=attention_mask)[0]
        return layer_0_output

    def set_trainable_adapters(self):
        # freeze the models
        for param in self.parameters():
            param.requires_grad = False
        # set adapter trained 
        for adapter in [self.adapter_embedding, self.adapter_0]:
            for param in adapter.parameters():
                param.requires_grad = True
        # set classifer trained
        for param in self.classifier.parameters():
            param.requires_grad = True
"""





