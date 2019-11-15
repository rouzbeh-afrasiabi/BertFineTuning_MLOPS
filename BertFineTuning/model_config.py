        
model_config={
            'model_name':'BertFineTuning',
            'num_classes':2,
            'dropout_prob':0.1,
            'in_features':768,
            'learning_rate_PT': 1e-5,
            'learning_rate_CLS': 1e-3,
            'weight_decay':5e-4,
            'epochs':4,
            'print_every':100,
             'validate_at_epoch':0
                   }

