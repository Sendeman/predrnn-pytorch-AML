# Netwerk trainen

Zorg ervoor dat je alle packages hebt. Ik heb zelf dit gedaan: \
```conda install python=3.12 numpy pandas matplotlib seaborne pytorch tensorflow scikit-learn scikit-image opencv pillow lpips optuna ```

Zorg ervoor dat je de laatste versie van de autoencoder notebook hebt gedraaid om de data juist te hebben

Pull de laatste versie van master


# Baseline
In ```predrnn/kth_script/predrnn_v2_kth_train.sh```:

Make sure device is set to your gpu: ```—device cuda \```

Set the train_data_paths and valid_data_paths to absolute location like this: \
```—train_data_paths /Users/maxneerken/Documents/aml/predrnn-pytorch-AML/dataset/kth \``` \
```—valid_data_paths /Users/maxneerken/Documents/aml/predrnn-pytorch-AML/dataset/kth \```

En eventueel de max_iterations parameter

Navigeer naar ```cd predrnn/kth_script/``` met powershell/git bash/wsl en run: ```sh predrnn_kth_train.sh```

# Latent
Doe hetzelfde als bij de baseline maar dan met ```predrnn/latent.sh```

Pas de data locaties aan naar de juiste dataset, en **kies waar het model moet worden opgeslagen** moet worden te veranderen op basis van welke architecture je traint. \ 
Pas ook num_hidden aan naar de latent size (i.e 64, 64, 64, 64)
