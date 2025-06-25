# Netwerk trainen

Zorg ervoor dat je alle packages hebt. Ik heb zelf dit gedaan: \
```conda install python=3.12 numpy pytorch tensorflow matplotlib  scikit-learn scikit-image opencv pillow lpips optuna pandas```

Zorg ervoor dat je de laatste versie van de autoencoder notebook hebt gedraaid om de data juist te hebben

Pull de laatste versie van master


# Baseline
In ```predrnn/kth_script/predrnn_kth_train.sh```:

Make sure device is set to your gpu: ```—device cuda \```

Set the train_data_paths and valid_data_paths to absolute location like this: \
```—train_data_paths /Users/maxneerken/Documents/aml/predrnn-pytorch-AML/dataset/kth \``` \
```—valid_data_paths /Users/maxneerken/Documents/aml/predrnn-pytorch-AML/dataset/kth \```

En eventueel de max_iterations parameter

Navigeer naar cd predrnn/kth_script/ met powershell/git bash/wsl en run: ```sh predrnn_kth_train.sh```

# Latent
Doe hetzelfde als bij de baseline maar dan met ```predrnn/latent.sh```

Vergeet niet nu ook de plek waar het model opgeslagen moet worden te veranderen op basis van welke architecture je traint.

