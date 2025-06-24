Zorg ervoor dat je alle packages hebt. Ik heb zelf dit gedaan: \
```conda install python=3.12 numpy pytorch tensorflow matplotlib  scikit-learn scikit-image opencv pillow lpips optuna pandas```

Pull de laatste versie van master


In ```predrnn/kth_script/predrnn_kth_train.sh```:

Make sure device is set to your gpu: ```—device cuda \```

Set the train_data_paths and valid_data_paths to absolute location like this: \
```—train_data_paths /Users/maxneerken/Documents/aml/predrnn-pytorch-AML/dataset/kth \``` \
```—valid_data_paths /Users/maxneerken/Documents/aml/predrnn-pytorch-AML/dataset/kth \```

Navigeer naar cd predrnn/kth_script/ met powershell/git bash/wsl en run: ```sh predrnn_kth_train.sh```
