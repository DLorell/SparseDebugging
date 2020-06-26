import os
import sys



model_dirs = [d for d in os.listdir("models") if os.path.isdir("models/"+d)]
for model_dir in model_dirs:
    saves = [f for f in os.listdir("models/"+model_dir) if f.endswith(".pt")]
    print(saves)
    if "200.pt" in saves:
        for save in saves:
            os.remove("models/"+model_dir+"/"+save)








