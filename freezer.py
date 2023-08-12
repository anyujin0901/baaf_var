def freeze_decoder_weights(model, eachbatch):
    """Freeze the weights of the decoders based on the batch number."""
    # Reset the requires_grad attribute for all decoders
    for param in model.Rmodel1.parameters():
        param.requires_grad = True
    for param in model.Rmodel2.parameters():
        param.requires_grad = True
    for param in model.Rmodel3.parameters():
        param.requires_grad = True

    # Decide which decoder to freeze based on eachbatch
    if (eachbatch-1) % 100 == 0:
        x = ((eachbatch - 1) / 100) % 3
        if x == 0:
            for param in model.Rmodel2.parameters():
                param.requires_grad = False
            for param in model.Rmodel3.parameters():
                param.requires_grad = False
        elif x == 1:
            for param in model.Rmodel1.parameters():
                param.requires_grad = False
            for param in model.Rmodel3.parameters():
                param.requires_grad = False
        elif x == 2:
            for param in model.Rmodel1.parameters():
                param.requires_grad = False
            for param in model.Rmodel2.parameters():
                param.requires_grad = False

def check_requires_grad(model):
    """Check and print the parameters that are frozen or not."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} will be updated (NOT frozen).")
        else:
            print(f"{name} is frozen.")
