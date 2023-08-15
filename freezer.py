def freeze_decoder_weights(model, eachbatch):
    # Train both the encoder and Rmodel1 for the first 40k batches
    if eachbatch <= 40000:
        for param in model.Tmodel.parameters():
            param.requires_grad = True
        for param in model.Rmodel1.parameters():
            param.requires_grad = True
        for param in model.Rmodel2.parameters():
            param.requires_grad = False
        for param in model.Rmodel3.parameters():
            param.requires_grad = False

    # Train Rmodel2 for the next 40k batches
    elif 40000 < eachbatch <= 80000:
        for param in model.Tmodel.parameters():
            param.requires_grad = False
        for param in model.Rmodel1.parameters():
            param.requires_grad = False
        for param in model.Rmodel2.parameters():
            param.requires_grad = True
        for param in model.Rmodel3.parameters():
            param.requires_grad = False

    # Train Rmodel3 for batches beyond 40k
    else:
        for param in model.Tmodel.parameters():
            param.requires_grad = False
        for param in model.Rmodel1.parameters():
            param.requires_grad = False
        for param in model.Rmodel2.parameters():
            param.requires_grad = False
        for param in model.Rmodel3.parameters():
            param.requires_grad = True

def check_requires_grad(model):
    """Check and print the parameters that are frozen or not."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} will be updated (NOT frozen).")
        else:
            print(f"{name} is frozen.")
