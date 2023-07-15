import torch
import time


def train(epochs, train_dataloader, val_dataloader, model, loss_fn, optimizer, device, model_name, writer, run_id, save_gradients=False):
    for epoch in range(epochs):
        print(f"\n{model_name} --- Epoch: {epoch}\n----------------------")
        train_step(data_loader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, device=device, model_name=model_name, writer=writer, epoch_id=epoch, save_gradients=save_gradients, run_id=run_id)
        val_step(data_loader=val_dataloader, model=model, loss_fn=loss_fn, device=device, model_name=model_name, writer=writer, epoch_id=epoch, run_id=run_id)


def store_gradients(model, writer, model_name, epoch_id, run_id):
    gradients = {}
    for name, param in model.named_parameters():
        if name.startswith('layer') and 'conv' in name:
            name = name.split('.')[0]
            current = torch.flatten(param.grad)
            if name in gradients.keys():
                gradients[name] = torch.cat((gradients[name], current))
            else:
                gradients[name] = current

    for k in gradients.keys():
        writer.add_histogram(f'{model_name}/Run{run_id}/{k}/Gradients', gradients[k], epoch_id)

        gradients[k] = torch.var_mean(gradients[k], 0)
        writer.add_scalar(f'{model_name}/Run{run_id}/{k}/Gradients mean', gradients[k][0], epoch_id)
        writer.add_scalar(f'{model_name}/Run{run_id}/{k}/Gradients variance', gradients[k][1], epoch_id)


def train_step(model, data_loader, loss_fn, optimizer, device, model_name, writer, epoch_id, run_id, save_gradients=False):
    train_loss, train_acc = 0, 0
    model.to(device)
    start_training_time = time.time()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end_training_time = time.time()

    writer.add_scalar(f'{model_name}/Run{run_id}/Epoch time', end_training_time - start_training_time, epoch_id)

    if save_gradients:
        store_gradients(model, writer, model_name, epoch_id, run_id)

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    writer.add_scalar(f'{model_name}/Run{run_id}/Loss/Train', train_loss, epoch_id)
    print(f"\nTrain loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def val_step(data_loader, model, loss_fn, device, model_name, writer, epoch_id, run_id):
    val_loss, val_acc = 0, 0
    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            val_pred = model(X)

            val_loss += loss_fn(val_pred, y)
            val_acc += accuracy_fn(y_true=y, y_pred=val_pred.argmax(dim=1))

        val_loss /= len(data_loader)
        val_acc /= len(data_loader)
        writer.add_scalar(f'{model_name}/Run{run_id}/Loss/Validation', val_loss, epoch_id)
        print(f"Validation loss: {val_loss:.5f} | Validation accuracy: {val_acc:.2f}%\n")


def accuracy_fn(y_true, y_pred):
    accuracy = torch.sum(y_true == y_pred).item()
    accuracy /= len(y_true)
    return accuracy * 100
