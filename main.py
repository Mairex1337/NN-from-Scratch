import torch
import pandas as pd
import matplotlib.pyplot as plt

# ____________________________________________________________________________________________________________________
# Input parsing

data_test = pd.read_csv('/Users/max/PycharmProjects/NN From Scratch/venv/mnist_test.csv')
data_train = pd.read_csv('/Users/max/PycharmProjects/NN From Scratch/venv/mnist_train.csv')
# Turn data into Tensors
# Separate and squeeze labels
test_tensor = torch.tensor(data_test.values[:, 1:])
test_label = torch.tensor(data_test.values.T[0:1])
test_label = test_label.squeeze(0)
train_tensor = torch.tensor(data_train.values[:, 1:])
train_label = torch.tensor(data_train.values.T[0:1])
train_label = train_label.squeeze(0)

# Normalize pixel values to be between 0 and 1
test_tensor = test_tensor / 255
train_tensor = train_tensor / 255

# ____________________________________________________________________________________________________________________
# Initializing the network


def initialize_parameters():
    # weights of input
    w1 = (2 * torch.rand(784, 100) - 1) * 0.1
    # biases of the hidden layer
    b1 = torch.zeros(100)
    # weights of hidden layer
    w2 = (2 * torch.rand(100, 10) - 1) * 0.1
    # biases of output
    b2 = torch.zeros(10)

    return w1, b1, w2, b2


# ____________________________________________________________________________________________________________________
# forward pass
def forward_pass(input_data, w1, b1, w2, b2, targets):
    with torch.no_grad():  # tell pytorch not to keep track of gradients
        z1 = (input_data @ w1) + b1
        za1 = torch.tanh(z1)
        z2 = (za1 @ w2) + b2
        # softmax activation
        z2_exp = torch.exp(z2)
        z2_sum = torch.sum(z2_exp, dim=1, keepdim=True)
        za2 = z2_exp / z2_sum

        # Calculate prediction accuracy
        _, predicted_classes = torch.max(za2, dim=1)
        correct_predictions = 0
        for i, c in zip(predicted_classes, targets):
            if i == c:
                correct_predictions += 1
        accuracy = (correct_predictions / za2.shape[0]) * 100
        if input_data.shape == torch.Size([60000, 784]):
            print(f"Average training accuracy: {accuracy:.2f}%")
        else:
            print(f"Average test accuracy: {accuracy:.2f}%")

        return za2, za1, accuracy


# ____________________________________________________________________________________________________________________
# Compute loss


def negative_log_likelihood(predictions, targets):
    with torch.no_grad():  # tell pytorch not to keep track of gradients
        # gather the predictions the model made for the target output
        correct_probs = predictions[range(predictions.shape[0]), targets]
        # compute the negative log likelihood of the probabilities
        negative_ll = -torch.log(correct_probs + 1e-10)
        # compute the average loss
        loss = negative_ll.mean()
        return loss.item()

# ____________________________________________________________________________________________________________________
# backward pass


def backward_pass(inputs, targets, za1, w2, predictions):
    # turn predictions into onehot vectors
    onehot = torch.zeros_like(predictions)
    onehot[range(onehot.shape[0]), targets] = 1

    dz2 = predictions - onehot
    dw2 = za1.T @ dz2
    db2 = dz2.sum(dim=0)

    dza1 = dz2 @ w2.T
    dz1 = dza1 * (1 - za1 ** 2)  # derivative of tanh

    dw1 = inputs.T @ dz1
    db1 = dz1.sum(dim=0)
    return dw1, db1, dw2, db2

# ____________________________________________________________________________________________________________________
# parameter updates


def parameter_updates(dw1, dw2, db1, db2):
    learning_rate = 0.001
    dlw1 = learning_rate * dw1
    dlw2 = learning_rate * dw2
    dlb1 = learning_rate * db1
    dlb2 = learning_rate * db2
    return dlw1, dlw2, dlb1, dlb2


# ____________________________________________________________________________________________________________________
# save params (not actually used yet)
def save_model_parameters(w1, b1, w2, b2, filename="model_parameters.pth"):
    torch.save({
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2
    }, filename)


# ____________________________________________________________________________________________________________________
# initializing params

w1, b1, w2, b2 = initialize_parameters()

# ____________________________________________________________________________________________________________________
# Plotting

plt.style.use('dark_background')

plt.ion()
fig, ax = plt.subplots()
ax.set_facecolor('#2C2F33')  # Setting the plot's background color

# Setting up the axis and the grid
ax.set_xlabel('Iterations', color='white')
ax.set_ylabel('Accuracy (%)', color='white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 100)
ax.xaxis.set_ticks(range(0, 11, 1))  # X-axis in steps of 10
ax.yaxis.set_ticks(range(0, 101, 10))  # Y-axis in steps of 10
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

# Making sure the labels and ticks are white for visibility
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

train_line, = ax.plot([], [], color='#E91E63', label='Train Accuracy')
test_line, = ax.plot([], [], color='#03A9F4', label='Test Accuracy')
ax.legend()

train_accuracies = []
test_accuracies = []
plt.pause(5)

# training loop
for iterations in range(10):

    # forward pass
    predictions, za1, train_accuracy = forward_pass(train_tensor, w1, b1, w2, b2, train_label)
    predictions_test, pza1, test_accuracy = forward_pass(test_tensor, w1, b1, w2, b2, test_label)

    # append accuracies to plot
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # compute loss
    loss_train = negative_log_likelihood(predictions, train_label)
    loss_test = negative_log_likelihood(predictions_test, test_label)

    # backward pass
    dw1, db1, dw2, db2 = backward_pass(train_tensor, train_label, za1, w2, predictions)

    # update params
    dlw1, dlw2, dlb1, dlb2 = parameter_updates(dw1, dw2, db1, db2)
    w1, w2, b1, b2 = w1 - dlw1, w2 - dlw2, b1 - dlb1, b2 - dlb2

    # print statements
    print("Iteration:", iterations + 1, "Current TRAINING negative log likelihood:", round(loss_train, 4))
    print("Current TEST negative log likelihood:", round(loss_test, 4))

    # Plotting
    train_line.set_xdata(range(iterations + 1))
    train_line.set_ydata(train_accuracies)
    test_line.set_xdata(range(iterations + 1))
    test_line.set_ydata(test_accuracies)

    if iterations > 0:
        train_text.remove()
        test_text.remove()

    train_text = ax.text(10, 86, f'Train Acc: {train_accuracy:.2f}%', color='#E91E63', fontsize=13, fontweight='bold', ha='right')
    test_text = ax.text(10, 82, f'Test Acc: {test_accuracy:.2f}%', color='#03A9F4', fontsize=13, fontweight='bold', ha='right')

    fig.canvas.flush_events()


plt.ioff()
plt.show()
