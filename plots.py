from matplotlib import pyplot as plt
import pandas as pd
import glob
import os

PLOT_OR_SAVE = 1 # 0 for plot, 1 for save

def plot_single_loss(metrics, model_name):
    plt.plot(metrics["loss"], label="train")
    plt.plot(metrics["val_loss"], label="validation")
    plt.xticks(range(1, len(metrics["loss"]) + 1, 1))
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if PLOT_OR_SAVE == 0:
        plt.show()
    else:
        plt.savefig("plots/" + model_name + "_loss.png")
        plt.clf()

def plot_single_accuracy(metrics, model_name):
    plt.plot(metrics["masked_acc"], label="train")
    plt.plot(metrics["val_masked_acc"], label="validation")
    plt.xticks(range(1, len(metrics["masked_acc"]) + 1, 1))
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    if PLOT_OR_SAVE == 0:
        plt.show()
    else:
        plt.savefig("plots/" + model_name + "_accuracy.png")
        plt.clf()

def plot_single_training_time(metrics, model_name):
    # Plot the training time per epoch 
    plt.plot(metrics["training_time"], label="train")
    plt.xticks(range(1, len(metrics["training_time"]) + 1, 1))
    plt.title("Training time per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Training time (s)")
    plt.legend()
    if PLOT_OR_SAVE == 0:
        plt.show()
    else:
        plt.savefig("plots/" + model_name + "_training_time.png")
        plt.clf()

def plot_multiple_loss(metrics, models):
    for i in range(len(metrics)):
        plt.plot(metrics[i]["loss"], label=models[i])
    plt.xticks(range(1, len(metrics[0]["loss"]) + 1, 1))
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if PLOT_OR_SAVE == 0:
        plt.show()
    else:
        plt.savefig("plots/multiple_loss.png")
        plt.clf()

def plot_multiple_accuracy(metrics, models):
    for i in range(len(metrics)):
        plt.plot(metrics[i]["masked_acc"], label=models[i])
    plt.xticks(range(1, len(metrics[0]["masked_acc"]) + 1, 1))
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    if PLOT_OR_SAVE == 0:
        plt.show()
    else:
        plt.savefig("plots/multiple_accuracy.png")
        plt.clf()

def plot_multiple_training_time(metrics, models):
    for i in range(len(metrics)):
        plt.plot(metrics[i]["training_time"], label=models[i])
    plt.xticks(range(1, len(metrics[0]["training_time"]) + 1, 1))
    plt.title("Training time per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Training time (s)")
    plt.legend()
    if PLOT_OR_SAVE == 0:
        plt.show()
    else:
        plt.savefig("plots/multiple_training_time.png")
        plt.clf()



def main():

    # #! This is for plotting a single model
    # # Load the metrics
    # MODEL_NAME = "en_es_20_6_256_512_8_0-3"
    # metrics = pd.read_csv("metrics/" + MODEL_NAME + ".csv")
    # metrics.index.name = "epoch"
    # metrics.index += 1
    # # print(metrics.head())

    # # Plot the loss
    # plot_single_loss(metrics, MODEL_NAME)
    # # Plot the accuracy
    # plot_single_accuracy(metrics, MODEL_NAME)
    # # Plot the training time
    # plot_single_training_time(metrics, MODEL_NAME)


    #! This is for plotting multiple models
    # Load all the models from the metrics/ folder
    files = glob.glob("metrics/*.csv")
    # Remove the .csv extension
    models = [os.path.splitext(os.path.basename(model))[0] for model in files]
    # Get the prefix of the model name en_es
    models = [model.split("_")[0] + "_" + model.split("_")[1] for model in models]

    metrics = []
    for metric in files:
        df = pd.read_csv(metric)
        df.index.name = "epoch"
        df.index += 1
        metrics.append(df)

    # print(len(metrics))
    # print(files)
    # print(models)

    # plot the loss for all the models
    plot_multiple_loss(metrics, models)
    # plot the accuracy for all the models
    plot_multiple_accuracy(metrics, models)
    # plot the training time for all the models
    plot_multiple_training_time(metrics, models)




if __name__ == '__main__':
    main()