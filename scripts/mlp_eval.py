import sys

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def print_accuracy(accuracy):
    color = ""
    if accuracy < 50:
        color = bcolors.FAIL
    elif 50 < accuracy < 75:
        color = bcolors.WARNING
    else:
        color = bcolors.OKGREEN
    print(color + 'Accuracy: {:.2f}%'.format(accuracy) + bcolors.ENDC)

def main():
    pred_path = sys.argv[1]
    label_path = sys.argv[2]

    with open(pred_path, encoding='utf-8') as f:
        preds = [l.strip() for l in f.readlines()]
    with open(label_path, encoding='utf-8') as f:
        labels = [l.strip() for l in f.readlines()]
    assert len(preds) == len(labels), "Length of predictions ({}) and labels ({}) are not the same"\
        .format(len(preds), len(labels))
    
    correct = 0
    for pred, label in zip(preds, labels):
        if pred == label:
            correct += 1
    print_accuracy((100.0 * correct) / len(labels))

if __name__ == "__main__":
    main()