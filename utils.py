import os


def read_data(dir):
    """Read data

    Args:
        dir (str): file location

    Returns:
        (list, list): text and labels
    """
    print("Reading data..")
    text = []
    labels = []

    # read positive points
    with open(os.path.join(dir, 'positives.csv'), 'r', encoding='utf-8') as fp:
        for line in fp:
            text.append(line.strip())
            labels.append(1)

    # read negative points
    with open(os.path.join(dir, 'negatives.csv'), 'r', encoding='utf-8') as fp:
        for line in fp:
            text.append(line.strip())
            labels.append(0)
    
    return text, labels


def clean_text(line):
    pass