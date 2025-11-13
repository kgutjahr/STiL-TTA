from sklearn.metrics import recall_score

def balanced_accuracy(y_true, y_pred, num_classes):
    total_classes = list(range(0, num_classes))
    return recall_score(y_true, y_pred, labels=total_classes, average='macro', zero_division=0)
    
def balanced_accuracy_by_hand(y_true, y_pred, num_classes):
    total_classes = list(range(0, num_classes))
    total_recall = 0
    for c in total_classes:
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fn = ((y_true == c) & (y_pred != c)).sum().item()
        recall = tp / (tp + fn)
        total_recall += recall
    return total_recall / num_classes