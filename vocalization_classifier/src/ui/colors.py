"""
This simply gets different colors for accuracy/loss terminal output
"""
# returns a color based on the accuracy number (target >= 92%)
def get_acc_color(acc):
    if acc >= 0.92:
        return "green"
    elif acc >= 0.85:
        return "yellow"
    else:
        return "red"

# returns a color based on the loss number (target <= 0.5)    
def get_loss_color(loss):
    if loss <= 0.5:
        return "green"
    elif loss <= 0.75:
        return "yellow"
    else:
        return "red"