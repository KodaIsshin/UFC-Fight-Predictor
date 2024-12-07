new_accuracy = 77.73
# Open the file in read mode to fetch the current accuracy
with open("Current Accuracy.txt", "r") as read_list:
    accuracy = float(read_list.read().strip())

# Compare and overwrite if the new accuracy is better
if new_accuracy >= accuracy:
    with open("Current Accuracy.txt", "w") as write_list:
        write_list.write(f"{new_accuracy:.2f}")