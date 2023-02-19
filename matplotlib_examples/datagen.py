import csv
import random
import time

x_value = 0
total_1 = 1000
total_2 = 1000

xx = ["x_value", "total_1", "total_2"]
with open("data.csv", "w") as csvFile:
    csvDictWriter = csv.DictWriter(csvFile, fieldnames=xx)
    csvDictWriter.writeheader()

while True:
    with open('data.csv', 'a') as csvFile:
        csvDictWriter = csv.DictWriter(csvFile, fieldnames=xx)
        info = {
            "x_value": x_value,
            "total_1": total_1,
            "total_2": total_2
        }
        csvDictWriter.writerow(info)
        print(info)

        x_value += 1
        total_1 = total_1 + random.randint(-5, 8)
        total_2 = total_2 + random.randint(-5, 6)
    time.sleep(1)
