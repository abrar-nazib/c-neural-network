import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn")
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)

data = pd.read_csv('data.csv')

ages = data['Age']
dev_salaries = data['All_Devs']
py_salaries = data['Python']
js_salaries = data['JavaScript']


# plt.plot(ages, py_salaries, label="Python")
ax2.plot(ages, py_salaries, label="Python")
# plt.plot(ages, js_salaries, label="JavaScript")
ax2.plot(ages, js_salaries, label="JavaScript")

# plt.plot(ages, dev_salaries, linestyle="--", label="All Devs", color="#444444")
ax1.plot(ages, dev_salaries, linestyle="--", label="All Devs", color="#444444")

# plt.title("Median Salary($) by Age")
ax1.set_title("Median Salary($) by Age")
# plt.xlabel("Age")
ax1.set_xlabel("Age")
# plt.ylabel("Median Salaries($)")
ax1 .set_ylabel("Median Salaries($)")
# plt.legend()
ax1.legend()

# plt.plot(ages, dev_salaries, linestyle="--", label="All Devs", color="#444444")
ax1.plot(ages, dev_salaries, linestyle="--", label="All Devs", color="#444444")

# plt.title("Median Salary($) by Age")
ax2.set_title("Median Salary($) by Age")
# plt.xlabel("Age")
ax2.set_xlabel("Age")
# plt.ylabel("Median Salaries($)")
ax2 .set_ylabel("Median Salaries($)")
# plt.legend()
ax2.legend()

plt.tight_layout()

plt.show()
