# BigData
Time-Series Classification

# Milestones
* Interim milestone - poster & discussion - 24 Mar - 3 credits
* Final submission of Report, Code, and contribution statements - 6 May - 9 credits
* Final presentation - 12 May - 3 credits

# Crucially
* A project that designs and implements a well-thought-out distributed solution for a relatively straightforward method/technique will receive a higher mark

## Environment Setup

### 1. Clone the repository :
git clone https://github.com/YourUser/BigData.git
cd BigData

###  2. Create the environment using environment.yml:
conda env create -f environment.yml

###  3. Activate the environment:
conda activate bigdata_env

###  4. Install packages from requirements.txt - if needed:
conda install --file requirements.txt

### 5. Configure Spark to Use the Conda Environment (Windows)


* To ensure Spark uses your custom Python interpreter from your 'conda environment':

* Set System Environment Variables:

Open System Properties → Advanced system settings → Environment Variables.
Under User variables or System variables, add or update:

```bash
- SPARK_HOME = [Your path]
- PYSPARK_PYTHON = [Your path]
- PYSPARK_DRIVER_PYTHON = [Your path]
```

* Restart Your Terminal/IDE:
Close and reopen your terminal or VS Code so the new settings take effect.

#####Verify the Configuration:

- Run a simple Spark script (e.g., create a file test_spark.py with the following content):
```bash
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").appName("EnvTest").getOrCreate()
print("Spark is using:", spark.sparkContext.pythonExec)
spark.stop()
```

- Execute the script and verify that the output points to:

```bash
[Yourpath]
```

####  5. Run main.py scripts:
python code/src/main.py
