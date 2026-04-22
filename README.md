# Distributed Time Series Classification with Spark

## Overview

This project implements a distributed pipeline for time series classification using PySpark, focusing on scalability, partition-aware learning, and distance-based models.

The system supports:
- **Global models** trained across the full dataset  
- **Local models** trained independently per data partition  

The goal is to analyse trade-offs between predictive performance, runtime, and scalability.

Experiments are conducted on the ECG5000 dataset.

---

## Key Features

- End-to-end Spark pipeline (ingestion → preprocessing → training → evaluation)
- Partition-aware training (local vs global models)
- Distance-based classification:
  - Dynamic Time Warping (DTW)
  - Euclidean
  - Manhattan
  - Cosine
- Integration with `aeon` (Proximity Tree / Forest)
- Configurable sampling and partitioning
- Detailed logging:
  - performance metrics
  - runtime
  - memory usage
  - model complexity
- Unit tests for core components

---

## Project Structure
        code/
        ├── src/
        │   ├── main.py
        │   ├── config.py
        │   ├── data_ingestion.py
        │   ├── preprocessing.py
        │   ├── distance_measures.py
        │   ├── local_model_manager.py
        │   ├── prediction_manager.py
        │   ├── utilities.py
        │   ├── visualization.py
        │   └── models_global/
        │
        ├── tests/
        │
        logs/
        ├── img/
        ├── archive/
        │
        models_local/
        models_global/


---

## Pipeline

Run the pipeline:

```bash
python code/src/main.py

---
### Steps

1. **Data Ingestion**
   - Loads CSV into Spark DataFrame  
   - Applies schema and optional sampling  

2. **Preprocessing**
   - Removes fully null rows  
   - Min-max normalisation of features  

3. **Train/Test Split**
   - Random or stratified  

4. **Model Training**
   - **Global:** trained on full dataset  
   - **Local:** trained per partition  

5. **Distance Computation**
   - DTW (exact / approximate)  
   - Euclidean / Manhattan / Cosine  

6. **Prediction & Evaluation**
   - Accuracy, precision, recall, F1  
   - Balanced accuracy  

7. **Logging**
   - JSON experiment reports  
   - Runtime and scaling metrics  

---

## Models

### Global Model
- Single model across all data  
- Higher computational cost  
- More stable performance  

### Local Models
- One model per partition  
- Faster and parallelisable  
- Sensitive to partition quality  

---
