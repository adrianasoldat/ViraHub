# ViraHub

VIRAHUB is an interactive platform designed to help small businesses in Ukraine connect with investors, mentors, and strategic partners. This repository contains Streamlit-based prototypes demonstrating different partner matching approaches.

---

## Option 1: Simple Rule-Based Matching

This prototype uses straightforward rules such as **industry match**, **funding availability**, and **mentorship capacity** to rank partners.

**Pros:**

- Simple and easy to understand.  
- Perfect for a quick prototype/demo.  
- Fewer dependencies, faster to run.  

**Cons:**

- Limited flexibility: can’t learn from data.  
- Doesn’t scale well with many features or complex patterns.  

**Best for:**

- Early-stage prototypes.  
- Demos for stakeholders.  
- Teaching the concept of partner matching.  

---

## Option 2: Hybrid Rule + ML Matching

This prototype combines rule-based logic with a small **ML model** (`MLPRegressor`), using features extracted from businesses and partners to improve recommendations.

**Pros:**

- More realistic for real-world applications.  
- Can combine rules + machine learning to improve recommendations.  
- Extensible: you can add more features or more sophisticated ML models.  
- Shows sophistication for demos or presentations to investors.  

**Cons:**

- More complex code.  
- Requires training the ML model (even a small one).  
- Harder to debug if something breaks.  

**Best for:**

- Advanced prototypes.  
- Demonstrating a “smart” system to stakeholders.  
- Showcasing potential for data-driven partner recommendations.  
