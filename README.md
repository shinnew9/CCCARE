# MentalHealthAssessmentBasedonLLMSimulation
**A Web-Based Dataset Evaluation Interface**
  A lightweight web application for hosting datasets and enabling structured human evaluation through a browser-based interface.

> **TL;DR**  
> MentalHealthAssessmentBasedonLLMSimulation is a lightweight Streamlit app that hosts datasets on the web
> and enables structured human evaluation without modifying the original data.


</br>

## Why I Built This
During the winter break at Lehigh University (late December of 2025 to early January 2026), I wanted to explore how datasets could be made **more usable and interactive** beyond static files.

**This project was driven by three questions:**
- How can datasets be reviewed without opening raw files?
- How can human evaluation be made more consistent?
- Does every data-facing system need a chat interface?

MentalHealthAssessmentBasedonLLMSimulation evolved into a **dataset-first evaluation platform**, focusing on structured scoring rather than conversational interaction.


## What This Project Is (and Is Not)
**This project is:**
- A web application that displays dataset entries
- A structured interface for scoring and evaluating data
- A foundation for human-in-the-loop evaluation workflows

**This project is not:**
- A chatbot
- A conversational AI interface
- A generation-focused application

This distinction strongly influenced both the system design and the UI.

## Core Idea
Static datasets are difficult to evaluate consistently.
By placing the dataset inside a web interface and guiding users through evaluation and scoring,
this project explores how **interface design can improve data quality and usability.**

## How It Works 

- Dataset entries are loaded and rendered dynamically on a web page
- Users can browse individual data samples
- Each entry can be scored or evaluated through predefined input components
- Evaluation results are stored separately from the raw dataset
This separation keeps the dataset immutable while allowing flexible evaluation logic.


## 🧩 Architecture (Simple)
    flowchart LR
      A[Dataset Files<br/>(JSON/CSV)] --> B[Streamlit App<br/>(UI)]
      B --> C[Dataset Viewer<br/>(Browse Entries)]
      C --> D[Evaluation Form<br/>(Score / Notes)]
      D --> E[Stored Outputs<br/>(CSV/JSON)]
      E --> F[Export / Analysis<br/>(Optional)]



## Design Decisions That Shaped the Project
### Dataset-first approach
  - The dataset itself is the central object, not user interaction or generation.

### Evaluation over conversation
  - Structured scoring was prioritized over free-form interaction.

### Web-based delivery
  - A browser interface lowers the barrier for participation and testing.

### Modularity
  - Dataset rendering and evaluation logic are decoupled for extensibility.

</br>

## What I Learned
  - Building evaluation tools requires a different mindset than building interactive apps
  - UI constraints directly influence the quality and consistency of human judgments
  - Not every data-driven system benefits from a conversational interface
  - Web interfaces can play a critical role in human-in-the-loop workflows

## Current State
  - Functional dataset visualization
  - Entry-level evaluation and scoring interface
  - Suitable for iterative experimentation and future extension

## What Comes Next
  - Support multiple evaluators and score aggregation
  - Add richer evaluation schemas
  - Export evaluation results for downstream analysis
  - Improve UI clarity based on user feedback

## Technologies
  - Python
  - Web framework (Flask / FastAPI)
  - HTML, CSS, JavaScript
  - JSON-based dataset handling

## Screenshots
<img width="2880" height="1546" alt="Image" src="https://github.com/user-attachments/assets/de8deda8-2e02-461f-85d2-54be94306e75" />




## 🚀 Live Demo
Try the app here: [MentalHealthAssessmentBasedonLLMSimulation](https://mentalhealthassessmentbasedonllm.streamlit.app/)
