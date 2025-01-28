# OutBox AI

## Overview

OutBox AI is an innovative, AI-powered medical diagnostics assistant designed to provide accessible and real-time health insights. The platform leverages advanced machine learning models to assist with diagnosing various medical conditions, including diabetes, heart disease, and Parkinson's disease. Additionally, it offers an AI medical assistant to analyze symptoms and medical images, making it an indispensable tool for healthcare providers in remote and underdeveloped areas.

## Features

### 1. **Multi-Disease Detection**

OutBox AI incorporates specialized machine learning models for diagnosing:

- **Diabetes**: Predicts diabetes risk based on user input.
- **Heart Disease**: Assesses the likelihood of heart disease using clinical parameters.
- **Parkinson's Disease**: Detects potential Parkinson's indicators from user-provided data.

### 2. **AI Medical Assistant**

The assistant can:

- Analyze symptoms and medical images in real-time.
- Provide diagnostic insights and recommendations.
- Convert diagnostic results into audio using text-to-speech functionality.

### 3. **Offline Compatibility**

OutBox AI is optimized for low-latency, edge-device usage, ensuring accessibility in areas with minimal internet connectivity.

## Problem Statement

In remote and underdeveloped regions, access to specialized healthcare services is limited. Delayed diagnostics and insufficient resources exacerbate medical conditions. OutBox AI aims to bridge this gap by offering reliable, AI-powered diagnostic tools that function offline, enabling healthcare providers to deliver timely care.

## Solution

OutBox AI uses AI/ML technologies to:

- Empower healthcare workers in remote areas.
- Provide instant diagnostic insights for multiple diseases.
- Analyze multimodal data, including text and medical images, without relying on cloud-based infrastructure.

## Installation

1. Clone the repository:
   ```bash
   git clone <https://github.com/M-Hamza-Hassaan/Ai-PMDA.git>
   ```
2. Navigate to the project directory:
   ```bash
   cd <Ai-PMDA>
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file in the project root.
   - Add the following variables:
     ```
     GROQ_API_TOKEN=<your_groq_api_token>
     ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Launch the application by running the above command.
2. Use the sidebar to navigate between different features:
   - **Home**: Learn about the project and its purpose.
   - **Diabetes Prediction**: Enter health parameters to predict diabetes risk.
   - **Heart Disease Prediction**: Provide clinical data to evaluate heart disease likelihood.
   - **Parkinson's Prediction**: Input diagnostic values to detect potential Parkinson's indicators.
   - **AI Medical Assistant**: Upload symptoms or medical images for real-time analysis.

## Dependencies

- **Python Libraries**: `streamlit`, `streamlit_option_menu`, `gtts`, `dotenv`, `Pillow`, `groq`
- **Machine Learning Models**: Pre-trained `.sav` files for diabetes, heart disease, and Parkinson's predictions.

## Folder Structure

```
project-directory/
|-- app.py                 # Main application file
|-- requirements.txt       # Python dependencies
|-- saved_models/          # Directory containing .sav files for ML models
|-- .env                   # Environment variables file
```

## Note

- The AI-powered insights provided by this application are designed to assist healthcare providers. They should not replace professional medical advice, diagnosis, or treatment.

## Contributing

1. Fork the repository.
2. Create a new branch for your feature/bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any inquiries or feedback, please contact [M. Hamza Hassaan](https://www.linkedin.com/in/muhammad-hamza-hassaan/).

