## Setup and Usage

### Docker

To run the application using Docker, follow these steps:

1. Ensure Docker is installed on your machine.
2. Build the Docker image using the provided Dockerfile:
    ```
    docker build -t project-name .
    ```
3. Once the image is built, run the container:
    ```
    docker run -p 8080:8080 project-name
    ```
4. Access the application via a web browser at `http://localhost:8080`.

### Python Virtual Environment

To run the application using a Python virtual environment:

1. For macOS, create a virtual environment:
    ```
    python -m venv .venv
    source .venv/bin/activate
    ```
2. Install the required Python packages:
    ```
    pip install -r requirements.txt
    ```
3. Run the application using Streamlit:
    ```
    streamlit run segmentation.py
    ```

## File Structure

- `segmentation.py`: Main application file.
- `requirements.txt`: List of required Python packages.
- `Dockerfile`: Configuration file for building the Docker image.
- `README.md`: Documentation file (you're currently reading it!).
- `config.yaml`: Configuration of model urls and images, see example, the required files can be created with the other notebooks

