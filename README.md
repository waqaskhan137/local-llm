# Ollama Docker Compose Setup

Welcome to the Ollama Docker Compose Setup! This project simplifies the deployment of Ollama using Docker Compose, making it easy to run Ollama with all its dependencies in a containerized environment.

## Getting Started

### Prerequisites

Make sure you have the following prerequisites installed on your machine:

- Docker
- Docker Compose

#### GPU Support (Optional)

If you have a GPU and want to leverage its power within a Docker container, follow these steps to install the NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure NVIDIA Container Toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU integration
docker run --gpus all nvidia/cuda:11.5.2-base-ubuntu20.04 nvidia-smi
```

### Configuration

1. Clone the Docker Compose repository:

    ```bash
    git clone https://github.com/waqaskhan137/ollama-docker.git
    ```

2. Change to the project directory:

    ```bash
    cd ollama-docker
    ```

## Usage

Start Ollama and its dependencies using Docker Compose:

if gpu is configured

```bash
docker-compose -f docker-compose-ollama-gpu.yaml up -d
```

else

```bash
docker-compose up -d
```

Visit [http://localhost:8000](http://localhost:8000) in your browser to access Ollama-webui.

### Model Installation

Navigate to settings -> model and install a model (e.g., llama2). This may take a couple of minutes depends on your internet speed, but afterward, you can use it just like ChatGPT.

I will list some famous opensource models

- gemma
  - Gemma is a family of lightweight, state-of-the-art open models built by Google DeepMind.

- llama2
  - Llama 2 is a collection of foundation language models ranging from 7B to 70B parameters.

- mistral
  - The 7B model released by Mistral AI, updated to version 0.2.

- mixtral
  - A high-quality Mixture of Experts (MoE) model with open weights by Mistral AI.

- llava
  - 🌋 LLaVA is a novel end-to-end trained large multimodal model that combines a vision encoder and Vicuna for general-purpose visual and language understanding. Updated to version 1.6.

- neural-chat
  - A fine-tuned model based on Mistral with good coverage of domain and language.

- codellama
  - A large language model that can use text prompts to generate and discuss code.

If you want to explore more you can find those [here](https://ollama.com/library)

### Explore Langchain and Ollama

You can explore Langchain and Ollama within the project. A third container named **app** has been created for this purpose. Inside, you'll find some examples.

## Stop and Cleanup

To stop the containers and remove the network:

```bash
docker-compose down
```

## Contributing

We welcome contributions! If you'd like to contribute to the Ollama Docker Compose Setup, please follow our [Contribution Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it according to the terms of the license. Just give me a mention and some credit

## Contact

If you have any questions or concerns, please contact us at [waqaskhan137@gmail.com](mailto:waqaskhan137@gmail.com).

Enjoy using Ollama with Docker Compose! 🐳🚀
